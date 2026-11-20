"""clipbot - turn a long video into ready-to-post vertical Shorts/Reels.

Pipeline:  download (yt-dlp) -> transcribe (faster-whisper, GPU)
           -> pick best moments (Claude) -> cut + 9:16 reframe + animated captions (ffmpeg)

Only use videos you have rights to clip (e.g. clipping-campaign source videos).

Examples:
  python clipbot.py "https://www.youtube.com/watch?v=XXXX" --clips 5
  python clipbot.py C:\\videos\\podcast.mp4 --layout blur
  python clipbot.py video.mp4 --manual "30-75,120-160"     (skip AI picking)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WINGET_LINKS = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"


# ---------------------------------------------------------------- helpers

def _registry_path() -> str:
    """PATH as stored in the registry (picks up installs made after this terminal opened)."""
    if os.name != "nt":
        return ""
    import winreg
    parts = []
    for root, sub in ((winreg.HKEY_CURRENT_USER, r"Environment"),
                      (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
        try:
            with winreg.OpenKey(root, sub) as k:
                parts.append(os.path.expandvars(winreg.QueryValueEx(k, "Path")[0]))
        except OSError:
            pass
    return os.pathsep.join(parts)


def find_exe(name: str) -> str:
    exe = (shutil.which(name) or shutil.which(name, path=str(WINGET_LINKS))
           or shutil.which(name, path=_registry_path()))
    if not exe:
        sys.exit(f"[error] '{name}' not found. Install it (winget) or restart your terminal.")
    return exe


def run(cmd: list[str], cwd: Path | None = None) -> None:
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd[:3])} ...\n{res.stderr[-2000:]}")


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:n] or "clip"


def enable_cuda_dlls() -> None:
    """faster-whisper on Windows needs the pip-installed NVIDIA DLLs on the search path."""
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    for base in nvidia.__path__:
        for sub in ("cublas", "cudnn"):
            d = Path(base) / sub / "bin"
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ["PATH"]


# ---------------------------------------------------------------- 1. download

def get_source(src: str, work: Path) -> Path:
    if Path(src).exists():
        return Path(src).resolve()
    out = work / "source.mp4"
    if out.exists():
        print("[1/4] source already downloaded")
        return out
    print("[1/4] downloading ...")
    run([find_exe("yt-dlp"), "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
         "--merge-output-format", "mp4", "-o", str(out), src])
    return out


# ---------------------------------------------------------------- 2. transcribe

def transcribe(video: Path, work: Path, model_name: str, lang: str | None) -> list[dict]:
    cache = work / "transcript.json"
    if cache.exists():
        print("[2/4] transcript cached")
        return json.loads(cache.read_text(encoding="utf-8-sig"))

    # Pick the engine BEFORE importing either: importing faster-whisper loads a cuDNN
    # that crashes PyTorch, so the two can't both be loaded in one process.
    engine = "faster" if _huggingface_ok() else "openai"
    print(f"[2/4] transcribing with whisper '{model_name}' ({engine}-whisper) ...")
    if engine == "faster":
        out, language = _transcribe_faster(video, model_name, lang)
    else:
        out, language = _transcribe_openai(video, model_name, lang)
    cache.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"      language: {language}, segments: {len(out)}")
    return out


def _huggingface_ok() -> bool:
    """faster-whisper downloads its models from Hugging Face; some networks block it."""
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request("https://huggingface.co/api/models/Systran/faster-whisper-small",
                                                      method="HEAD"), timeout=6)
        return True
    except Exception:
        return False


def _seg(start: float, end: float, text: str, words: list[tuple[float, float, str]]) -> dict:
    print(f"      {end:7.1f}s  {text.strip()[:70]}")
    return {"start": round(start, 2), "end": round(end, 2), "text": text.strip(),
            "words": [{"s": round(a, 2), "e": round(b, 2), "w": w.strip()} for a, b, w in words]}


def _transcribe_faster(video: Path, model_name: str, lang: str | None):
    """Fast path: CTranslate2 models from Hugging Face."""
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    path = download_model(model_name)  # fails fast if Hugging Face is unreachable
    enable_cuda_dlls()  # only now: these DLLs clash with PyTorch's cuDNN in the fallback path
    try:
        model = WhisperModel(path, device="cuda", compute_type="float16")
    except RuntimeError:  # no usable GPU
        model = WhisperModel(path, device="cpu", compute_type="int8")
    segs, info = model.transcribe(str(video), language=lang, word_timestamps=True, vad_filter=True)
    return [_seg(s.start, s.end, s.text, [(w.start, w.end, w.word) for w in (s.words or [])]) for s in segs], info.language


def _transcribe_openai(video: Path, model_name: str, lang: str | None):
    """Fallback: OpenAI's reference Whisper (models from openaipublic.azureedge.net)."""
    import torch
    import whisper
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    res = model.transcribe(str(video), language=lang, word_timestamps=True, fp16=(device == "cuda"))
    segs = [_seg(s["start"], s["end"], s["text"], [(w["start"], w["end"], w["word"]) for w in s.get("words", [])])
            for s in res["segments"]]
    return segs, res.get("language")


# ---------------------------------------------------------------- 3. pick moments

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "hook_text": {"type": "string"},
                    "post_caption": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "score": {"type": "integer"},
                    "why": {"type": "string"},
                },
                "required": ["start", "end", "title", "hook_text", "post_caption", "hashtags", "score", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

PICK_PROMPT = """You are an expert short-form video editor who makes clips that go viral on YouTube Shorts, Instagram Reels and TikTok.

Below is a timestamped transcript of a long video. Pick the {n} best moments to turn into vertical clips.

Rules for every clip:
- Length between {mn} and {mx} seconds.
- It must START on a strong hook: a bold claim, surprising fact, question, conflict, or emotional moment. Never start mid-sentence or with filler ("so", "um", "yeah").
- It must make sense on its own, with no context from the rest of the video.
- It must END at a natural sentence boundary, ideally on a payoff or punchline.
- Clips must not overlap.
- Use the exact start/end times from the transcript lines.

For each clip also write:
- title: short internal name.
- hook_text: on-screen text for the first 3 seconds, max 7 words, curiosity-driving, no emojis.
- post_caption: the caption to post with the clip, 1-2 sentences.
- hashtags: 3-5 relevant hashtags without the # sign.
- score: 1-100, how likely it is to perform well.
- why: one sentence on why this moment works.

Order clips from highest to lowest score.

TRANSCRIPT:
{transcript}"""


def pick_clips_ai(segments: list[dict], n: int, mn: int, mx: int, work: Path) -> list[dict]:
    cache = work / "picks.json"
    if cache.exists():
        print("[3/4] clip picks cached (delete picks.json to re-pick)")
        return json.loads(cache.read_text(encoding="utf-8-sig"))

    print("[3/4] asking Claude for the best moments ...")
    import anthropic

    lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": CLIP_SCHEMA}},
        messages=[{"role": "user", "content": PICK_PROMPT.format(n=n, mn=mn, mx=mx, transcript=lines)}],
    )
    if resp.stop_reason == "refusal":
        sys.exit("[error] Claude declined to process this transcript.")
    text = next(b.text for b in resp.content if b.type == "text")
    clips = json.loads(text)["clips"]
    cache.write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    u = resp.usage
    print(f"      picked {len(clips)} clips (tokens in {u.input_tokens}, out {u.output_tokens})")
    return clips


def manual_clips(spec: str) -> list[dict]:
    clips = []
    for i, part in enumerate(spec.split(","), 1):
        a, b = (float(x) for x in part.split("-"))
        clips.append({"start": a, "end": b, "title": f"clip {i}", "hook_text": "", "post_caption": "",
                      "hashtags": [], "score": 0, "why": "manual"})
    return clips


def snap(clip: dict, segments: list[dict]) -> tuple[float, float]:
    """Snap start/end to the nearest segment boundaries so we never cut mid-word."""
    starts = [s["start"] for s in segments] or [clip["start"]]
    ends = [s["end"] for s in segments] or [clip["end"]]
    a = min(starts, key=lambda t: abs(t - clip["start"]))
    b = min(ends, key=lambda t: abs(t - clip["end"]))
    if b <= a:
        a, b = clip["start"], clip["end"]
    return max(0.0, a - 0.15), b + 0.35


# ---------------------------------------------------------------- 4. render

def ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_escape(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def build_ass(segments: list[dict], a: float, b: float, hook: str, path: Path,
              words_per_line: int = 3, highlight: str = "&H0000E5FF&") -> None:
    # Colors are &HBBGGRR. Default highlight = yellow.
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,3,2,60,60,560,1
Style: Hook,Arial Black,58,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,1,0,0,0,100,100,0,0,3,20,0,8,120,120,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if hook:
        events.append(f"Dialogue: 1,{ass_time(0)},{ass_time(3.0)},Hook,,0,0,0,,{{\\fad(150,200)}}{ass_escape(hook.upper())}")

    words = [w for s in segments for w in s["words"] if w["s"] >= a - 0.05 and w["e"] <= b + 0.05 and w["w"]]

    # Group words into short lines; never let a line run across a sentence end.
    groups, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == words_per_line or w["w"][-1:] in ".?!":
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)

    for gi, group in enumerate(groups):
        next_group_start = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else group[-1]["e"] + 0.8
        for i, w in enumerate(group):
            start = w["s"] - a
            if i + 1 < len(group):
                end = group[i + 1]["s"] - a
            else:  # last word: linger briefly, but never overlap the next line
                end = min(next_group_start, w["e"] + 0.8) - a
            parts = []
            for j, x in enumerate(group):
                t = ass_escape(x["w"].upper())
                parts.append(f"{{\\c{highlight}\\fscx108\\fscy108}}{t}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}" if j == i else t)
            events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{' '.join(parts)}")
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def has_nvenc(ffmpeg: str) -> bool:
    res = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True)
    return "h264_nvenc" in res.stdout


def render(video: Path, a: float, b: float, ass: Path, out: Path, layout: str, ffmpeg: str, nvenc: bool) -> None:
    if layout == "blur":
        vf = ("split[bg][fg];"
              "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=25:2[bg2];"
              "[fg]scale=1080:-2[fg2];[bg2][fg2]overlay=(W-w)/2:(H-h)/2,"
              f"subtitles={ass.name}")
    else:  # center crop
        vf = f"crop='min(iw,ih*9/16)':ih,scale=1080:1920,setsar=1,subtitles={ass.name}"

    enc = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", "-b:v", "0"] if nvenc
           else ["-c:v", "libx264", "-preset", "medium", "-crf", "20"])
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{a:.2f}", "-to", f"{b:.2f}", "-i", str(video),
           "-filter_complex" if layout == "blur" else "-vf", vf,
           *enc, "-pix_fmt", "yuv420p", "-r", "30",
           "-c:a", "aac", "-b:a", "160k", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
           "-movflags", "+faststart", out.name]
    run(cmd, cwd=out.parent)  # cwd = clip dir so the subtitles path needs no escaping


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="Long video -> vertical clips with captions")
    p.add_argument("source", help="YouTube/video URL or local video file")
    p.add_argument("--clips", type=int, default=5, help="number of clips to make (default 5)")
    p.add_argument("--min", type=int, default=20, help="min clip length, seconds")
    p.add_argument("--max", type=int, default=60, help="max clip length, seconds")
    p.add_argument("--layout", choices=["crop", "blur"], default="crop",
                   help="crop = fill screen (podcasts, talking heads); blur = full frame on blurred bg")
    p.add_argument("--whisper", default="small", help="whisper model: base / small / medium / large-v3")
    p.add_argument("--lang", default=None, help="force language code, e.g. en, hi (default: auto)")
    p.add_argument("--manual", default=None, help='skip AI, use your own ranges: "30-75,120-160"')
    p.add_argument("--out", default=str(HERE / "output"), help="output folder")
    args = p.parse_args()

    ffmpeg = find_exe("ffmpeg")
    os.environ["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + os.environ["PATH"]  # whisper calls "ffmpeg"
    key = re.sub(r"[^a-zA-Z0-9_-]", "_", Path(args.source).stem if Path(args.source).exists()
                 else (re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{6,})", args.source) or [None, args.source[-20:]])[1])
    work = Path(args.out) / key
    work.mkdir(parents=True, exist_ok=True)

    video = get_source(args.source, work)
    segments = transcribe(video, work, args.whisper, args.lang)
    clips = manual_clips(args.manual) if args.manual else pick_clips_ai(segments, args.clips, args.min, args.max, work)

    print(f"[4/4] rendering {len(clips)} clips ({args.layout}) ...")
    nvenc = has_nvenc(ffmpeg)
    clip_dir = work / "clips"
    clip_dir.mkdir(exist_ok=True)
    report = []
    for i, c in enumerate(clips, 1):
        a, b = snap(c, segments)
        name = f"{i:02d}_{slug(c['title'])}"
        ass = clip_dir / f"{name}.ass"
        out = clip_dir / f"{name}.mp4"
        build_ass(segments, a, b, c.get("hook_text", ""), ass)
        try:
            render(video, a, b, ass, out, args.layout, ffmpeg, nvenc)
        except RuntimeError as e:
            if nvenc:  # retry on CPU encoder
                render(video, a, b, ass, out, args.layout, ffmpeg, False)
            else:
                raise e
        ass.unlink(missing_ok=True)
        tags = " ".join("#" + t.lstrip("#") for t in c.get("hashtags", []))
        report.append({"file": out.name, "start": round(a, 1), "end": round(b, 1), "seconds": round(b - a, 1),
                       "score": c.get("score"), "hook": c.get("hook_text"),
                       "caption": f"{c.get('post_caption', '')} {tags}".strip(), "why": c.get("why")})
        print(f"      [ok] {out.name}  ({b - a:.0f}s, score {c.get('score')})")

    (clip_dir / "posting_sheet.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(clip_dir / "posting_sheet.txt", "w", encoding="utf-8") as f:
        for r in report:
            f.write(f"{r['file']}  |  {r['seconds']}s  |  score {r['score']}\n  HOOK: {r['hook']}\n"
                    f"  CAPTION: {r['caption']}\n  WHY: {r['why']}\n\n")
    print(f"\nDone -> {clip_dir}")


if __name__ == "__main__":
    main()
