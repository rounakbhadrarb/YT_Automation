# Clipbot: clipping-campaign starter kit

Turns one long video into 5+ ready-to-post vertical clips with animated captions, a hook title, and a posting sheet (caption + hashtags) for each.

```
long video ──► download ──► transcribe (Whisper, GPU) ──► Claude picks best moments ──► cut + 9:16 + captions
```

---

## 1. One-time setup (≈10 min)

0. **Install the tools** (Windows, NVIDIA GPU recommended):
   ```powershell
   winget install -e --id Python.Python.3.12 --scope user
   winget install -e --id Gyan.FFmpeg
   winget install -e --id yt-dlp.yt-dlp
   # keep the venv on a short path (Windows long-path limit)
   & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv "$env:USERPROFILE\.venvs\clipbot"
   $py = "$env:USERPROFILE\.venvs\clipbot\Scripts\python.exe"
   & $py -m pip install torch --index-url https://download.pytorch.org/whl/cu124
   & $py -m pip install -r requirements.txt
   ```
   Transcription uses faster-whisper when Hugging Face is reachable, and otherwise OpenAI's Whisper. The first run downloads a ~460 MB model.
1. **Anthropic API key.** Create one at console.anthropic.com → API Keys. Add $5 of credit; one video costs about $0.02–0.10. Then run this in PowerShell:
   ```powershell
   setx ANTHROPIC_API_KEY "sk-ant-..."
   ```
   Open a new terminal afterwards so it takes effect.
2. **Accounts to post from.** Create a fresh YouTube channel, Instagram account and TikTok account, all with the same handle. Pick a niche-neutral name such as "ClipVault" or "DailyClipsHQ".
3. **Campaign platform.** Sign up on **Whop**, then go to Discover → **Content Rewards**. Also look for creators who run their own clipping programs, usually linked in their Discord or YouTube description.

## 2. Daily workflow (≈45–60 min/day)

| Step | Who | What |
|---|---|---|
| 1 | You | Open Content Rewards. Pick 1–2 campaigns with a good rate (≥ $1 per 1k views), a budget still left, and content you'd actually watch. |
| 2 | You | Read the campaign rules: required hashtags or tags, min/max length, allowed platforms, and whether captions or watermarks are required. |
| 3 | **Clipbot** | `.\clip.ps1 "<source video link from the campaign>" --clips 6` |
| 4 | You | Watch every clip (2 min). Delete weak ones and fix any wrong captions. |
| 5 | You | Post 3–5 clips per platform per day, using the caption from `posting_sheet.txt` plus the campaign's required tags. Space posts 2–3 hours apart. |
| 6 | You | Submit each post link in the campaign dashboard. |
| 7 | You | Once a week, record views per clip in a sheet. Note which hooks and lengths win. |

## 3. Commands

```powershell
.\clip.ps1 "https://www.youtube.com/watch?v=XXXX"             # 5 AI-picked clips, center crop
.\clip.ps1 "https://..." --clips 8 --min 15 --max 45           # more, shorter clips
.\clip.ps1 "https://..." --layout blur                         # whole frame on blurred background (gameplay, wide shots)
.\clip.ps1 "C:\path\video.mp4" --whisper medium                # better transcription (slower)
.\clip.ps1 "C:\path\video.mp4" --lang hi                       # force Hindi
.\clip.ps1 "C:\path\video.mp4" --manual "62-110,300-345"       # your own time ranges, no AI
```

Output goes to `output\<video-id>\clips\`: the clips as `.mp4`, plus `posting_sheet.txt`.
To re-pick clips for the same video, delete `picks.json` in that folder. The transcript is cached, so it doesn't run again.

## 4. Rules that protect your income

- **Only clip campaign videos or content you have permission to use.** Random famous videos lead to copyright strikes and zero payout.
- **Follow each campaign's rules exactly.** Rejected submissions don't get paid.
- **Post manually, or with the platforms' own schedulers** (YouTube Studio, Meta Business Suite, TikTok's scheduler). Avoid bot-upload tools and never buy views. Campaigns check for both and ban accounts.
- **Review every clip before posting.** That review is the "human in the loop" that keeps the accounts healthy.

## 5. What to learn in months 0–3 (feeds your future channel)

Track these per clip: **hook text, length, first-3-second visual, views after 48h, retention %**.
By month 3 you'll have real data on which hooks work. Reuse those findings when you launch the AI-tools channel.
