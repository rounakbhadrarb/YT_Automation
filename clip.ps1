# Usage:  .\clip.ps1 "<video URL or file>" [--clips 5] [--layout crop|blur] [--manual "30-75,120-160"]
& "$env:USERPROFILE\.venvs\clipbot\Scripts\python.exe" "$PSScriptRoot\clipbot.py" @args
