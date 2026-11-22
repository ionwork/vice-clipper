# vice-clipper

Local pipeline that turns GTA 6 streams and videos into 9:16 clips. No LLM agent at runtime.

## Flow
sources (recorder | harvester | vod_scout) -> Candidate -> judge (local Ollama vision model via frames+transcript+loudness, or Gemini; 1 call, JSON)
-> captions (faster-whisper -> ASS) -> render (ffmpeg) -> deliver (Telegram)

## Rules
- Python 3.12, asyncio, type hints everywhere, Pydantic v2 models in src/vc/models.py
- All tunables in config.yaml, all secrets in .env, never hard-code either
- Every stage persists state in SQLite (table clips, column status) and must be resumable
- Subprocesses (ffmpeg, streamlink, yt-dlp) via asyncio.create_subprocess_exec, never shell=True
- Paths via pathlib, must work on Windows and macOS
- No generated content: only cut, caption and mix existing footage
- Each module gets a CLI entry in src/vc/cli.py (Typer) and at least one pytest test
- Keep modules small; do not refactor other modules unless the task says so

## Clip status
NEW -> JUDGED -> CAPTIONED -> RENDERED -> DELIVERED -> POSTED | REJECTED | FAILED

## Commands
uv run vc setup         guided key setup (.env, live checks, Telegram chat id)
uv run vc launch        switch to GTA 6 mode (--undo = rehearsal with GTA V)
uv run vc --help        all commands
uv run pytest -q        tests
