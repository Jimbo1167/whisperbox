# Skill: `transcribe-locally` — design

- **Date:** 2026-05-07
- **Status:** Approved (design); implementation pending

## Goal

Expose a Claude Code user-scope skill that, when the user asks Claude to transcribe a video or audio file on their local machine, makes Claude actually run the transcription via the `whisperbox` project — without the user having to walk Claude through the steps each time.

## Non-goals

- Setting up the project (no auto `make setup`).
- Editing `.env` or installing the HuggingFace token.
- Stopping a model server that the skill started.
- Deleting or rearranging files in `transcripts/`.
- Handling cloud/remote transcription paths.

## Skill location and wiring

- Path: `~/projects/skills/skills/transcribe-locally/SKILL.md`.
- Picked up automatically because `~/projects/skills/skills/` is symlinked to `~/.claude/skills/`. No per-skill wiring needed.
- Single file: `SKILL.md`. No supporting files.

## Frontmatter

```yaml
---
name: transcribe-locally
description: "Use when the user asks to transcribe a video or audio file on their local machine (mp4, mov, m4v, wav, mp3, m4a, aac). Routes through Jim's whisperbox project at ~/projects/whisperbox — prefers the warm model server on :8000 when running, auto-starts it if not. Speaker diarization is opt-in: only enabled when the user explicitly asks for speaker labels."
---
```

The description value is double-quoted because it contains a `: ` sequence (`opt-in: only enabled`) that strict YAML parsers (e.g., GitHub's) interpret as a nested mapping. Claude Code's loader is more permissive, but quoting keeps the file portable.

The `description` is the only trigger Claude reads when scanning available skills. Phrases the skill should fire on: "transcribe this video", "transcribe foo.mp4", "get a transcript of …", "what did they say in this recording", "who said what in this interview".

## Behavior

### 1. Project pin and path resolution

Project lives at `~/projects/whisperbox`. If `~/projects/whisperbox` does not exist on disk, stop and report that the project is missing.

Order matters: **resolve the user's input file to an absolute path *before* `cd`-ing into the project.** Otherwise a relative path like `./meeting.mp4` from the user's current directory will be silently re-resolved against the project root and fail. Concretely: capture `realpath` (or equivalent) of the user-provided path first, then `cd ~/projects/whisperbox` for the rest of the run.

The pin is explicit in the skill body, so this is not a silent assumption — it's the documented contract of the skill.

### 2. Preflight (one Bash call, fail fast)

In order:

1. `venv/` directory exists. If not → stop, surface `make setup`.
2. `.env` file exists. If not → stop, surface `cp .env.example .env`.
3. The input file (resolved to absolute path per §1) exists, and its extension is one of `mp4, mov, m4v, wav, mp3, m4a, aac`. If not → stop, surface what was wrong.
4. If diarization is on for this run: `.env` contains a non-placeholder `HF_TOKEN=` line (i.e. not `your_huggingface_token_here`). If missing → stop, tell user to set HF_TOKEN.

When preflight fails, the skill reports the issue and the fix command. It never runs setup or token-installation steps itself.

### 3. Diarization decision

Off by default. Turn on **only when the user explicitly asks for speaker information** — e.g. "diarize", "with speakers", "label the speakers", "who said what", "separate by speaker", "distinguish voices".

Words like "interview", "meeting", "call", or "conversation" are *not* sufficient on their own — those describe the recording, not what the user wants out of the transcript. Diarization is slower and requires `HF_TOKEN`, so the cost of a false positive is real (silently slower runs, or hard failures for users who haven't set the token). When in doubt, default off.

The skill body lists this rule explicitly so Claude doesn't re-derive it each invocation.

### 4. Routing: warm vs. cold

The skill calls the **Click CLI at `scripts/transcribe.py`** directly. It does **not** use the `make transcribe` / `make diarize` / `make client-transcribe` targets. This is a real divergence, not a stylistic choice — see the note below.

All commands run inside the project venv:

```bash
. venv/bin/activate
```

Probe the model server (one call, exit code is the signal — `curl -sf` returns non-zero on any non-2xx, so no need to inspect the status code):

```bash
curl -sf -m 2 http://localhost:8000/health
```

- **Exit 0** → server is warm. Run:
  ```
  python scripts/transcribe.py client transcribe "<absolute-input-path>" [--diarize]
  ```
- **Non-zero / connection refused** → start the model server in the background, **bypassing the Makefile so the backgrounded PID is the actual python process** (otherwise `$!` is `make`, and killing it doesn't stop the python child). Concretely:
  ```bash
  nohup python scripts/model_server.py --host 0.0.0.0 --port 8000 \
      > /tmp/whisperbox-server.log 2>&1 &
  server_pid=$!
  ```
  (The skill executes through Claude's Bash tool, which is bash, even though Jim's interactive shell is fish.)
  Then poll `/health` every ~2s until 2xx, capped at 60s total. Once warm, log "server is now warm at :8000" and route via the warm command above. The next invocation's preflight will find it.
  - If the 60s cap is hit without the server becoming healthy, `kill $server_pid` (and `kill -9` after a brief grace period if it doesn't exit), then fall back to a one-shot run:
    ```
    python scripts/transcribe.py transcribe "<absolute-input-path>" [--diarize]
    ```
- The skill **does not stop a server it started** when the run *succeeds*. Leaving it warm is the desired behavior.

**Note on the Makefile divergence.** This project has two parallel CLIs:

- `make transcribe` and `make diarize` invoke `transcribe_video.py`, a thin script with its own `argparse` that supports only `input_path` and `-o/--output`. It pulls model size, language, format, and diarization from the `.env` (with `INCLUDE_DIARIZATION=true` injected as an env var for `make diarize`).
- `python scripts/transcribe.py …` is a Click CLI with full options: `--diarize`, `--model`, `--language`, `--format` (including `pretty`), plus subcommands for `client`, `stream`, `batch`, `server`.

The skill bypasses `make transcribe` because `transcribe_video.py` is a separate code path that doesn't expose the flags the skill needs. The Click CLI is the only path that gives uniform `--diarize` (and other flag) handling across the cold and warm paths.

### 5. Model selection

Don't override `WHISPER_MODEL` from `.env`. Only pass `--model <size>` to the underlying CLI if the user explicitly asked for a different model size in their request (e.g. "use the small model", "do this with large-v3").

### 6. Reporting back

When the transcription completes successfully:

- Tell the user the output path (default: `transcripts/<input-stem>.<ext>`, where `<ext>` follows `OUTPUT_FORMAT` from `.env`, default `txt`).
- Offer — but don't automatically perform — reading or summarizing the transcript. Transcripts can be long; auto-reading would dump them into the conversation.

When the transcription errors (non-zero exit from the underlying CLI):

- Surface the stderr/output relevant to the failure.
- Don't retry automatically.

## Failure modes — explicit summary

| Condition | Behavior |
|---|---|
| Project dir missing | Stop, report. |
| `venv/` missing | Stop, surface `make setup`. |
| `.env` missing | Stop, surface `cp .env.example .env`. |
| Input file missing or unsupported extension | Stop, report which. |
| Diarization on but `HF_TOKEN` unset/placeholder | Stop, tell user to set token. |
| Server probe fails | Start `python scripts/model_server.py …` in background (capture PID), poll /health up to 60s. |
| Server start times out | Kill the captured server PID, fall back to one-shot `python scripts/transcribe.py transcribe`. |
| Transcription CLI errors | Relay stderr, don't retry. |

## Things the skill explicitly does not do

- Run `make setup` or otherwise install dependencies.
- Edit `.env` or any other config files.
- Delete or move existing files in `transcripts/`.
- Try to "fix" unsupported file types (e.g. by transcoding).
- Stop a model server it started.

## Acceptance

The skill is working when, in a fresh Claude Code session:

1. User says "transcribe `/path/to/foo.mp4`" → Claude invokes the skill, runs the transcription, and reports the output file path. No manual command-walking required.
2. User says "transcribe this interview, I want to know who said what" → diarization is enabled automatically.
3. With no server running, the skill auto-starts one, completes the transcription, and leaves the server warm for the next request in the session.
4. Preflight failures (e.g. missing `.env`) are reported with the fix command, not silently worked around.
