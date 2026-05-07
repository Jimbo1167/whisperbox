# Skill: `transcribe-locally` — design

- **Date:** 2026-05-07
- **Status:** Approved (design); implementation pending

## Goal

Expose a Claude Code user-scope skill that, when the user asks Claude to transcribe a video or audio file on their local machine, makes Claude actually run the transcription via the `local_video_transcriber` project — without the user having to walk Claude through the steps each time.

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
description: Use when the user asks to transcribe a video or audio file on their local machine (mp4, mov, wav, mp3, m4a, aac, etc.). Routes through Jim's local_video_transcriber project at ~/projects/local_video_transcriber — prefers the warm model server on :8000 when running, auto-starts it if not. Handles speaker diarization when the user signals interest in "who said what".
---
```

The `description` is the only trigger Claude reads when scanning available skills. Phrases the skill should fire on: "transcribe this video", "transcribe foo.mp4", "get a transcript of …", "what did they say in this recording", "who said what in this interview".

## Behavior

### 1. Project pin

Project lives at `~/projects/local_video_transcriber`. If `pwd` doesn't match, `cd` there for the duration of the run. The pin is explicit in the skill body, so this is not a silent assumption — it's the documented contract of the skill.

If `~/projects/local_video_transcriber` does not exist on disk, stop and report that the project is missing.

### 2. Preflight (one Bash call, fail fast)

In order:

1. `venv/` directory exists. If not → stop, surface `make setup`.
2. `.env` file exists. If not → stop, surface `cp .env.example .env`.
3. The input file path the user named exists, and its extension is one of `mp4, mov, m4v, wav, mp3, m4a, aac`. If not → stop, surface what was wrong.
4. If diarization is on for this run: `.env` contains a non-placeholder `HF_TOKEN=` line (i.e. not `your_huggingface_token_here`). If missing → stop, tell user to set HF_TOKEN.

When preflight fails, the skill reports the issue and the fix command. It never runs setup or token-installation steps itself.

### 3. Diarization decision

Off by default. Turn on (`--diarize` / `make diarize`) when the user's request mentions any of: "speakers", "who said what", "diarize", "diarization", "interview", "meeting", "multiple voices", "two people", "panel", "conversation between …".

If the request is ambiguous (e.g. "transcribe this call") and diarization isn't obviously needed, default to off. The skill body lists this rule explicitly so Claude doesn't re-derive it each invocation.

### 4. Routing: warm vs. cold

The skill calls the Python CLI directly (not `make` targets) so that the diarization flag works uniformly across both paths. All commands run inside the project venv:

```fish
. venv/bin/activate
```

Probe the model server:

```fish
curl -sf -m 2 http://localhost:8000/health
```

- **2xx** → server is warm. Run:
  ```
  python -m scripts.transcribe client transcribe "<file>" [--diarize]
  ```
- **non-2xx / connection refused** → start the server in the background. The Makefile's `make server` target runs in the foreground, so the skill backgrounds it explicitly:
  ```
  nohup make server > /tmp/local_video_transcriber-server.log 2>&1 &
  ```
  Then poll `/health` every ~2s until 2xx, capped at 60s total. Once warm, route via the warm command above.
  - If the 60s cap is hit without the server coming up, kill the backgrounded process and fall back to a one-shot run:
    ```
    python -m scripts.transcribe transcribe "<file>" [--diarize]
    ```
- The skill **does not stop a server it started** after the run completes. Leaving it running is the desired behavior.

Note: `make transcribe` / `make diarize` / `make client-transcribe` are convenient shortcuts but the skill prefers the underlying `python -m scripts.transcribe …` form because the Makefile doesn't expose `--diarize` on the client target. Routing through one CLI keeps the skill body uniform.

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
| Server probe fails | Start `make server` in background, poll /health up to 60s. |
| Server start times out | Kill backgrounded server, fall back to one-shot `python -m scripts.transcribe transcribe`. |
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
