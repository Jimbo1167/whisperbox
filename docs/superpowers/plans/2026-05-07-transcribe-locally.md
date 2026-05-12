# transcribe-locally Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Claude Code user-scope skill at `~/projects/skills/skills/transcribe-locally/SKILL.md` that makes Claude run video/audio transcriptions through the `whisperbox` project when the user asks for a transcript.

**Architecture:** Single `SKILL.md` (markdown + YAML frontmatter) — no supporting files, no helper scripts. The skill body encodes a decision tree: project pin + path resolution → preflight → diarization decision → warm-server probe → routing → reporting. Skill is auto-discovered via the existing `~/projects/skills/skills/` → `~/.claude/skills/` symlink.

**Tech Stack:** Markdown + YAML frontmatter. Verification commands use bash, curl, python (project venv), and the existing `scripts/transcribe.py` Click CLI plus `scripts/model_server.py`.

**Spec:** `docs/superpowers/specs/2026-05-07-transcribe-locally-skill-design.md`. Read before starting — every task references it.

**Note on TDD:** The deliverable is prose that Claude reads at session start, not code. There is no unit-test rhythm. Instead, each task verifies the assumption it depends on **before** encoding it in the skill. If a verification step fails, stop and fix the spec before continuing — don't paper over it in the SKILL.md.

---

## File Structure

- Create: `~/projects/skills/skills/transcribe-locally/SKILL.md`
- Modify: `~/projects/skills/README.md` — add an entry under "Current skills"
- Verify (read-only): `~/projects/whisperbox/scripts/transcribe.py`, `~/projects/whisperbox/scripts/model_server.py`, `~/projects/whisperbox/.env`, `~/projects/whisperbox/venv/`

The skill is one file. Don't create supporting `.md` files; if any sub-topic becomes long enough to warrant splitting later, that's a future change.

---

### Task 1: Verify the Click CLI invocation actually runs

The spec pins `python scripts/transcribe.py …` as the form to use. Before encoding it, confirm it works in this checkout. The reviewer specifically flagged this.

**Files:** None (verification only).

- [ ] **Step 1: Activate the venv and ask the CLI for help**

```bash
cd ~/projects/whisperbox
. venv/bin/activate
python scripts/transcribe.py --help
```

Expected: a Click help screen listing subcommands `transcribe`, `stream`, `batch`, `server`, `client`, `completion`. No `ModuleNotFoundError`, no traceback.

- [ ] **Step 2: Confirm the `transcribe` subcommand exposes `--diarize` and `--model`**

```bash
python scripts/transcribe.py transcribe --help
```

Expected output includes:
- `--diarize, -d` flag
- `--model, -m` option with choices including `tiny, base, small, medium, large`
- `--format, -f` option with choices including `txt, srt, vtt, json, pretty`

- [ ] **Step 3: Confirm the `client` subcommand accepts `--diarize` against the running server**

```bash
python scripts/transcribe.py client --help
```

Expected: shows `--server` option and that `client transcribe <file>` is the form for transcribing through a running server. Reading the file directly (the actual `--diarize` flag is implemented in `scripts/model_client.py:258` and gets passed through by the Click wrapper) is fine for confirmation.

- [ ] **Step 4: If any of the above fail, stop**

Don't continue. Either the venv is broken (`make setup` fixes it) or the CLI has drifted from what the spec assumes. Report the failure and fix the spec before proceeding.

- [ ] **Step 5: Note: no commit yet — verification only**

---

### Task 2: Verify the backgrounded model-server pattern

The spec backgrounds the model server via `nohup python scripts/model_server.py … &` and captures `$!` so it can be killed on timeout. Confirm this pattern actually starts the server, that `/health` becomes 2xx, and that killing the captured PID stops it cleanly.

**Files:** None (verification only).

- [ ] **Step 1: Make sure no model server is already running**

```bash
curl -sf -m 2 http://localhost:8000/health && echo "server up — stop it before running this task" || echo "no server — ok to proceed"
```

If a server is up, stop it first (find the PID with `pgrep -f model_server.py` and `kill` it).

- [ ] **Step 2: Background the server, capture its PID**

```bash
cd ~/projects/whisperbox
. venv/bin/activate
nohup python scripts/model_server.py --host 0.0.0.0 --port 8000 \
    > /tmp/whisperbox-server.log 2>&1 &
server_pid=$!
echo "server_pid=$server_pid"
```

Expected: a numeric PID is printed. The shell returns immediately.

- [ ] **Step 3: Poll /health for up to 60s**

```bash
for i in $(seq 1 30); do
  if curl -sf -m 2 http://localhost:8000/health > /dev/null; then
    echo "healthy after ${i} polls (~$((i*2))s)"
    break
  fi
  sleep 2
done
curl -sf -m 2 http://localhost:8000/health || { echo "still not healthy"; cat /tmp/whisperbox-server.log | tail -40; }
```

Expected: prints `healthy after N polls` within ~60s on a warm machine. If it doesn't come up, read the log and fix before continuing.

- [ ] **Step 4: Confirm the captured PID is the python process (not a shell wrapper)**

```bash
ps -p $server_pid -o pid,comm,args | head
```

Expected: `comm` is `python` (or `python3.11`). The `args` should include `scripts/model_server.py`. This is the whole point of bypassing `make` — the captured PID *is* the python process.

- [ ] **Step 5: Kill the server and confirm it stops**

```bash
kill $server_pid
sleep 2
curl -sf -m 2 http://localhost:8000/health && echo "STILL UP — bad" || echo "stopped — good"
ps -p $server_pid 2>&1 | grep -q "no such" && echo "process gone" || echo "process lingering"
```

Expected: `stopped — good` and `process gone`. If the process lingers, `kill -9 $server_pid` and note that the SKILL.md should include a `kill -9` fallback after a grace period.

- [ ] **Step 6: No commit — verification only**

---

### Task 3: Create the skill directory and write the frontmatter + structural skeleton

Now that both verification tasks have passed, create the file. This task lays down the frontmatter and section headers; later tasks fill each section.

**Files:**
- Create: `~/projects/skills/skills/transcribe-locally/SKILL.md`

- [ ] **Step 1: Confirm the symlink that auto-loads skills is in place**

```bash
ls -la ~/.claude/skills
```

Expected: `~/.claude/skills` is a symlink to `~/projects/skills/skills`. If not, the user's existing setup is broken — stop and surface that to them; don't try to fix it.

- [ ] **Step 2: Create the directory**

```bash
mkdir -p ~/projects/skills/skills/transcribe-locally
```

- [ ] **Step 3: Write the SKILL.md skeleton**

Create `~/projects/skills/skills/transcribe-locally/SKILL.md` with this content:

```markdown
---
name: transcribe-locally
description: Use when the user asks to transcribe a video or audio file on their local machine (mp4, mov, m4v, wav, mp3, m4a, aac). Routes through Jim's whisperbox project at ~/projects/whisperbox — prefers the warm model server on :8000 when running, auto-starts it if not. Speaker diarization is opt-in: only enabled when the user explicitly asks for speaker labels.
---

# transcribe-locally

When the user asks Claude to transcribe a video or audio file on their machine, run the transcription through the `whisperbox` project at `~/projects/whisperbox`. Don't walk the user through the commands — just do it and tell them where the transcript ended up.

## 1. Project pin and path resolution

(filled in by Task 4)

## 2. Preflight

(filled in by Task 4)

## 3. Diarization: off unless the user asks for speakers

(filled in by Task 5)

## 4. Routing: warm server preferred, cold fallback

(filled in by Task 6)

## 5. Model selection

(filled in by Task 7)

## 6. Reporting back

(filled in by Task 7)

## Failure modes

(filled in by Task 8)

## What this skill does NOT do

(filled in by Task 8)
```

- [ ] **Step 4: Sanity-check the YAML frontmatter parses**

```bash
python3 -c "import yaml, sys; doc = open('$HOME/projects/skills/skills/transcribe-locally/SKILL.md').read(); fm = doc.split('---')[1]; print(yaml.safe_load(fm))"
```

Expected: prints a dict with `name` = `transcribe-locally` and a `description` string. No `yaml.YAMLError`.

- [ ] **Step 5: Commit**

```bash
cd ~/projects/skills
git add skills/transcribe-locally/SKILL.md
git commit -m "Scaffold transcribe-locally skill (frontmatter + section headers)"
```

---

### Task 4: Fill in project pin, path resolution, and preflight

**Files:**
- Modify: `~/projects/skills/skills/transcribe-locally/SKILL.md` (replace the §1 and §2 placeholders)

- [ ] **Step 1: Replace the §1 placeholder**

Find the block:

```markdown
## 1. Project pin and path resolution

(filled in by Task 4)
```

Replace with:

```markdown
## 1. Project pin and path resolution

The project lives at `~/projects/whisperbox`. If that directory does not exist, stop and tell the user the project is missing — don't try to clone or recreate it.

**Resolve the user's input file to an absolute path *before* `cd`-ing into the project.** If the user said `./meeting.mp4` from `~/Downloads`, the literal path won't work after `cd`, so convert it first:

```bash
input_abs=$(realpath "<user-supplied-path>")
cd ~/projects/whisperbox
```

Use `$input_abs` everywhere downstream. Don't pass relative paths to the CLI.
```

- [ ] **Step 2: Replace the §2 placeholder**

Find the block:

```markdown
## 2. Preflight

(filled in by Task 4)
```

Replace with:

````markdown
## 2. Preflight (one Bash call, fail fast)

After `cd`, run these checks in order. If any fails, stop and tell the user the fix command — never run setup or token-installation steps for them.

1. **`venv/` exists** — if not: surface `make setup` and stop.
2. **`.env` exists** — if not: surface `cp .env.example .env` and stop.
3. **Input file exists and has a supported extension** — `mp4, mov, m4v, wav, mp3, m4a, aac`. If not: tell the user which check failed.
4. **If diarization is on for this run** (see §3): `.env` contains a non-placeholder `HF_TOKEN=`. Check with:
   ```bash
   grep -E '^HF_TOKEN=.+' .env | grep -v 'your_huggingface_token_here'
   ```
   If empty, stop and tell the user to set `HF_TOKEN` in `.env`.

Compact one-shot version:

```bash
cd ~/projects/whisperbox || { echo "project missing"; exit 1; }
[ -d venv ] || { echo "venv missing — run: make setup"; exit 1; }
[ -f .env ] || { echo ".env missing — run: cp .env.example .env"; exit 1; }
[ -f "$input_abs" ] || { echo "input file not found: $input_abs"; exit 1; }
case "${input_abs##*.}" in
  mp4|mov|m4v|wav|mp3|m4a|aac) ;;
  *) echo "unsupported extension: ${input_abs##*.}"; exit 1 ;;
esac
```

(`HF_TOKEN` check goes here too, but only when diarization is on.)
````

- [ ] **Step 3: Verify the file is still well-formed**

```bash
python3 -c "import yaml; fm = open('$HOME/projects/skills/skills/transcribe-locally/SKILL.md').read().split('---')[1]; print(yaml.safe_load(fm)['name'])"
```

Expected: prints `transcribe-locally`.

- [ ] **Step 4: Commit**

```bash
cd ~/projects/skills
git add skills/transcribe-locally/SKILL.md
git commit -m "Add §1 (project pin) and §2 (preflight) to transcribe-locally"
```

---

### Task 5: Fill in the diarization decision section

**Files:**
- Modify: `~/projects/skills/skills/transcribe-locally/SKILL.md` (replace §3 placeholder)

- [ ] **Step 1: Replace the §3 placeholder**

Find the block:

```markdown
## 3. Diarization: off unless the user asks for speakers

(filled in by Task 5)
```

Replace with:

```markdown
## 3. Diarization: off unless the user asks for speakers

**Default: off.** Only turn diarization on (`--diarize` on the CLI) when the user *explicitly asks for speaker information*. Phrases that count:

- "diarize", "with diarization"
- "with speakers", "label the speakers", "speaker labels"
- "who said what"
- "separate by speaker", "distinguish voices"

Phrases that **do not** count on their own:

- "interview", "meeting", "call", "conversation", "podcast"

Those describe the recording, not what the user wants out of the transcript. Diarization is slower and requires `HF_TOKEN`, so a false positive costs the user real time (or a hard failure if their token isn't set).

When the user's request is ambiguous, default off. They can always ask again with explicit speaker wording.
```

- [ ] **Step 2: Commit**

```bash
cd ~/projects/skills
git add skills/transcribe-locally/SKILL.md
git commit -m "Add §3 (diarization heuristic) to transcribe-locally"
```

---

### Task 6: Fill in the routing logic (warm vs cold)

**Files:**
- Modify: `~/projects/skills/skills/transcribe-locally/SKILL.md` (replace §4 placeholder)

- [ ] **Step 1: Replace the §4 placeholder**

Find the block:

```markdown
## 4. Routing: warm server preferred, cold fallback

(filled in by Task 6)
```

Replace with:

````markdown
## 4. Routing: warm server preferred, cold fallback

Use the **Click CLI at `scripts/transcribe.py`** for both warm and cold paths. **Do not** use `make transcribe` / `make diarize` / `make client-transcribe` — they invoke a different script (`transcribe_video.py`) that doesn't expose the flags this skill needs. See "Why not the Makefile" at the end of this section.

Activate the venv:

```bash
. venv/bin/activate
```

Probe the model server. `curl -sf` returns non-zero on any non-2xx response, so the exit code alone is the signal:

```bash
curl -sf -m 2 http://localhost:8000/health
```

### If the server is warm (curl exit 0)

Run the transcription through the client. Add `--diarize` only if §3 says diarization is on for this run:

```bash
python scripts/transcribe.py client transcribe "$input_abs" [--diarize]
```

### If the server is not warm (curl non-zero)

Start the server in the background. **Do not use `make server`** — it invokes python through `make`, so `$!` would be the make PID, not the python PID, and killing it later wouldn't actually stop the server. Run the python directly:

```bash
nohup python scripts/model_server.py --host 0.0.0.0 --port 8000 \
    > /tmp/whisperbox-server.log 2>&1 &
server_pid=$!
```

Poll `/health` every 2 seconds, capped at 60s total:

```bash
for i in $(seq 1 30); do
  if curl -sf -m 2 http://localhost:8000/health > /dev/null; then
    echo "server is now warm at :8000"
    break
  fi
  sleep 2
done
```

If `/health` becomes 2xx within the cap → log "server is now warm at :8000" and run the warm command above. The next invocation in this session will find the server already running, so the preflight check stays cheap.

If the cap is hit without health → fall back to a one-shot run. **Kill the backgrounded process first** so it doesn't linger:

```bash
kill $server_pid 2>/dev/null
sleep 2
kill -9 $server_pid 2>/dev/null   # grace period elapsed; force
python scripts/transcribe.py transcribe "$input_abs" [--diarize]
```

### When the run succeeds

Do **not** stop a server the skill started. Leaving it warm is the desired behavior — subsequent transcribes in the same session reuse it.

### Why not the Makefile

This project has two parallel CLIs:

- `make transcribe` and `make diarize` invoke `transcribe_video.py`, a thin script with its own `argparse` that supports only `input_path` and `-o/--output`. Everything else (model size, language, format, diarization toggle) comes from the `.env`. `make diarize` flips diarization on with an injected env var.
- `python scripts/transcribe.py …` is a Click CLI with full options: `--diarize`, `--model`, `--language`, `--format` (including `pretty`), plus `client`, `stream`, `batch`, `server` subcommands.

The skill goes through the Click CLI because it's the only path that gives uniform `--diarize` (and other flag) handling across cold and warm runs. Don't conflate them — they are not a wrapper relationship.

The skill executes through Claude's Bash tool (bash), even though the user's interactive shell is fish. All shell snippets in this skill are bash.
````

- [ ] **Step 2: Commit**

```bash
cd ~/projects/skills
git add skills/transcribe-locally/SKILL.md
git commit -m "Add §4 (warm/cold routing) to transcribe-locally"
```

---

### Task 7: Fill in model selection and reporting-back sections

**Files:**
- Modify: `~/projects/skills/skills/transcribe-locally/SKILL.md` (replace §5 and §6 placeholders)

- [ ] **Step 1: Replace the §5 placeholder**

Find the block:

```markdown
## 5. Model selection

(filled in by Task 7)
```

Replace with:

```markdown
## 5. Model selection

Don't override `WHISPER_MODEL` from `.env`. Pass `--model <size>` only when the user explicitly asks for a different size — for example "use the small model", "do this with large-v3", "tiny model is fine for this".

Valid sizes (`scripts/transcribe.py` Click choices): `tiny, base, small, medium, large`.
```

- [ ] **Step 2: Replace the §6 placeholder**

Find the block:

```markdown
## 6. Reporting back

(filled in by Task 7)
```

Replace with:

```markdown
## 6. Reporting back

When the run succeeds:

- Tell the user the output path. Default is `transcripts/<input-stem>.<ext>`, where `<ext>` is `OUTPUT_FORMAT` from `.env` (default `txt`).
- **Offer** to read or summarize the transcript — don't auto-read. Transcripts can be long; pulling thousands of lines into the conversation isn't useful.

When the run fails (non-zero exit from the CLI):

- Surface the relevant stderr. Don't retry automatically.
- Don't try to "recover" by switching paths (e.g., warm → cold) unless the failure is specifically that the server vanished mid-run.
```

- [ ] **Step 3: Commit**

```bash
cd ~/projects/skills
git add skills/transcribe-locally/SKILL.md
git commit -m "Add §5 (model selection) and §6 (reporting) to transcribe-locally"
```

---

### Task 8: Fill in failure-modes table and "what this skill does NOT do"

**Files:**
- Modify: `~/projects/skills/skills/transcribe-locally/SKILL.md` (replace last two placeholders)

- [ ] **Step 1: Replace the failure-modes placeholder**

Find the block:

```markdown
## Failure modes

(filled in by Task 8)
```

Replace with:

```markdown
## Failure modes

| Condition | Behavior |
|---|---|
| `~/projects/whisperbox` missing | Stop, report. Don't try to clone. |
| `venv/` missing | Stop, surface `make setup`. |
| `.env` missing | Stop, surface `cp .env.example .env`. |
| Input file missing or unsupported extension | Stop, report which. |
| Diarization on but `HF_TOKEN` unset/placeholder | Stop, tell user to set `HF_TOKEN` in `.env`. |
| Server probe fails (curl non-zero) | Start `python scripts/model_server.py` in background, capture PID, poll `/health` up to 60s. |
| Server start times out | `kill $server_pid` (then `kill -9` after grace), fall back to `python scripts/transcribe.py transcribe …`. |
| Transcription CLI exits non-zero | Relay stderr, don't retry. |
```

- [ ] **Step 2: Replace the "does NOT do" placeholder**

Find the block:

```markdown
## What this skill does NOT do

(filled in by Task 8)
```

Replace with:

```markdown
## What this skill does NOT do

- Run `make setup` or otherwise install dependencies.
- Edit `.env` or any other config files.
- Install or rotate `HF_TOKEN`.
- Delete or move existing files in `transcripts/`.
- Try to "fix" unsupported file types by transcoding.
- Stop a model server it started after a successful run.
- Use the `make transcribe` / `make diarize` / `make client-transcribe` targets — those go through a different, more limited script (see §4 "Why not the Makefile").
```

- [ ] **Step 3: Commit**

```bash
cd ~/projects/skills
git add skills/transcribe-locally/SKILL.md
git commit -m "Add failure-modes table and exclusions to transcribe-locally"
```

---

### Task 9: Final read-through and consistency check

**Files:**
- Read: `~/projects/skills/skills/transcribe-locally/SKILL.md`

- [ ] **Step 1: Read the full SKILL.md top to bottom**

```bash
cat ~/projects/skills/skills/transcribe-locally/SKILL.md
```

Pretend you are a fresh Claude session that just loaded this skill. Check:

- Does §3 give you a clear yes/no rule for diarization, with no ambiguity?
- Does §4 tell you exactly which command to run for each branch?
- Are `$input_abs` and `$server_pid` both defined before they're used?
- Are the failure-modes table entries each implementable from the body sections above?
- Are there any inline TODOs, placeholders, or "(filled in by Task N)" leftovers?

- [ ] **Step 2: Check for placeholder leftovers**

```bash
grep -n "filled in by" ~/projects/skills/skills/transcribe-locally/SKILL.md && echo "FOUND PLACEHOLDERS" || echo "clean"
grep -nE "TODO|TBD|FIXME" ~/projects/skills/skills/transcribe-locally/SKILL.md && echo "FOUND" || echo "clean"
```

Expected: both print `clean`. If either finds something, fix it before continuing.

- [ ] **Step 3: Frontmatter sanity check**

```bash
python3 -c "
import yaml
doc = open('$HOME/projects/skills/skills/transcribe-locally/SKILL.md').read()
parts = doc.split('---', 2)
assert len(parts) >= 3, 'missing frontmatter delimiters'
fm = yaml.safe_load(parts[1])
assert fm['name'] == 'transcribe-locally'
assert 'description' in fm and len(fm['description']) > 50
print('frontmatter ok:', fm['name'])
"
```

Expected: `frontmatter ok: transcribe-locally`.

- [ ] **Step 4: If anything in step 1–3 looks wrong, fix and amend**

```bash
cd ~/projects/skills
git add skills/transcribe-locally/SKILL.md
git commit -m "Final pass on transcribe-locally"   # only if there were changes
```

If no changes, skip the commit.

---

### Task 10: Update the skills repo README

**Files:**
- Modify: `~/projects/skills/README.md`

- [ ] **Step 1: Read the current README**

```bash
cat ~/projects/skills/README.md
```

Look for the "Current skills" section (around line 27).

- [ ] **Step 2: Add the new skill entry**

In `~/projects/skills/README.md`, find the line:

```markdown
- **`writing-adrs`** — Jim's Architecture Decision Record practice: when to write one, the template, what makes a good ADR, lifecycle, seeding new repos, and a survey of open-source ADR tooling.
```

After it, add:

```markdown
- **`transcribe-locally`** — Run video/audio transcriptions through `~/projects/whisperbox`. Prefers the warm model server on :8000, auto-starts it if not. Diarization opt-in via explicit speaker wording.
```

(The existing README also lists `design-vocabulary` and `sweep-modules` as skills present in the directory — if they aren't already in the "Current skills" section, leave them be; this task is only about adding `transcribe-locally`. Don't restructure the README.)

- [ ] **Step 3: Commit**

```bash
cd ~/projects/skills
git add README.md
git commit -m "List transcribe-locally in skills README"
```

- [ ] **Step 4: Push the skills repo**

```bash
cd ~/projects/skills
git push
```

(Per the user's global `CLAUDE.md`: "After committing, always push (and rebase if needed) unless explicitly told not to.")

---

### Task 11: Commit the implementation plan and spec to the project repo

The spec and plan live in `~/projects/whisperbox/docs/superpowers/`. The spec was already committed during brainstorming; this task makes sure the plan is committed too if it wasn't already.

**Files:**
- (no source changes; just confirm this plan file is committed)

- [ ] **Step 1: Check status in the project repo**

```bash
cd ~/projects/whisperbox
git status
```

If `docs/superpowers/plans/2026-05-07-transcribe-locally.md` is untracked or modified, commit it:

```bash
git add docs/superpowers/plans/2026-05-07-transcribe-locally.md
git commit -m "Add implementation plan for transcribe-locally skill"
git push
```

Otherwise: nothing to do.

---

### Task 12: Manual end-to-end smoke test (user-driven)

This task is for the user, not the implementing engineer. The skill is loaded at Claude Code session start, so the only way to confirm it works is to start a fresh session and try it.

**Files:** None.

- [ ] **Step 1: Start a fresh Claude Code session in any directory**

In a new terminal, `claude` (or open a new VS Code Claude Code panel).

- [ ] **Step 2: Confirm the skill is in the available-skills list**

Ask: "What skills do you have available?" or check the system reminder at session start. Look for `transcribe-locally` in the list.

- [ ] **Step 3: Try the cold path (no server running)**

Make sure no model server is running first:

```bash
pgrep -f model_server.py && pkill -f model_server.py
```

Then ask Claude in the new session: "Transcribe `~/projects/whisperbox/test.wav`."

Expected: Claude invokes the skill, runs preflight, sees no server, starts one in the background, polls /health, then runs the transcription. At the end, reports the path of the output file in `transcripts/`.

- [ ] **Step 4: Try the warm path (server already up from step 3)**

In the same session, ask Claude to transcribe a different file. The skill's preflight should find the server already healthy and route directly to the client without restarting anything.

- [ ] **Step 5: Try the diarization path**

Ask: "Transcribe `~/projects/whisperbox/test.wav` and label the speakers." The skill should turn on `--diarize`. (This may fail if `HF_TOKEN` is unset — that's a feature, not a bug; the skill should report it as a preflight error.)

- [ ] **Step 6: Try a "shouldn't trigger diarization" phrasing**

Ask: "Transcribe this meeting recording: `<file>`." The skill should default diarization OFF (because "meeting" alone doesn't count per §3).

- [ ] **Step 7: Surface findings**

If any step misbehaves, note exactly what the skill did vs. what was expected. The fix may be a tweak to wording in `SKILL.md` (e.g., diarization wording set is too narrow / too loose).

---

## Self-review

After writing the plan, the author re-checked it against the spec:

**Spec coverage** — every spec section has a task:
- Frontmatter, location, single-file structure → Task 3
- §1 project pin + path resolution → Task 4
- §2 preflight → Task 4
- §3 diarization → Task 5
- §4 warm/cold routing + Makefile divergence note → Task 6
- §5 model selection → Task 7
- §6 reporting back → Task 7
- Failure modes table → Task 8
- "Things skill explicitly does NOT do" → Task 8
- Acceptance criteria #1–#4 → Task 12

**Placeholder scan** — Tasks 4, 6, 7, 8 each replace a placeholder block from Task 3's skeleton with a concrete body. Task 9 specifically scans for `filled in by` / `TODO` / `TBD` leftovers and fails loudly if any remain.

**Type/identifier consistency** — `$input_abs` is defined in §1 (Task 4) and used in §2 (Task 4) and §4 (Task 6). `$server_pid` is defined in §4 (Task 6) and used later in §4. Failure-modes table entries (Task 8) reference both. Consistent.

**Verification before encoding** — Tasks 1 and 2 verify the two assumptions the reviewer flagged (CLI form actually runs; PID handling actually works) before any of that text is written into the skill body. If either verification fails, the spec gets fixed first.
