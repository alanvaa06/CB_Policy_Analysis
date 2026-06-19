---
description: Snapshot any over-cap docs/context file to archive, then hard-compact it in place.
---

Context files have hard caps (approx tokens = bytes/4): `memory.md` 11k · `lessons.md` 7k · `todo.md` 2.5k · `results.md` 6k · `sesion-log.md` 4k. This command snapshots an over-cap file, then shrinks the active copy. Run it when you see `[context-size] OVER CAP` — not inline during feature work.

Optional argument: a specific filename to compact. No argument → check all five.

## Steps

1. **Find over-cap files.** For each `docs/context/*.md` (or just `$ARGUMENTS` if given), compare byte size to its cap above. Skip files under cap — don't touch them.

2. **For each over-cap `<file>`:**
   a. **Snapshot first (never lose history).** Copy the current file to `docs/context/archive/<file-stem>/<YYYY-MM-DD>.md`, creating dirs as needed. Use today's date.
   b. **Hard-compact the ACTIVE file in place:**
      - One line per entry, no prose. Dedupe — drop any line whose fact another line already implies.
      - `todo.md`: keep ONLY `pending` / `in_progress`. Append `done` items to `results.md` (they archive from there next cycle).
      - `memory.md`: keep live architecture decisions; collapse superseded ones into the decision that replaced them.
      - `lessons.md`: keep distinct rules; merge near-duplicates into the sharpest phrasing.
      - `results.md`: keep recent outcomes; older detail already lives in the snapshot.
      - `sesion-log.md`: keep recent sessions; older lines already live in the snapshot.

3. **Re-check + report.** Restat each compacted file. Report a line per file: `old size → new size`, plus its archive path.

Never touch anything outside `docs/context/`.
