# 005 — FOMC Statement Tracker (Dashboard v2): readable redline + depth analytics

**Status:** approved (design) · **Date:** 2026-06-30 · **Owner:** Alan
**Track:** descriptive analytics / monitoring (NOT predictive) · **Builds on:** the monitor engine (PRD 004, merged to main): `python -m cbp.monitor`, committed `data/monitor/tone_history.csv` + `latest_redline.json`, torch-free CI render → gh-pages.

## 1. Problem & goal

The v1 dashboard ships but three panels carry little signal:
1. **Redline is unreadable** — it diffs at *sentence* level, so a structurally different statement pair (e.g. a full statement-with-votes vs a short "maintain" statement) shares no identical sentences and difflib dumps the whole prior block (struck) + whole new block (added) = two walls of amber, boilerplate included.
2. **Tone charts say little** — all three measures crush a statement to one stance scalar that mostly *re-encodes the decision already known*: `action` = ±1/0 = hike/hold/cut; `lexicon` = hawk−dove count, mostly 0 and silent post-2024; `roberta` unpopulated.
3. **No context** — `action`/`lexicon`/`roberta` appear with no definitions, legend, or tooltips; a viewer cannot know what they mean.

**Goal:** turn the monitor into a **FOMC Statement Tracker** — a descriptive product that answers *what changed*, *what the Fed is focused on*, and *how much the statement shifted* — by: (A) a readable **word-level** redline, (B) new **depth analytics** (themes, change-magnitude, communication style) computed torch-free from the text, and (C) an on-page **glossary** so every measure is self-explaining. All additive to PRD 004's contract (committed CSV → torch-free CI render).

Non-goal: prediction/alpha (unchanged), new ML models, per-statement section parsing of the full 1999–present corpus (themes/metrics are robust word-level measures, not a parser).

## 2. Scope

**In scope (v2):**
- `metrics.py`: `clean_statement`, `word_count`, `flesch` (in-repo, no new dep), `count_themes`, `uncertainty_count`, `change_magnitude`.
- `data/lexicons/themes.json`: ~6 corpus-checkable theme term-lists + an uncertainty list.
- Extend `tone_history.csv` with the new per-statement metric columns; backfill recomputes all rows.
- Rewrite `contrast.redline` to **word-level** over **cleaned** text.
- Reorganize the dashboard into 6 panels incl. a **glossary** and a **theme heatmap**.
- Rewrite the README monitor section to feature the live tracker + link, framed by value.
- Offline tests for every pure function.

**Out of scope (YAGNI):** section-aware statement segmentation; sentence-embedding topic models; readability libraries (textstat); per-theme sentiment; changing the trigger/CI/publish path; populating the RoBERTa column (still a separate `.[infer]` run).

## 3. Architecture

Extends `src/cbp/monitor/`. New pure module + data file; existing modules edited at known seams. The committed `tone_history.csv` remains the single render contract.

```
NEW:
  monitor/metrics.py        # clean_statement, flesch, count_themes, uncertainty_count,
                            #   word_count, change_magnitude  (pure, torch-free, tested)
  data/lexicons/themes.json # {theme: [stems...]} for ~6 themes + "uncertainty"
EXTENDED:
  monitor/history.py        # HISTORY_COLUMNS += metric columns; load tolerates legacy rows
  monitor/score.py          # score_all_measures also computes metrics per statement
  monitor/contrast.py       # redline() -> WORD-level over clean_statement(text)
  monitor/site.py           # 6-panel layout: glossary, redline, theme heatmap,
                            #   change-magnitude, comm-style, stance-in-context
  monitor/__main__.py       # pass prior text into metrics for change_magnitude (uses cache)
  README.md                 # value-framed tracker section + live link
```

Render stays torch-free and reads only the committed CSV + `latest_redline.json`. No change to the calendar, CI, or gh-pages publish.

## 4. Data contracts

- **themes.json** — `{ "themes": { "inflation": [stems], "employment": [...], "growth": [...], "balance_sheet": [...], "financial_conditions": [...] }, "uncertainty": [stems], "sources": [...], "notes": str }`. Stems are lowercase prefixes matched like the hawk/dove lexicon (`startswith`). Corpus-checkable; seeded now, refinable.
- **tone_history.csv** — extend `HISTORY_COLUMNS` to:
  `date, action, lexicon_tone, roberta_stance, n_sentences,` **`word_count, flesch, uncertainty_per1k, change_magnitude, theme_inflation, theme_employment, theme_growth, theme_balance_sheet, theme_financial_conditions`**.
  Theme columns are intensity = hits per 1000 words (float). `change_magnitude` ∈ [0,1] vs the prior statement (first row = NaN). `flesch` is the Flesch Reading Ease score. All computed on `clean_statement(text)`.
- **latest_redline.json** — unchanged shape `{date_prior, date_latest, segments}`, but `segments` are now **word-level**: `{op, prev, curr}` with `op ∈ {equal, insert, delete, replace}` over word runs.

Signatures (typed, all pure):
- `clean_statement(text: str) -> str`
- `word_count(text: str) -> int`
- `flesch(text: str) -> float`
- `count_themes(text: str, themes: dict[str, frozenset[str]]) -> dict[str, int]`
- `uncertainty_count(text: str, terms: frozenset[str]) -> int`
- `change_magnitude(prev_text: str, curr_text: str) -> float`
- `load_themes(path: Path) -> tuple[dict[str, frozenset[str]], frozenset[str]]`  (themes, uncertainty)

## 5. Metric definitions

- **clean_statement** — remove, conservatively and by clearly-identified markers only (never drop substance):
  the release header up to and incl. "EDT"/"EST"/"Share"; the voting roster from "Voting for the monetary policy action" (or "Voting against") to the end of that block; the "For media inquiries" line; "Implementation Note issued …". Historical (1999–2005) statements lack these and pass through largely unchanged. Unit-tested on a modern statement with full boilerplate and on a bare historical one.
- **word_count** — `len(tokenize(clean))` reusing the lexicon tokenizer (alphabetic tokens).
- **flesch** — `206.835 − 1.015·(words/sentences) − 84.6·(syllables/words)`, sentences via `split_sentences`, syllables via a vowel-group heuristic (`[aeiouy]+` runs, min 1/word). Approximate by construction (documented); used for *relative* trend, not absolute grade.
- **count_themes** — for each theme, count tokens whose prefix matches any stem (like the lexicon `_count_side`). Reported as intensity per 1000 words.
- **uncertainty_count** — same matcher over the uncertainty list; reported per 1000 words (`uncertainty_per1k`).
- **change_magnitude** — `1 − difflib.SequenceMatcher(None, words(prev_clean), words(curr_clean)).ratio()` ∈ [0,1]; 0 = identical wording, 1 = fully rewritten. NaN when there is no prior statement.

## 6. Redline (word-level)

`redline(prev, curr)`: clean both, tokenize to **words** (whitespace runs preserved for rendering), run `difflib.SequenceMatcher` over the word lists, emit `{op, prev, curr}` runs. `site.build_redline_html` renders inline: unchanged words plain, `insert` green, `delete` struck red, `replace` shows old(struck)+new(green) inline. The result reads as one paragraph with only the changes highlighted, not two blocks. Boilerplate is gone (cleaned). Unit-tested on a near-identical pair (few words changed) and a heavily-reworded pair.

## 7. Dashboard panels (`site.py`)

1. **How to read this (glossary)** — collapsible/section: one plain-English line per measure (`action`: the rate decision verb, +1 hike / 0 hold / −1 cut — mirrors the decision; `lexicon`: net hawkish−dovish stance words, transparent, silent on 2024+; `roberta`: ML per-sentence stance, mean — populated on the heavy run; each theme; `change_magnitude`; `flesch`) + the honest framing reworded as capability ("reads what the statement *says* and *how it changed* — not a rate forecast").
2. **Latest statement** — date, decision label, and the word-level redline vs prior.
3. **What the Fed is focused on** — `go.Heatmap`: x = meeting dates, y = the 5 themes, z = per-1k intensity; shows focus shifts over time.
4. **How much the statement changed** — `change_magnitude` line over time; top-N spikes annotated with their dates.
5. **Communication style** — `word_count`, `flesch`, `uncertainty_per1k` as three traces on separate y-axes (or stacked small-multiples).
6. **Stance, in context** — the existing levels + meeting-over-meeting deltas, now with hover tooltips pulling the glossary one-liners; RoBERTa line gapped honestly until populated.

All Plotly, one self-contained `index.html`, render path unchanged (torch-free, reads committed CSV + redline JSON).

## 8. README (value-framed)

Rewrite the "Statement monitor" section: lead with the **live link** (`https://alanvaa06.github.io/CB_Policy_Analysis/`) and what the tracker shows — *the word-level redline of the latest statement, what the Fed is focused on (themes over time), how much each statement changed, and how its communication style evolves*. Frame as a transparent, reproducible, fully-tested research tool. Keep the existing research findings table intact; do not add negative caveats to the tracker section (the honest framing lives on the dashboard glossary).

## 9. Error handling / edge cases

- First statement (no prior) → `change_magnitude = NaN`; redline panel shows the "need ≥2 statements" placeholder.
- `clean_statement` that removes everything (unexpected layout) → fall back to the raw text + log, never emit empty (so metrics never divide by zero).
- `flesch` with zero sentences/words → return `0.0` (guarded), logged.
- Statement with no theme/uncertainty hits → intensities `0.0` (valid, not NaN).
- Legacy `tone_history.csv` missing the new columns → `load_history` fills them as NaN; a `--backfill`/normal run recomputes. Backfill recomputes the whole series so the shipped CSV is complete.

## 10. Stack & conventions

Typed Python + pandas + stdlib (`json`, `re`, `difflib`); **no new runtime dep** (Flesch is in-repo). Plotly stays isolated to `site.py` under `[site]`. Tests offline under `pythonpath=["src"]`, never importing torch or the network; metrics are pure functions tested on crafted text. Heavy `[infer]` still only for populating `roberta_stance`.

## 11. Success criteria (Definition of Done)

1. `pytest` green, incl.: `clean_statement` strips modern boilerplate and passes a historical statement; `flesch` within tolerance on a known sentence; `count_themes`/`uncertainty_count` on crafted text; `change_magnitude` = 0 for identical, ~1 for disjoint, NaN for no-prior; word-level `redline` opcodes on near-identical and reworded pairs; extended `HISTORY_COLUMNS` round-trips through save/load.
2. `python -m cbp.monitor --no-roberta` (torch-free) rebuilds `tone_history.csv` with all new columns populated for the 226 statements and `latest_redline.json` word-level.
3. `--rebuild-only` renders the 6-panel page (glossary, readable redline, theme heatmap, change-magnitude, comm-style, stance-in-context); torch-free + network-free.
4. README features the live link + value framing; no negative caveats in the tracker section.
5. CI publishes the v2 page to gh-pages unchanged.

## 12. Caveats / risks

- **Themes are word-count proxies**, not a topic model — a "theme" is term presence, so e.g. "inflation" fires whether the Fed is worried or reassured. Honest in the glossary; intensity (not sentiment) is the claim.
- **Flesch is approximate** (heuristic syllables) — read the *trend*, not the absolute grade level.
- **clean_statement is heuristic** across two page eras — conservative markers minimize over-stripping; the fallback-to-raw guard prevents empties; corpus-spot-checked.
- **change_magnitude conflates restructuring with substance** — a reordered-but-same statement scores high; paired with the redline (which shows *what*) this is informative, alone it is a coarse "edit size".
- **Still descriptive, not predictive** — the glossary states it; no metric is a forecast.
- **RoBERTa column remains NaN** until a `.[infer]` run; themes/style/change-magnitude give the modern-statement depth in the meantime.
