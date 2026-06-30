# 003 — Transparent Hawkish/Dovish Lexicon Tone: Baseline vs FOMC-RoBERTa (Phase 2a)

**Status:** approved (design) · **Date:** 2026-06-29 · **Owner:** Alan
**Track:** live real-time signal, research-grade · **Builds on:** Phase 1 harness (002), merged to main; Phase 1 verdict = NO-GO (RoBERTa stance adds no OOS value beyond the BS surprise, canonical-confirmed).

## 1. Problem & goal

Phase 1 measured FOMC statement stance with a transformer (FOMC-RoBERTa) and found
it adds **no** out-of-sample value beyond the Bauer-Swanson policy surprise. The
measure is a black box: we cannot see *which words* drove a stance, so we cannot
tell whether the null reflects "no information in the text" or "the model's
encoding misses it."

**Goal:** build a **transparent, deterministic hawkish/dovish lexicon** tone measure
(net word-count), run it through the *same* Phase 1 nested-OOS harness, and answer two
questions:
1. **Descriptive** — what does a transparent tone(t) series look like 1999–2023, and
   does it track RoBERTa's stance? (interpretability + sanity)
2. **Predictive** — does a simple word-count tone add OOS value beyond the BS surprise
   where RoBERTa did not? (almost certainly another null, but an *interpretable* one,
   and it pre-empts the reviewer question "did you try a transparent baseline?")

**Why a lexicon:** it measures the same hawk–dove axis as RoBERTa, so the comparison
is apples-to-apples; every score is auditable (you see the words that fired); it has
no license encumbrance (unlike CC BY-NC RoBERTa); and the Lucca-Trebbi-style
transparent baseline was a top-3 build candidate in the project's deep-research run.

Non-goal: beating RoBERTa or a production signal. This closes the interpretability gap
left by Phase 1.

## 2. Scope

**In scope (Phase 2a v1):**
- A small, **corpus-validated policy-stance** term list (seeded from published
  monetary-policy dictionaries — Apel & Blix Grimaldi 2012; Bennani & Neuenkirch 2017 —
  but pruned to words that actually fire in FOMC *statements* and are directionally clean).
- A pure, tested lexicon scorer producing per-statement net tone, **document-level**.
- A CLI switch to run `--mode phase1` with the lexicon measure instead of RoBERTa,
  reusing the surprise control, alignment, and nested OOS unchanged.
- A tone-over-time plot overlaying lexicon vs RoBERTa stance, saved as a figure.
- A documented comparison: lexicon ΔR² table vs the RoBERTa verdict.

**Out of scope (YAGNI):** negation handling, per-sentence aggregation, Loughran-McDonald,
SO-PMI/PMI corpora, stemming via heavy NLP deps, new targets/horizons, new statement
fetching (reuse the 197 cached statements).

## 3. Architecture

Extends `src/cbp`. One responsibility per module; scoring is a pure function so it is
fully unit-testable offline (no network, no model).

```
NEW:
  models/lexicon_scorer.py   # tokenize + stem-match + score_statements_lexicon (pure)
  data/lexicons/hawk_dove.json  # versioned term lists + provenance (data, not code)
  viz/tone_compare.py        # build_tone_comparison (pure pandas, unit-tested)
  scripts/plot_tone_timeseries.py  # matplotlib wrapper over build_tone_comparison -> PNG
EXTENDED:
  cli.py     # --tone-method {roberta,lexicon}; only the stance-building line branches
  config.py  # lexicon path (default)
```

The pure comparison logic lives in `src/cbp/viz/tone_compare.py` (pandas only) so it is
importable under pytest (`pythonpath=["src"]`); matplotlib stays in the `scripts/`
wrapper, never imported by the harness or tests.

Downstream (`stance_frame_from_scores`, `build_aligned_panel`, `nested_oos`,
`residual_stance_regression`) is **untouched** — the lexicon scorer emits the same
`[date, stance]` contract RoBERTa does.

## 4. Data contracts

- **Lexicon** — `hawk_dove.json`: `{"hawk": [str, ...], "dove": [str, ...], "sources": [...], "notes": str}`.
  Terms are lowercase stems (e.g. `tighten`, `accommodat`, `inflationar`).
- **Stance** — `DataFrame[date: datetime64, stance: float]`, identical shape to the
  RoBERTa path; `stance = (n_hawk − n_dove)/(n_hawk + n_dove)` ∈ [−1,+1], `0.0` when
  neither side fires. One row per statement.

Signatures:
- `tokenize(text: str) -> list[str]`  (lowercase, `\b\w+\b`)
- `load_lexicon(path: Path) -> tuple[frozenset[str], frozenset[str]]`
- `score_statement_lexicon(text: str, hawk, dove) -> float`  (document-level net tone)
- `score_statements_lexicon(statements: pd.DataFrame, hawk, dove) -> pd.DataFrame`
  (mirrors `score_statements` output: `[date, stance]`; empty-text rows logged + skipped)

## 5. Lexicon & matching

- **Policy-stance terms only.** v1 counts a SMALL set of words that name the Fed's
  policy *stance/action* (e.g. `restrictive`, `accommodative`, `tightening`, `firming`,
  `easing`) — explicitly **not** economic-condition words (`weak`, `robust`, `downside`,
  `slack`) and **not** dual-mandate boilerplate (`inflation` appears in ~51% of statements
  regardless of stance). Condition/boilerplate words conflate the *economy* (recession →
  more "downside" language) with Fed *intent*; they are excluded by design (see §11).
  These exclusions are enforced by a unit test, not just convention.
- **Corpus-validated, not blind.** Candidate stems are seeded from the published
  dictionaries but **finalized by validation against the cached statements**: each kept
  stem must (a) actually fire and (b) be directionally clean in a sampled-context check.
  Dead seeds (`hawkish`, `dovish`, `vigilant` — 0 hits in statements) and flip-prone
  terms are dropped. Kept/dropped decisions + residual caveats are recorded in the JSON
  `notes`/`excluded` keys. The final list is expected to be **small (~4–8 stems/side)**
  and to fire on a *subset* of statements (→ honest `0.0` neutrals when no stance word appears).
- **Stem match**: a token counts for a side if it *starts with* any stem on that side
  (`tighten` → `tighten`/`tightening`/`tightened`). Dependency-free, no nltk/Porter;
  stems hand-checked for false prefixes (prefer the polarity-stable adjective
  `accommodative` over the flip-prone noun stem `accommodat`, which also matches
  "removing accommodation" = hawkish). Unit-tested.
- **Document-level ratio**: `(n_hawk − n_dove)/(n_hawk + n_dove)` over the whole
  statement — a length-normalized proportion, `0.0` when no stance word fires. No
  sentence split (also exercises the Phase-2 "read the whole document" idea).

## 6. Eval — reuse Phase 1 nested OOS

For each `target ∈ {DGS2, DGS1}`, `h ∈ {1,5,22}`: build the panel from the lexicon
stance + the **same** BS orthogonalized surprise control, run `nested_oos`
(A=[surprise] vs B=[surprise, stance]) and `residual_stance_regression`. Report ΔR²,
in-sample stance_t, residual event-study — exactly the Phase 1 table, now for the
lexicon measure. No new eval code.

## 7. Visualization

`scripts/plot_tone_timeseries.py` reads both stance frames (lexicon + cached RoBERTa
scores `_phase1_stance_scores.csv`), aligns on date, and writes a PNG to
`docs/results/figures/tone-timeseries.png`: two lines over 1999–2023 (the "tone across
time" series), plus a scatter/correlation of lexicon vs RoBERTa stance. Matplotlib,
reproducible, version-controlled.

## 8. Error handling / edge cases

- Empty / no-keyword statement → `stance = 0.0` (neutral), not dropped (it *is* a valid
  measurement: no directional words). Distinguish from RoBERTa's empty-text skip; log count.
- Missing/garbled lexicon JSON → fail fast with a clear `ValueError` naming the path.
- Statement with hawk and dove both 0 → ratio `0/0` guarded to `0.0`.
- Reuse Phase 1 alignment drops/skips unchanged (missing surprise → drop + log).

## 9. Stack & conventions

Pure-Python + pandas; **no new runtime deps** (the lexicon loads via stdlib `json`).
The pure tone-comparison logic lives in `src/cbp/viz/tone_compare.py` (pandas only) so it
unit-tests under `pythonpath=["src"]`; matplotlib is imported only inside
`scripts/plot_tone_timeseries.py` `main()` (dev/optional, never by the harness or tests). Typed; tests accompany every
module. The lexicon path runs in seconds — no GPU, no download.

## 10. Success criteria (Definition of Done)

1. `pytest` green, including: tokenize edge cases; stem-match true/false-prefix guards;
   a **known statement → known net-tone** score; `0/0 → 0.0`; empty-text → logged 0.0;
   frame-shape parity with `score_statements`; a CLI test for `--tone-method lexicon`.
2. CLI runs `--mode phase1 --tone-method lexicon` end-to-end and prints the nested OOS
   table (ΔR² per target × horizon) + residual event-study.
3. The tone-timeseries PNG is produced and committed.
4. A documented verdict appended to the Phase 1 results: does the **transparent** tone
   add OOS value beyond the BS surprise, and how does lexicon stance correlate with
   RoBERTa stance? (interpretability close-out for the text-signal thread.)

## 11. Caveats / risks

- **No negation / residual polarity flip:** "not accommodative", "less restrictive",
  and crucially "removing accommodation" / "policy firming may be appropriate" flip the
  intent of a stance token. v1 document-level word-count cannot parse this; the
  corpus-validation prefers polarity-stable adjectives and records the residual, but this
  is the measure's known floor — it reads stated-stance *language*, not policy *intent*.
- **Confound exclusions (why the list is small):** condition words (`weak`/`robust`/
  `downside`) and boilerplate (`inflation`) were dropped because document-level counts of
  them track the *economy*, not Fed stance (a recession statement accumulates "downside"
  regardless of policy). This makes v1 fire on a subset of statements — a smaller, honest
  signal rather than a large confounded one. Validated against the corpus (§5).
- **Stem false-positives:** prefix matching can over/under-count; mitigated by hand-checked
  stems + unit tests, but the kept list is a judgment call and its corpus-validation is the defense.
- **Document-level ≠ RoBERTa per-sentence:** the two measures aggregate differently, so a
  low lexicon-vs-RoBERTa correlation could be aggregation, not construct. Note when comparing.
- **Same control, same caveats:** BS surprise is intraday-calibrated vs daily DGS (Phase 1
  caveat carries over). The marginal-text question is unchanged.
- Expected outcome is another null; the value is interpretability + the transparent
  baseline on record, not a new signal.
