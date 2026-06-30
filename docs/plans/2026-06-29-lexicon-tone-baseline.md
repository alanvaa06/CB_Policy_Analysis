# Lexicon Tone Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a transparent, document-level hawkish/dovish word-count tone measure that runs through the existing Phase 1 nested-OOS harness, plus a tone(t) plot vs RoBERTa.

**Architecture:** One new pure module `models/lexicon_scorer.py` + a versioned `data/lexicons/hawk_dove.json`. The scorer emits the same `[date, stance]` frame RoBERTa's `score_statements` does, so `stance_frame_from_scores`, `build_aligned_panel`, and `nested_oos` are untouched. The CLI gains `--tone-method {roberta,lexicon}`; only the stance-building line branches.

**Tech Stack:** Python 3.11+ (runs on 3.14), pandas, stdlib `json`/`re`, pytest. Matplotlib for the plot script only (not imported by harness or tests).

**Spec:** `docs/prd/003-lexicon-tone-baseline.md`

---

### Task 1: Policy-stance lexicon (corpus-validated) + loader

**Files:**
- Create: `data/lexicons/hawk_dove.json`
- Create: `src/cbp/models/lexicon_scorer.py`
- Test: `tests/test_lexicon_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lexicon_scorer.py
import json
import pytest
from pathlib import Path
from cbp.models.lexicon_scorer import load_lexicon


def test_load_lexicon_returns_two_nonempty_lowercase_frozensets(tmp_path):
    p = tmp_path / "lex.json"
    p.write_text(json.dumps({"hawk": ["Tightening", "RESTRICTIVE"], "dove": ["accommodative"],
                             "sources": ["x"], "notes": "y"}))
    hawk, dove = load_lexicon(p)
    assert isinstance(hawk, frozenset) and isinstance(dove, frozenset)
    assert hawk == {"tightening", "restrictive"}   # lowercased
    assert dove == {"accommodative"}


def test_load_lexicon_missing_file_raises_valueerror():
    with pytest.raises(ValueError, match="lexicon"):
        load_lexicon(Path("does/not/exist.json"))


def test_repo_lexicon_is_small_disjoint_and_policy_stance():
    hawk, dove = load_lexicon(Path("data/lexicons/hawk_dove.json"))
    assert 3 <= len(hawk) <= 8 and 3 <= len(dove) <= 8   # small, corpus-validated set
    assert hawk.isdisjoint(dove)
    # confound words MUST be excluded (verifier finding): boilerplate + condition valence + dead seeds
    banned = {"inflation", "weak", "downside", "robust", "slack", "upside",
              "gradual", "patient", "elevated", "hawkish", "dovish", "vigilan"}
    assert banned.isdisjoint(hawk | dove)
    # polarity-stable adjective chosen over flip-prone noun stem
    assert "accommodative" in dove and "accommodat" not in dove
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lexicon_scorer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cbp.models.lexicon_scorer'`

- [ ] **Step 3: Corpus-validate candidate stems, then create the lexicon data file**

First, check each candidate stem against the cached statements — it must fire in
several statements AND read directionally clean in sampled contexts. Run:

```bash
for stem in tighten restrictive firming restraint accommodative easing stimulus supportive; do
  n=$(rg -l -i "${stem}" data/raw/statements 2>/dev/null | wc -l)
  echo "== ${stem}: ${n} files =="
  rg -i -o ".{30}${stem}[a-z]*.{30}" data/raw/statements 2>/dev/null | head -3
done
```

Keep a stem only if it fires in several files and its contexts are directionally
consistent (drop e.g. `restraint` if it shows up as "fiscal restraint"; drop any that
mostly appears flipped like "removing accommodation"). If a side falls below 3 clean
stems, **flag it** — that means the measure barely fires; do not pad the list with
confounded words to hit the count. Then write the validated subset:

```json
{
  "hawk": ["tighten", "restrictive", "firming"],
  "dove": ["accommodative", "easing", "stimulus"],
  "sources": [
    "Apel & Blix Grimaldi (2012), 'The Information Content of Central Bank Minutes'",
    "Bennani & Neuenkirch (2017), 'The (Home) Bias of European Central Bankers'"
  ],
  "excluded": {
    "inflation": "dual-mandate boilerplate, ~51% of statements, not a stance signal",
    "weak/downside/robust/slack/upside": "economic-condition valence — tracks the economy, not Fed intent",
    "gradual/patient/elevated": "context-dependent, flips with syntax",
    "hawkish/dovish/vigilant/withdraw/resolute": "0-1 hits in FOMC statements (dicts built for speeches/minutes)"
  },
  "notes": "Policy-stance words only; corpus-validated against data/raw/statements (Step 3). Stem-match (startswith). 'accommodative' (adjective) chosen over 'accommodat' (noun stem) because 'removing accommodation' is hawkish. Document-level word-count; v1 ignores negation/polarity flip (PRD 003 §11). Final stems = the validated subset; adjust per the Step 3 counts and record changes here."
}
```

The exact kept stems are whatever Step 3 validates — the JSON above is the expected
result, but defer to the corpus. The unit test
`test_repo_lexicon_is_small_disjoint_and_policy_stance` enforces the size bounds and the
banned-word exclusions regardless of which clean stems you keep.

- [ ] **Step 4: Implement `load_lexicon`**

```python
# src/cbp/models/lexicon_scorer.py
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_lexicon(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """Load the hawkish/dovish stem lists from a JSON file.

    Returns (hawk, dove) as lowercased frozensets. Raises ValueError naming the
    path if the file is missing or malformed.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        hawk = frozenset(w.lower() for w in data["hawk"])
        dove = frozenset(w.lower() for w in data["dove"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"could not load lexicon from {path}: {e}") from e
    return hawk, dove
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_lexicon_scorer.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add data/lexicons/hawk_dove.json src/cbp/models/lexicon_scorer.py tests/test_lexicon_scorer.py
git commit -m "feat(lexicon): seed hawk/dove lexicon + load_lexicon"
```

---

### Task 2: Tokenizer

**Files:**
- Modify: `src/cbp/models/lexicon_scorer.py`
- Test: `tests/test_lexicon_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_lexicon_scorer.py
from cbp.models.lexicon_scorer import tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("The Committee will TIGHTEN, gradually.") == \
        ["the", "committee", "will", "tighten", "gradually"]


def test_tokenize_drops_digits_and_empty():
    assert tokenize("Rate at 5.25% — easing?") == ["rate", "at", "easing"]


def test_tokenize_empty_returns_empty_list():
    assert tokenize("   ") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lexicon_scorer.py -k tokenize -v`
Expected: FAIL — `ImportError: cannot import name 'tokenize'`

- [ ] **Step 3: Implement `tokenize`**

```python
# add to src/cbp/models/lexicon_scorer.py (after imports)
import re

_WORD = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphabetic word tokens. Drops digits and
    punctuation. Deterministic and dependency-free."""
    return _WORD.findall(text.lower())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lexicon_scorer.py -k tokenize -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/models/lexicon_scorer.py tests/test_lexicon_scorer.py
git commit -m "feat(lexicon): tokenize"
```

---

### Task 3: Document-level net-tone scorer

**Files:**
- Modify: `src/cbp/models/lexicon_scorer.py`
- Test: `tests/test_lexicon_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_lexicon_scorer.py
from cbp.models.lexicon_scorer import score_statement_lexicon

HAWK = frozenset({"tighten", "restrictive"})
DOVE = frozenset({"accommodative", "easing"})


def test_net_tone_all_hawk_is_plus_one():
    assert score_statement_lexicon("tighten restrictive tightening", HAWK, DOVE) == 1.0


def test_net_tone_all_dove_is_minus_one():
    assert score_statement_lexicon("accommodative easing", HAWK, DOVE) == -1.0


def test_net_tone_balanced_is_zero():
    # 2 hawk (tighten, restrictive), 2 dove (accommodative, easing)
    assert score_statement_lexicon("tighten restrictive accommodative easing", HAWK, DOVE) == 0.0


def test_net_tone_no_keywords_is_zero_not_nan():
    assert score_statement_lexicon("the committee met today", HAWK, DOVE) == 0.0


def test_stem_prefix_matches_inflection():
    # "tightened" must match stem "tighten"
    assert score_statement_lexicon("tightened", HAWK, DOVE) == 1.0


def test_polarity_stable_adjective_not_flip_prone_noun():
    # "accommodative" stem must NOT match the flip-prone noun "accommodation"
    # ("removing accommodation" is hawkish) -> 0.0, not -1.0
    assert score_statement_lexicon("accommodation", HAWK, DOVE) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lexicon_scorer.py -k net_tone -v`
Expected: FAIL — `ImportError: cannot import name 'score_statement_lexicon'`

- [ ] **Step 3: Implement `score_statement_lexicon`**

```python
# add to src/cbp/models/lexicon_scorer.py
def _count_side(tokens: list[str], stems: frozenset[str]) -> int:
    return sum(1 for t in tokens if any(t.startswith(s) for s in stems))


def score_statement_lexicon(text: str, hawk: frozenset[str], dove: frozenset[str]) -> float:
    """Document-level net tone = (n_hawk - n_dove) / (n_hawk + n_dove) over all
    tokens in `text`. Returns 0.0 when neither side fires (a valid neutral
    measurement, not NaN)."""
    tokens = tokenize(text)
    h = _count_side(tokens, hawk)
    d = _count_side(tokens, dove)
    total = h + d
    return 0.0 if total == 0 else (h - d) / total
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lexicon_scorer.py -k net_tone -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/models/lexicon_scorer.py tests/test_lexicon_scorer.py
git commit -m "feat(lexicon): document-level net-tone scorer"
```

---

### Task 4: Statement-frame scorer (`[date, stance]` contract)

**Files:**
- Modify: `src/cbp/models/lexicon_scorer.py`
- Test: `tests/test_lexicon_scorer.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_lexicon_scorer.py
import pandas as pd
from cbp.models.lexicon_scorer import score_statements_lexicon


def test_score_statements_lexicon_shape_and_values():
    statements = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-29", "2020-03-15"]),
        "text": ["tighten restrictive", "accommodative easing"],
    })
    out = score_statements_lexicon(statements, HAWK, DOVE)
    assert list(out.columns) == ["date", "stance"]
    assert len(out) == 2
    assert out.loc[0, "stance"] == 1.0
    assert out.loc[1, "stance"] == -1.0


def test_score_statements_lexicon_empty_text_scores_zero(caplog):
    statements = pd.DataFrame({"date": pd.to_datetime(["2020-01-29"]), "text": ["   "]})
    out = score_statements_lexicon(statements, HAWK, DOVE)
    assert len(out) == 1               # not dropped: empty text is a valid 0.0 reading
    assert out.loc[0, "stance"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lexicon_scorer.py -k score_statements -v`
Expected: FAIL — `ImportError: cannot import name 'score_statements_lexicon'`

- [ ] **Step 3: Implement `score_statements_lexicon`**

```python
# add to src/cbp/models/lexicon_scorer.py
import pandas as pd


def score_statements_lexicon(
    statements: pd.DataFrame, hawk: frozenset[str], dove: frozenset[str]
) -> pd.DataFrame:
    """Score each statement's document-level net tone. Mirrors the output of
    `models.stance_scorer.score_statements`: columns [date, stance], one row per
    statement. Unlike the RoBERTa path, empty/keyword-less text is kept as a
    valid 0.0 (neutral) reading and logged, not skipped."""
    rows = []
    n_zero = 0
    for _, r in statements.iterrows():
        stance = score_statement_lexicon(r["text"], hawk, dove)
        if stance == 0.0:
            n_zero += 1
        rows.append({"date": r["date"], "stance": stance})
    if n_zero:
        logger.info("lexicon: %d/%d statements scored neutral 0.0 (no directional words)",
                    n_zero, len(statements))
    return pd.DataFrame(rows, columns=["date", "stance"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lexicon_scorer.py -k score_statements -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cbp/models/lexicon_scorer.py tests/test_lexicon_scorer.py
git commit -m "feat(lexicon): score_statements_lexicon frame scorer"
```

---

### Task 5: Config lexicon path

**Files:**
- Modify: `src/cbp/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config.py
from pathlib import Path
from cbp.config import Config


def test_config_has_lexicon_path_default():
    cfg = Config()
    assert cfg.lexicon_path == Path("data/lexicons/hawk_dove.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -k lexicon -v`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'lexicon_path'`

- [ ] **Step 3: Implement**

```python
# modify src/cbp/config.py — add field after roberta_model_id
    roberta_model_id: str = "gtfintechlab/FOMC-RoBERTa"
    lexicon_path: Path = Path("data/lexicons/hawk_dove.json")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -k lexicon -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cbp/config.py tests/test_config.py
git commit -m "feat(config): lexicon_path default"
```

---

### Task 6: CLI `--tone-method` wiring + offline integration test

**Files:**
- Modify: `src/cbp/cli.py:66-111` (the `main()` argparse + the phase1 stance-building block)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

This test exercises the lexicon → stance-frame → nested-OOS path offline (no
network, no model), proving the lexicon scorer plugs into the existing harness.

```python
# append to tests/test_cli.py
from cbp.models.lexicon_scorer import score_statements_lexicon, load_lexicon
from cbp.data.stance import stance_frame_from_scores
from pathlib import Path


def test_lexicon_stance_runs_through_nested_report_offline():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", "2014-12-31", tz="UTC")
    rel_ts = pd.bdate_range("2010-01-27", "2014-12-15", freq="7W", tz="UTC")
    rel_date = rel_ts.tz_convert("America/New_York").normalize().tz_localize(None)
    # alternate hawkish/dovish statement text so lexicon stance is non-degenerate
    texts = ["tighten restrictive tightening" if i % 2 else "accommodative easing stimulus"
             for i in range(len(rel_ts))]
    statements = pd.DataFrame({"date": rel_date, "text": texts})
    hawk, dove = load_lexicon(Path("data/lexicons/hawk_dove.json"))
    scores = score_statements_lexicon(statements, hawk, dove)
    cal = pd.DataFrame({"release_date": rel_date, "release_ts": rel_ts})
    stance_df = stance_frame_from_scores(scores, cal)
    assert stance_df["stance"].abs().sum() > 0          # non-degenerate

    surprise = rng.normal(size=len(rel_ts))
    base = np.zeros(len(idx))
    for ts, u in zip(rel_ts, surprise):
        base[idx > ts] += 0.03 * u
    market = pd.DataFrame(index=idx)
    market["DGS2"] = 2.0 + base + 0.001 * rng.normal(size=len(idx))
    surprise_df = pd.DataFrame({"date": rel_date, "surprise": surprise})
    cfg = Config(horizons=(1,), target_series=("DGS2",))
    report = run_nested_report(market, stance_df, surprise_df, cfg)
    assert np.isfinite(report["nested"][("DGS2", 1)]["delta_r2"])


def test_cli_argparser_tone_method_default_and_choices():
    import argparse
    from cbp.cli import build_parser
    p = build_parser()
    assert p.parse_args([]).tone_method == "roberta"
    assert p.parse_args(["--tone-method", "lexicon"]).tone_method == "lexicon"
    with pytest.raises(SystemExit):
        p.parse_args(["--tone-method", "bogus"])
```

Add `import pytest` at the top of `tests/test_cli.py` if not present.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -k "tone_method or lexicon_stance" -v`
Expected: FAIL — `ImportError: cannot import name 'build_parser'`

- [ ] **Step 3: Refactor argparse into `build_parser` and add the flag + branch**

Extract the parser so it is unit-testable, then branch the stance source.

```python
# src/cbp/cli.py — replace the inline parser construction in main() with this
# module-level function, and call it from main().
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FOMC stance eval harness")
    ap.add_argument("--mode", choices=["phase0", "phase1"], default="phase1")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default="2024-06-30")
    ap.add_argument("--model", default=None,
                    help="override RoBERTa model id (default: config.roberta_model_id)")
    ap.add_argument("--tone-method", choices=["roberta", "lexicon"], default="roberta",
                    dest="tone_method",
                    help="stance source for --mode phase1 (default: roberta)")
    return ap
```

```python
# src/cbp/cli.py — in main(), replace `ap = argparse...; args = ap.parse_args()` with:
    args = build_parser().parse_args()
```

```python
# src/cbp/cli.py — in the phase1 block, replace the single scoring line:
#     scores = score_statements(statements, load_fomc_roberta(args.model or cfg.roberta_model_id))
# with the branch:
    if args.tone_method == "lexicon":
        from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon
        hawk, dove = load_lexicon(cfg.lexicon_path)
        scores = score_statements_lexicon(statements, hawk, dove)
    else:
        scores = score_statements(statements, load_fomc_roberta(args.model or cfg.roberta_model_id))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -k "tone_method or lexicon_stance" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `pytest -q`
Expected: all green (prior 62 + new lexicon/CLI tests)

- [ ] **Step 6: Commit**

```bash
git add src/cbp/cli.py tests/test_cli.py
git commit -m "feat(cli): --tone-method roberta|lexicon for phase1"
```

---

### Task 7: Tone time-series plot

**Files:**
- Create: `src/cbp/viz/__init__.py`
- Create: `src/cbp/viz/tone_compare.py`  (pure pandas — unit-tested)
- Create: `scripts/plot_tone_timeseries.py`  (matplotlib wrapper)
- Test: `tests/test_plot_tone.py`

The pure data-prep (`build_tone_comparison`) lives in `src/cbp/viz/` so it imports under
pytest (`pythonpath=["src"]` — `scripts/` is NOT on sys.path, per the verifier finding);
matplotlib stays in the `scripts/` wrapper's `main()`, never imported by tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot_tone.py
import pandas as pd
from cbp.viz.tone_compare import build_tone_comparison


def test_build_tone_comparison_inner_joins_on_date_and_correlates():
    lex = pd.DataFrame({"date": pd.to_datetime(["2020-01-29", "2020-03-15", "2020-04-29"]),
                        "stance": [1.0, -1.0, 0.5]})
    rob = pd.DataFrame({"date": pd.to_datetime(["2020-01-29", "2020-03-15"]),
                        "stance": [0.8, -0.6]})
    merged, corr = build_tone_comparison(lex, rob)
    assert list(merged.columns) == ["date", "stance_lexicon", "stance_roberta"]
    assert len(merged) == 2                       # inner join drops the unmatched date
    assert -1.0 <= corr <= 1.0
    assert corr > 0                               # both move the same direction here
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_plot_tone.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cbp.viz'`

- [ ] **Step 3: Implement the pure comparison module**

Create `src/cbp/viz/__init__.py` (empty), then:

```python
# src/cbp/viz/tone_compare.py
from __future__ import annotations

import pandas as pd


def build_tone_comparison(lex: pd.DataFrame, rob: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Inner-join lexicon and RoBERTa stance on date; return (merged, pearson_corr).
    merged columns: [date, stance_lexicon, stance_roberta]."""
    m = (lex.rename(columns={"stance": "stance_lexicon"})
            .merge(rob.rename(columns={"stance": "stance_roberta"}), on="date", how="inner")
            .sort_values("date").reset_index(drop=True))
    corr = float(m["stance_lexicon"].corr(m["stance_roberta"]))
    return m[["date", "stance_lexicon", "stance_roberta"]], corr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_plot_tone.py -v`
Expected: PASS

- [ ] **Step 5: Implement the matplotlib wrapper**

```python
# scripts/plot_tone_timeseries.py
"""Plot lexicon vs FOMC-RoBERTa statement tone over time + correlation scatter.

Usage: python scripts/plot_tone_timeseries.py
Reads the cached RoBERTa scores and recomputes lexicon scores from cached
statements, writes docs/results/figures/tone-timeseries.png.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from cbp.viz.tone_compare import build_tone_comparison


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cbp.config import Config
    from cbp.data.fomc_statements import fetch_statements
    from cbp.data.mp_surprise import load_surprise
    from cbp.models.lexicon_scorer import load_lexicon, score_statements_lexicon

    cfg = Config()
    surprise = load_surprise(cfg.data_dir / "raw" / "monetary-policy-surprises-data.xlsx",
                             sheet_name="FOMC (update 2023)", date_col="Date", surprise_col="MPS_ORTH")
    surprise = surprise[surprise["date"].dt.year >= 1999].reset_index(drop=True)
    statements = fetch_statements([d.date() for d in surprise["date"]], cfg.data_dir / "raw" / "statements")
    hawk, dove = load_lexicon(cfg.lexicon_path)
    lex = score_statements_lexicon(statements, hawk, dove)

    rob = pd.read_csv(cfg.data_dir / "raw" / "_phase1_stance_scores.csv", parse_dates=["date"])
    merged, corr = build_tone_comparison(lex, rob[["date", "stance"]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    ax1.plot(merged["date"], merged["stance_roberta"], label="FOMC-RoBERTa", lw=1.2)
    ax1.plot(merged["date"], merged["stance_lexicon"], label="lexicon", lw=1.2)
    ax1.axhline(0, color="0.6", lw=0.8); ax1.legend(); ax1.set_title("FOMC statement tone over time")
    ax2.scatter(merged["stance_roberta"], merged["stance_lexicon"], s=12)
    ax2.set_xlabel("RoBERTa"); ax2.set_ylabel("lexicon"); ax2.set_title(f"corr = {corr:.2f}")
    out = Path("docs/results/figures/tone-timeseries.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"wrote {out}  (n={len(merged)}, corr={corr:.3f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/cbp/viz/__init__.py src/cbp/viz/tone_compare.py scripts/plot_tone_timeseries.py tests/test_plot_tone.py
git commit -m "feat(viz): lexicon vs RoBERTa tone time-series + corr"
```

---

### Task 8: Live run, figure, and verdict (DoD §2-4)

**Files:**
- Create: `docs/results/2026-06-29-lexicon-baseline-verdict.md`
- Create: `docs/results/figures/tone-timeseries.png` (generated)
- Modify: `docs/context/{results,memory,todo,sesion-log}.md`

Not a TDD code task — an execution + documentation task. Requires `FRED_API_KEY`
and the cached statements/BS xlsx already on disk.

- [ ] **Step 1: Run the lexicon nested-OOS end-to-end**

Run: `python -m cbp.cli --mode phase1 --tone-method lexicon`
Capture the nested OOS table (ΔR² per target × horizon) + residual event-study.

- [ ] **Step 2: Generate the figure**

Run: `python scripts/plot_tone_timeseries.py`
Expected: `wrote docs/results/figures/tone-timeseries.png (n=..., corr=...)`

- [ ] **Step 3: Write the verdict doc**

Record: the lexicon ΔR² table, whether it adds OOS value beyond the BS surprise
(vs the RoBERTa null), the lexicon↔RoBERTa correlation, the neutral-0.0 count,
and the §11 caveats (no negation, stem judgment, document-level aggregation).
Mirror the structure of `docs/results/2026-06-19-phase1-verdict.md`.

- [ ] **Step 4: Update context files**

Per `CLAUDE.md` task management: one line in `results.md`, a `# decision`/`# VERDICT`
line in `memory.md`, clear the Phase-2a item in `todo.md`, append a `sesion-log.md` line.
Watch the context-size caps.

- [ ] **Step 5: Commit**

```bash
git add docs/results/2026-06-29-lexicon-baseline-verdict.md docs/results/figures/tone-timeseries.png docs/context/
git commit -m "docs: lexicon tone baseline verdict + tone(t) figure"
```

---

## Self-Review

**Spec coverage** (003 → task):
- §2 corpus-validated policy-stance list → Task 1 (validation Step 3 + JSON + sources/excluded).
- §3 new `lexicon_scorer.py`, `hawk_dove.json`, `viz/tone_compare.py`, plot wrapper; CLI/config extensions → Tasks 1-7.
- §4 contracts (`load_lexicon`, `tokenize`, `score_statement_lexicon`, `score_statements_lexicon`) → Tasks 1-4.
- §5 policy-stance-only + corpus-validation + stem prefix-match + banned-word exclusion + document-level → Task 1 (`test_repo_lexicon_is_small_disjoint_and_policy_stance`, Step 3 validation) + Task 3 (`test_polarity_stable_adjective_not_flip_prone_noun`).
- §6 reuse nested OOS → Task 6 offline integration test + Task 8 live run.
- §7 tone(t) + correlation plot → Task 7 (pure fn in `cbp.viz`) + Task 8 figure.
- §8 empty/no-keyword → 0.0 (Task 4), garbled JSON → ValueError (Task 1), 0/0 guard (Task 3).
- §10 DoD: pytest green (Tasks 1-7), CLI runs lexicon (Task 6/8), PNG (Task 7/8), verdict (Task 8).
- §11 caveats (no-negation/flip, confound exclusions) → JSON `excluded`/`notes` (Task 1) + verdict (Task 8).

**Verifier fixes folded in (workflow `wq9k0i9y5`):**
- *Test-harness blocker* — pure `build_tone_comparison` moved to `src/cbp/viz/tone_compare.py` (imports under `pythonpath=["src"]`); matplotlib isolated in the `scripts/` wrapper. Task 7.
- *Measure-validity blockers* — dropped `inflation` (51% boilerplate), the valence words (`weak`/`downside`/`robust`), the context-flippers (`gradual`/`patient`/`elevated`), and dead seeds (`hawkish`/`vigilant`); switched to corpus-validated policy-stance words + a unit-test-enforced banned-word exclusion; chose adjective `accommodative` over flip-prone noun stem `accommodat`. Tasks 1/3, spec §5/§11.

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `load_lexicon → (frozenset, frozenset)` consumed unchanged in Tasks 3/4/6/7; `score_statements_lexicon → [date, stance]` matches `stance_frame_from_scores` input (verified against `src/cbp/data/stance.py`); `build_tone_comparison` defined in `cbp.viz.tone_compare`, imported by both the test and the `scripts/` wrapper; `build_parser()` used in both `main()` and the CLI test.

**One open judgment (not a blocker):** the final kept stems depend on Task 1 Step 3 corpus-validation; the JSON shows the expected subset but execution defers to the counts. If a side validates below 3 clean stems, that is a real signal (the measure barely fires) to surface, not pad.
