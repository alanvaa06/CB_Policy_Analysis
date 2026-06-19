# Phase 1 Verdict — Does FOMC statement stance add predictive value beyond the policy surprise?

**Date:** 2026-06-19 · **Status:** decided · **Result: NO-GO (publishable null)** · Spec: `docs/prd/002-fomc-marginal-effect.md`

## Question (PRD §1)
Does the stance of the FOMC post-meeting statement carry out-of-sample predictive information for market rates **beyond** the known Bauer-Swanson monetary-policy surprise? Operationalized as a nested OOS comparison: feature set A = `[surprise]` vs B = `[surprise, stance]`; the text adds value iff `ΔR² = oos_r2(B) − oos_r2(A) > 0`.

## Method
- **Stance:** FOMC-RoBERTa per-sentence label (Dovish −1 / Hawkish +1 / Neutral 0), averaged per statement.
- **Control:** Bauer-Swanson **orthogonalized** surprise (`MPS_ORTH`, SF Fed `monetary-policy-surprises-data.xlsx`, FOMC sheet).
- **Targets:** `DGS2` (primary), `DGS1`; horizons h ∈ {1, 5, 22} business days. Leak-safe expanding-window walk-forward, n₀ = 20, baseline = ZeroChange.
- **Cross-check:** event-study of the surprise-only OOS residual regressed on stance.

## Data
- **197 aligned FOMC statements, 1999–2023** (201 BS announcement dates 1999+ with a valid orthogonalized surprise; 197 had a fetchable statement; 0 dropped at alignment). n = **177** OOS predictions per cell.
- Statement candidate dates, the release calendar, and the surprise control all derive from the **same BS announcement-date list**, so they align 1:1. (An earlier run keyed off the TDW `fomc_dates.csv` multi-doc calendar and collapsed to 55 contaminated releases — discarded; see lessons.)
- Stance distribution non-degenerate: mean −0.038, sd 0.247, range [−0.778, +0.615].
- Model: ungated mirror `tim9510019/FOMC-RoBERTa` (same base + label scheme as the gated `gtfintechlab/FOMC-RoBERTa`; weights not byte-verified vs the gated original — provenance caveat).

## Result — Nested OOS
```
 target  h    n   R2_base  R2_full      ΔR²   stance_t(IS)
   DGS2  1  177  -0.0397  -0.0426  -0.0029     -0.87
   DGS2  5  177  -0.0224  -0.0291  -0.0066     -0.60
   DGS2 22  177  -0.0434  -0.0595  -0.0162     +0.88
   DGS1  1  177  -0.0598  -0.0654  -0.0056     -0.92
   DGS1  5  177  -0.0481  -0.0605  -0.0124     -0.28
   DGS1 22  177  -0.0618  -0.0641  -0.0023     +2.52
```
**ΔR² < 0 at all six (target, horizon) cells** — adding stance *worsens* OOS fit everywhere. In-sample stance t-stats are insignificant except DGS1 h=22 (+2.52), and even that single in-sample signal fails to produce any OOS gain (ΔR² = −0.0023).

### Residual cross-check (residual after surprise ~ stance)
```
   DGS2 h1  slope -0.030 t -1.41 | h5 -0.049 t -1.29 | h22 +0.040 t +0.44
   DGS1 h1  slope -0.034 t -2.19 | h5 -0.042 t -1.33 | h22 +0.122 t +1.64
```
No coherent independent text effect: signs flip across horizons; the one "significant" cell (DGS1 h1, t −2.19) is the *wrong* sign for the hawkish-text → higher-yield hypothesis.

## Verdict (PRD §10 DoD §3)
**NO-GO.** FOMC statement *tone*, as measured by the FOMC-RoBERTa per-sentence mean, carries **no out-of-sample predictive value beyond the Bauer-Swanson surprise** on DGS2 or DGS1 at any tested horizon. This is a valid, publishable null: once you control for what the market already learned from the rate decision (the orthogonalized surprise), the statement's measured tone adds nothing.

Note the surprise-only base model itself has weak/negative OOS R² on multi-day Treasury changes (the BS surprise is calibrated to intraday futures windows, not daily DGS moves) — but the question here is the **marginal** contribution of text, which is unambiguously ≤ 0.

## Caveats (PRD §11)
- **Aggregation:** per-sentence-mean is crude; a null may reflect the aggregation, not the absence of information. A document-level or attention-weighted stance could differ.
- **Model provenance:** ungated mirror, not the byte-verified gated original. Re-run with the canonical `gtfintechlab/FOMC-RoBERTa` once HF access is granted (`--model` flag) to confirm; weights are expected to match.
- **Target/control mismatch:** BS surprise is intraday-calibrated; DGS daily changes are noisier. Intraday OIS targets (out of scope) would be a cleaner test bed.
- **Sample:** n = 177 OOS — adequate but magnitudes are noisy.

## Reproduce
```
pip install -e ".[dev,infer]"
# place data/raw/monetary-policy-surprises-data.xlsx (frbsf.org)
FRED_API_KEY=... python -m cbp.cli --mode phase1 --model tim9510019/FOMC-RoBERTa
# canonical (after HF access granted): drop --model to use gtfintechlab/FOMC-RoBERTa
```
