# Phase 2a Verdict — Does a transparent hawk/dove lexicon add value beyond the policy surprise (and vs FOMC-RoBERTa)?

**Date:** 2026-06-29 · **Status:** decided · **Result: NO-GO as a robust signal (instructive near-miss)** · Spec: `docs/prd/003-lexicon-tone-baseline.md`

## Question
Does a transparent, document-level hawkish/dovish **word-count** tone (policy-stance lexicon)
carry out-of-sample predictive information for market rates **beyond** the Bauer-Swanson
orthogonalized surprise — and how does it compare to the FOMC-RoBERTa stance (Phase 1, which was a flat null)?

## Method
- **Stance:** corpus-validated policy-stance lexicon (`data/lexicons/hawk_dove.json`):
  hawk `{tighten, restrictive, firming}`, dove `{accommodative, easing, stimulus}`. Document-level
  net tone `(n_hawk − n_dove)/(n_hawk + n_dove)`. Confound words (`inflation` boilerplate, valence
  `weak`/`downside`/`robust`, context-flippers, dead seeds) excluded by design + unit test.
- **Control / targets / horizons / harness:** identical to Phase 1 — BS `MPS_ORTH`, DGS2/DGS1,
  h∈{1,5,22}, leak-safe expanding walk-forward n₀=20, nested OOS (ΔR²), residual event-study.
- 197 statements, n=177 OOS per cell (same sample as Phase 1).

## Result — Nested OOS (lexicon)
```
 target  h    n   R2_base  R2_full      ΔR²   stance_t(IS)
   DGS2  1  177  -0.0397  -0.0281  +0.0116     -2.46
   DGS2  5  177  -0.0224  -0.0376  -0.0152     -0.57
   DGS2 22  177  -0.0434  -0.0796  -0.0363     -0.47
   DGS1  1  177  -0.0598  -0.0046  +0.0552     -3.64
   DGS1  5  177  -0.0481  -0.0486  -0.0004     -1.62
   DGS1 22  177  -0.0618  -0.1138  -0.0520     -0.46
```
Unlike RoBERTa (ΔR²<0 at all six), the lexicon shows **ΔR²>0 at h=1** for both targets, with a
significant *negative* in-sample stance coefficient (DGS1 h1 t=−3.64) and a significant residual
event-study (DGS1 h1: slope −0.0162, t −3.59, r²=0.068). At h=5/22 it is negative/null.

## Hardening — the h=1 positive does NOT survive scrutiny
Reproducible: `python scripts/lexicon_confound_check.py`.

1. **Near-degenerate, regime-correlated measure.** Stance distribution ≈ 3-level
   {−1: 100, 0: 66, +1: 28}, mean −0.371; **corr(stance, front-end rate level) = +0.65**,
   corr(stance, RoBERTa) = 0.41. Statements with a dove word but no hawk word (or vice-versa)
   dominate → the measure is essentially an **easing-era vs hiking-era vocabulary indicator**.
2. **Regime control absorbs it.** Stance's marginal OOS R² over `[surprise]` vs over
   `[surprise, rate-level]`: DGS2_h1 +0.0116 → **+0.0013** (gone); DGS1_h1 +0.0552 → **+0.0162**
   (shrinks ~3×). The rate level alone adds nothing (C−A<0) — i.e. stance is partly a noisier
   proxy for cycle position.
3. **Era-fragile.** Era jackknife of DGS1_h1 (ΔR² of stance over surprise): dropping ZLB/2016-19/
   2020-23 keeps it +0.04…+0.06, but **dropping pre-2008 flips it to −0.0407**. The whole positive
   result is carried by 1999–2007.
4. **Counterintuitive sign.** Negative coefficient: dovish-vocabulary statements → *higher* next-day
   yields. Consistent with regime/level dynamics, not a hawkish-text → higher-yield news channel.

## Verdict
**NO-GO as a robust marginal signal.** The apparent h=1 edge is concentrated at the 1-day front-end,
sign-flips out of sample when the pre-2008 sub-period is removed, is largely/entirely absorbed by a
policy-regime control, and is produced by a near-degenerate regime-correlated 3-level measure with a
backwards sign. It is **not** evidence that transparent statement tone predicts rates beyond the surprise.

**But an instructive near-miss, two ways RoBERTa was not:**
- The transparent baseline *surfaced* a front-end h=1 association the transformer entirely missed
  (RoBERTa: flat null at every cell).
- Because it is transparent, we could *diagnose* that association as **regime vocabulary**, not text
  news — a diagnosis impossible with the black-box model. Interpretability is the deliverable.

## Lexicon vs RoBERTa
`docs/results/figures/tone-timeseries.png`: tone(t) overlay + scatter, corr = 0.41 — the two measures
agree only moderately; the lexicon is the cruder, regime-tilted of the two.

## Caveats (PRD 003 §11)
- **No negation / residual polarity flip** ("removing accommodation" is hawkish) — v1 floor.
- **Confound exclusions** keep the list small (fires on 66% of statements; 34% neutral 0.0).
- **Degenerate distribution** ≈ {−1,0,+1} → effectively a 3-group contrast, not a graded tone.
- Same BS-control caveat as Phase 1 (intraday-calibrated surprise vs daily DGS).
- Small OOS sample (n=177); the h=1 result's era-sensitivity is itself a small-sample signal.

## Reproduce
```
pip install -e ".[dev,infer]"
FRED_API_KEY=... python -m cbp.cli --mode phase1 --tone-method lexicon
python scripts/plot_tone_timeseries.py          # tone(t) figure
python scripts/lexicon_confound_check.py        # regime/era hardening
```
