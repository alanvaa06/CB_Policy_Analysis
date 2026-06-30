# Todo

> Open work only. Status: `pending` / `in_progress`. Done → move to `results.md`. Cap ~2.5k tokens.

Phase 0 — eval harness: DONE (T1-T13). Built + validated + live verdict (NO-GO on throwaway, harness trustworthy). See results.md/memory.md. Branch: phase-0-eval-harness (30 tests green).

Phase 0 merged to main + per-doc-type diagnostic DONE (see results.md).

Phase 1 — marginal-effect harness: DONE + merged (112aed2) + canonical-confirmed (2026-06-29). Verdict NO-GO: docs/results/2026-06-19-phase1-verdict.md. ΔR²<0 all 6 cells, n=177; canonical gtfintechlab/FOMC-RoBERTa = mirror to 4 decimals (provenance caveat CLOSED). Open:
- [pending] (optional) if revisiting: document-level / attention-weighted stance instead of per-sentence-mean; intraday OIS target (cleaner than daily DGS).

Phase 2a — lexicon tone baseline: DONE (branch phase-2a-lexicon-tone, 81 tests). Verdict NO-GO (regime confound): docs/results/2026-06-29-lexicon-baseline-verdict.md. Open:
- [pending] merge/PR phase-2a-lexicon-tone → main (await user).
- [pending] (optional) if pursuing the h=1 near-miss: intraday OIS target + explicit regime-controlled spec; expected to confirm confound.
