# Todo

> Open work only. Status: `pending` / `in_progress`. Done → move to `results.md`. Cap ~2.5k tokens.

Phase 0 — eval harness: DONE (T1-T13). Built + validated + live verdict (NO-GO on throwaway, harness trustworthy). See results.md/memory.md. Branch: phase-0-eval-harness (30 tests green).

Phase 0 merged to main + per-doc-type diagnostic DONE (see results.md).

Phase 1 — marginal-effect harness: CODE DONE (PRD 002 + plan + T1-T16, branch phase-1-marginal-effect, 59 tests green; see results.md/memory.md). Open:
- [pending] Phase 1 live run + go/no-go verdict (DoD §2-3): install [infer] extra; download BS surprise xlsx to data/raw/monetary-policy-surprises-data.xlsx + CONFIRM orthogonal col-name (default MP1_orthogonal) and 1999 coverage (else SF Fed USMPD / BS-2023 set); run `python -m cbp.cli --mode phase1`; write verdict (does statement stance add OOS R² beyond BS surprise on DGS2?).
- [pending] merge phase-1-marginal-effect → main once verdict recorded.
