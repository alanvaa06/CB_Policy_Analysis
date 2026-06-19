# Todo

> Open work only. Status: `pending` / `in_progress`. Done → move to `results.md`. Cap ~2.5k tokens.

Phase 0 — eval harness: DONE (T1-T13). Built + validated + live verdict (NO-GO on throwaway, harness trustworthy). See results.md/memory.md. Branch: phase-0-eval-harness (30 tests green).

Phase 0 merged to main + per-doc-type diagnostic DONE (see results.md).

Phase 1 — marginal-effect harness: DONE. Code (T1-T16) + live verdict (NO-GO) complete. Verdict: docs/results/2026-06-19-phase1-verdict.md. ΔR²<0 at all 6 (DGS2/DGS1 × h1/5/22), n=177, 197 statements. Branch phase-1-verdict (62 tests green). Open:
- [pending] merge phase-1-verdict → main.
- [pending] (optional, low-priority) canonical re-run with gated gtfintechlab/FOMC-RoBERTa once HF access granted (drop --model); expected to confirm null.
- [pending] (optional) if revisiting: document-level / attention-weighted stance instead of per-sentence-mean; intraday OIS target (cleaner than daily DGS).
