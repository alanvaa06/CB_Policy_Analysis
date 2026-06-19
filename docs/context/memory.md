# Memory

> Architecture decisions. One line each. Format: `# decision: sentence`. Cap ~11k tokens.

# constraint: live FOMC signals limited to PUBLIC real-time corpus (statements May-1999+, minutes ~3wk lag, press-conf transcripts 2011+, speeches); meeting TRANSCRIPTS + Tealbook/Bluebook alternative statements carry ~5yr release lag → historical-research-only, not live signal.
# decision (pending): backbone model candidate = FOMC-RoBERTa "Trillion Dollar Words" (gtfintechlab, ACL 2023) — but CC BY-NC 4.0 = non-commercial only; blocks commercial deployment without relicense.
# finding: all academic "predictive" results to date are in-sample/correlational (Lucca-Trebbi ~30bp, IMF ~20bp); no common-corpus OUT-OF-SAMPLE FOMC head-to-head benchmark exists → that gap is the main innovation white-space.
# decision: Phase 0 eval harness BUILT + VALIDATED (offline synthetic-recovery + null-rejection; live confounded signal → no false positive). Apparatus is trustworthy; build on it for Phase 1.
# finding: combined-doc-type TDW stance (minutes+pressers+speeches averaged per date) has NO usable OOS forward power on DGS2 (negative OOS R², sub-0.5 hit) — confounded by mixed doc timing/role. Phase 1 must use a CLEAN single-doc-type, market-timed stance (statements or pressers), NOT a doc-mixed average.
# lesson-source: offline test fixtures used tz-aware market index; real FRED is tz-naive → integration bug only the live run caught. Prefer fixtures that mirror real data shape.
