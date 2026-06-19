# Memory

> Architecture decisions. One line each. Format: `# decision: sentence`. Cap ~11k tokens.

# constraint: live FOMC signals limited to PUBLIC real-time corpus (statements May-1999+, minutes ~3wk lag, press-conf transcripts 2011+, speeches); meeting TRANSCRIPTS + Tealbook/Bluebook alternative statements carry ~5yr release lag → historical-research-only, not live signal.
# decision (pending): backbone model candidate = FOMC-RoBERTa "Trillion Dollar Words" (gtfintechlab, ACL 2023) — but CC BY-NC 4.0 = non-commercial only; blocks commercial deployment without relicense.
# finding: all academic "predictive" results to date are in-sample/correlational (Lucca-Trebbi ~30bp, IMF ~20bp); no common-corpus OUT-OF-SAMPLE FOMC head-to-head benchmark exists → that gap is the main innovation white-space.
