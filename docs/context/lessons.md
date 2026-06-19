# Lessons

> Rules from corrections. Friction only — never repeat a logged mistake. List format. Cap ~7k tokens.

- Test fixtures must mirror real data SHAPE, not just values. Offline aligner/CLI fixtures built the market index tz-aware (`tz="UTC"`) but real FRED data is tz-naive → a tz-comparison bug (InvalidComparison) passed all 27 offline tests and only surfaced in the live run. When mocking an external source, replicate its dtypes/tz/null conventions, and keep at least one integration test on the real shape.
- Interpreting signal metrics: a low sign-test p-value with hit-rate BELOW 0.5 is a significant WRONG-direction relationship, not evidence of signal. Always read OOS R² and hit-rate direction together, never p-value alone.
