# Todo

> Open work only. Status: `pending` / `in_progress`. Done → move to `results.md`. Cap ~2.5k tokens.

Phase 0 — FOMC stance eval harness. Plan: docs/plans/2026-06-18-fomc-eval-harness.md (TDD, one commit/task).

- [pending] T1 project bootstrap (pyproject, cbp pkg, pytest sanity)
- [pending] T2 Config frozen dataclass
- [pending] T3 io/store parquet round-trip
- [pending] T4 data/fred parser + thin client
- [pending] T5 data/fomc_calendar (2pm ET → UTC)
- [pending] T6 data/stance loader joined to calendar
- [pending] T7 align/aligner — leak-safe forward target (CORE)
- [pending] T8 models/baseline (ZeroChange, MeanModel, SimpleOLS)
- [pending] T9 eval/metrics (oos_r2, rmse, hit_rate, sign_test)
- [pending] T10 eval/walkforward — expanding OOS + leak guard
- [pending] T11 eval/eventstudy diagnostic
- [pending] T12 cli end-to-end + offline integration test
- [pending] T13 full suite + live run + go/no-go verdict
