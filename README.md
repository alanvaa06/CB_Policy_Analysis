# CB_Policy_Analysis

Agentic research pipeline for **Federal Reserve / FOMC policy text analysis** — retrieve
FOMC materials, classify monetary-policy stance (hawkish/dovish), and construct
out-of-sample, look-ahead-safe time-series signals to measure forward-guidance
effectiveness against market and policy rates.

**Track:** live real-time signal, research-grade. Python + pytest.

## Roadmap

- **Phase 0 — Eval harness first** (current): a tested, leak-proof walk-forward
  harness scoring a release-aligned stance signal against `DGS2` (2y Treasury,
  primary) and `EFFR` (secondary), exercised by a throwaway stance series
  (Trillion Dollar Words dataset, 1996–2022). Proves out-of-sample predictive
  power exists before building fetch/classify infra. See
  [docs/prd/001-fomc-eval-harness.md](docs/prd/001-fomc-eval-harness.md).
- **Phase 1 — End-to-end thin slice**: real fetch → FOMC-RoBERTa stance →
  signal on the live-feasible corpus, dropped into the proven harness.
- **Later**: counterfactual (Doh-Song-Yang) and transcript-uncertainty
  (Cieslak et al.) extensions — historical-research track only (~5yr data lag).

## Working memory

Durable project context lives in `docs/context/` (read on demand, not bulk-loaded).
See `CLAUDE.md` for conventions.
