# Project
Undergrad Thesis – Spatio-temporal modeling of seizures (counts).

# Environment
- Python 3.12
- Use `venv` and `pip`. Entry: `make setup`, `make run`, `make eval`.

# Data
Expect `data/curated/events.parquet` with columns:
  region (str), date (YYYY-MM-DD), y (int count)
Optionally: `x_*` covariates.

# Tasks & tests
- `make test` must pass.
- Baseline 1 (Bayesian DSTM, PyMC) trains and produces `artifacts/dstm_forecast.parquet`.
- Baseline 2 (ESN) trains and produces `artifacts/esn_forecast.parquet`.
- `make eval` computes RMSE/MAE and WAIC (for DSTM) → `artifacts/metrics.json`.

# Conventions
- No file renames without explicit instruction.
- Prefer small, auditable patches.
- Never commit secrets. Use `.env.example`.

# Commands
- Setup: `make setup`
- Run DSTM: `make run MODEL=dstm`
- Run ESN: `make run MODEL=esn`
- Evaluate: `make eval`
