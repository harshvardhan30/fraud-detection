#!/usr/bin/env bash
# End-to-end pipeline. Takes ~3-4 minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")"
python src/generate_data.py      # data/transactions_synthetic.csv (deterministic, seed 42)
python src/eda.py                # reports/eda.md
python src/train.py              # artifacts/ + reports/model_results.md
python src/validate.py           # reports/release_checks.md
python src/monitor.py            # reports/monitoring_report.md
python -m pytest -q              # leakage / parity / validation tests
