# Time series forecasting demo — clinic operations

**Synthetic daily surgery counts** (multiple specialties × multiple hospitals) for teaching and demos: patterns, **metric choice**, **model choice**, **holiday effects**, and **feature engineering**. Suitable for talks and tech posts; **no real patient data**—only generated data.

## Contents

| Area | What you get |
|------|----------------|
| **Data** | Config-driven generator: weekly/yearly seasonality, holidays, trends, hospital scaling, specialty-specific noise. |
| **Code** | `src/data_generator.py`, `src/evaluation.py` (time split + MAE / RMSE / MAPE). |
| **Notebooks** | Five runnable notebooks, one concept each (see below). |

## Prerequisites

- Python **3.9+**
- A virtual environment (recommended)

## Quick start

```bash
git clone <your-repo-url>
cd Times-series-forecast-concept-demo
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_generator      # writes data/surgeries.csv
```

Optional CLI flags:

```bash
python -m src.data_generator --config config/data_config.yaml --output data/surgeries.csv --seed 42
```

**Note:** `data/*.csv` is gitignored so the repo stays small. After clone, run the generator once (or let notebook 01 create the file if missing).

## Notebooks (run from project root)

Run **01 → 05** in order. Paths assume the repo root is the working directory; notebooks also resolve the project root if you open them from `notebooks/`.

| # | Notebook | Concept |
|---|----------|---------|
| 01 | [`notebooks/01_patterns_and_eda.ipynb`](notebooks/01_patterns_and_eda.ipynb) | EDA: patterns by specialty/hospital, weekday & monthly seasonality, holidays. |
| 02 | [`notebooks/02_metric_selection.ipynb`](notebooks/02_metric_selection.ipynb) | MAE vs RMSE vs MAPE; low-volume / near-zero actuals. |
| 03 | [`notebooks/03_model_comparison.ipynb`](notebooks/03_model_comparison.ipynb) | Naive, seasonal naive, ARIMA, Prophet, Ridge + lags. |
| 04 | [`notebooks/04_holiday_effects.ipynb`](notebooks/04_holiday_effects.ipynb) | Prophet without vs with holiday regressors. |
| 05 | [`notebooks/05_feature_engineering.ipynb`](notebooks/05_feature_engineering.ipynb) | Minimal vs rich features; Ridge coefficients. |

## Project layout

```
config/data_config.yaml    # Hospitals, specialties, dates, seasonality, holidays
src/data_generator.py      # Synthetic surgery counts → CSV
src/evaluation.py          # train_val_split, mae, rmse, mape
notebooks/                 # 01–05 concept demos
data/                      # Generated surgery counts CSV (not committed)
```

## Customizing the synthetic data

Edit [`config/data_config.yaml`](config/data_config.yaml): add hospitals/specialties, change `date_range`, `weekend_ratio`, `trend_per_year`, `noise_scale`, `holidays`, and `hospital_scale`. Regenerate with `python -m src.data_generator`.

## Troubleshooting

- **Prophet / CmdStan**: First `Prophet.fit` can be slow while CmdStan compiles or runs; subsequent runs are usually faster.
- **Import errors**: Run notebooks and the CLI from the **repository root** (or ensure `sys.path` includes the root, as the notebooks do).

## License

MIT.
