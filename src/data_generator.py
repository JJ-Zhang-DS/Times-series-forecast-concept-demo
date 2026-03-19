"""
Synthetic daily surgery counts for multiple hospitals and specialties.
Generates config-driven, reproducible time series with seasonality, holidays, and trend.
"""

from pathlib import Path
import pandas as pd
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


def _load_config(config_path: str | Path | None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "data_config.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML config. Install with: pip install PyYAML")
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_dates(config: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    dr = config.get("date_range")
    if not dr or "start" not in dr or "end" not in dr:
        raise ValueError("config.date_range must have 'start' and 'end' (YYYY-MM-DD)")
    start = pd.Timestamp(dr["start"])
    end = pd.Timestamp(dr["end"])
    if start >= end:
        raise ValueError(f"date_range start must be before end; got start={start}, end={end}")
    return start, end


def _build_holiday_set(config: dict) -> set[tuple[int, int]]:
    """Set of (month, day) for holiday dates (all years)."""
    holidays = config.get("holidays") or []
    out = set()
    for h in holidays:
        part = h.get("date") or h.get("date_str")
        if not part or len(part) != 5 or part[2] != "-":
            raise ValueError(f"Each holiday must have 'date' as MM-DD; got {h}")
        try:
            m, d = int(part[:2]), int(part[3:])
        except ValueError:
            raise ValueError(f"Invalid holiday date MM-DD: {part}")
        if not (1 <= m <= 12 and 1 <= d <= 31):
            raise ValueError(f"Invalid month/day for holiday: {part}")
        out.add((m, d))
    return out


def generate_surgery_counts(
    config_path: str | Path | None = None,
    output_path: str | Path | None = "data/surgeries.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily surgery counts and optionally write to CSV.

    Args:
        config_path: Path to data_config.yaml. If None, uses config/data_config.yaml.
        output_path: If set, writes DataFrame to this path (CSV). If None, does not write.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: date, hospital_id, specialty_id, hospital_name, specialty_name, surgery_count.
    """
    rng = np.random.default_rng(seed)
    config = _load_config(config_path)
    start, end = _parse_dates(config)
    holiday_dates = _build_holiday_set(config)
    holiday_ratio = float(config.get("holiday_ratio", 0.2))
    hospital_scale = config.get("hospital_scale") or {}
    yearly = config.get("yearly_seasonality")
    if yearly and len(yearly) != 12:
        raise ValueError("yearly_seasonality must have exactly 12 elements (Jan–Dec)")

    hospitals = config.get("hospitals")
    specialties = config.get("specialties")
    if not hospitals:
        raise ValueError("config.hospitals must be a non-empty list")
    if not specialties:
        raise ValueError("config.specialties must be a non-empty list")

    dates = pd.date_range(start=start, end=end, freq="D")
    n_days = len(dates)
    rows = []

    for hosp in hospitals:
        hid = hosp.get("id")
        hname = hosp.get("name") or hid
        scale = hospital_scale.get(hid, 1.0)

        for spec in specialties:
            sid = spec.get("id")
            sname = spec.get("name") or sid
            base = float(spec.get("base_daily_mean", 10))
            weekend_ratio = float(spec.get("weekend_ratio", 0.5))
            trend_per_year = float(spec.get("trend_per_year", 0.0))
            noise_scale = float(spec.get("noise_scale", 0.15))

            # Daily level: base * hospital_scale * yearly * weekly * (1 + trend * t) * holiday
            t = np.arange(n_days, dtype=float)
            years_elapsed = t / 365.25
            trend_mult = 1.0 + trend_per_year * years_elapsed

            month_idx = dates.month.values - 1
            if yearly:
                yearly_mult = np.array([yearly[i] for i in month_idx])
            else:
                yearly_mult = np.ones(n_days)

            weekday = dates.dayofweek.values  # 0=Mon .. 6=Sun
            weekend = (weekday >= 5).astype(float)
            weekly_mult = 1.0 - (1.0 - weekend_ratio) * weekend

            holiday_mult = np.ones(n_days)
            for i, d in enumerate(dates):
                if (d.month, d.day) in holiday_dates:
                    holiday_mult[i] = holiday_ratio

            mu = base * scale * yearly_mult * weekly_mult * trend_mult * holiday_mult
            mu = np.maximum(mu, 0.01)

            # Multiplicative noise then Poisson-style rounding to get integers
            noise = 1.0 + rng.normal(0, noise_scale, size=n_days)
            mu_noisy = mu * np.maximum(noise, 0.1)
            counts = np.round(mu_noisy).astype(int)
            counts = np.maximum(counts, 0)

            for i in range(n_days):
                rows.append({
                    "date": dates[i],
                    "hospital_id": hid,
                    "specialty_id": sid,
                    "hospital_name": hname,
                    "specialty_name": sname,
                    "surgery_count": int(counts[i]),
                })

    df = pd.DataFrame(rows)
    if (df["surgery_count"] < 0).any():
        raise AssertionError("Generated surgery_count contained negative values; check generator logic.")
    if df["surgery_count"].dtype not in (np.int64, np.int32):
        df["surgery_count"] = df["surgery_count"].astype(int)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, date_format="%Y-%m-%d")

    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate synthetic surgery counts CSV.")
    p.add_argument("--config", default=None, help="Path to data_config.yaml")
    p.add_argument("--output", default="data/surgeries.csv", help="Output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()
    generate_surgery_counts(config_path=args.config, output_path=args.output, seed=args.seed)
    print(f"Wrote {args.output}")
