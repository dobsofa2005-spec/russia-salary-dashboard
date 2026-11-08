"""Offline, deterministic salary and inflation computations for the dashboard."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "salaries_2000_2023.csv"

SECTORS = {
    "mining": "Добыча полезных ископаемых",
    "manufacturing": "Обрабатывающие производства",
    "education": "Образование",
}

SHORT_SECTORS = {
    "mining": "Добыча",
    "manufacturing": "Обработка",
    "education": "Образование",
}

COLORS = {
    "mining": "#59B9F6",
    "manufacturing": "#F7B56C",
    "education": "#A8A0F2",
}


def load_raw(path=DATA_PATH):
    """Load local, versioned source data without web requests."""
    frame = pd.read_csv(path)
    expected = ["year", *SECTORS, "inflation_pct", "unemployment_pct"]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные столбцы: {sorted(missing)}")
    if frame[expected].isna().any().any():
        raise ValueError("В исходной таблице есть пропущенные значения.")
    frame = frame[expected].sort_values("year").reset_index(drop=True)
    if frame["year"].tolist() != list(range(2000, 2024)):
        raise ValueError("Ожидаются все годы от 2000 до 2023 без пропусков и дублей.")
    if (frame[list(SECTORS)] <= 0).any().any():
        raise ValueError("Зарплаты должны быть положительными.")
    if (frame["inflation_pct"] <= -100).any():
        raise ValueError("Инфляция не может быть меньше или равна -100%.")
    return frame


def calculate_indicators(raw):
    """Return a long-form table suitable for plotting and downloading.

    CPI is December-on-December; price index has 2000 as the reference year.
    Hence 2000 inflation is not accumulated, and estimates in '2023 rubles'
    are proxies, not official annual-average-price wages.
    """
    df = raw.copy().sort_values("year").reset_index(drop=True)
    factors = (1 + df["inflation_pct"] / 100).to_numpy(dtype=float)
    factors[0] = 1.0
    df["price_index"] = np.cumprod(factors)
    index_2023 = float(df.loc[df["year"] == 2023, "price_index"].iloc[0])
    df["unemployment_change_pp"] = df["unemployment_pct"].diff()

    stacked = []
    for key, label in SECTORS.items():
        part = df[["year", "inflation_pct", "unemployment_pct",
                   "unemployment_change_pp", "price_index"]].copy()
        part["sector_key"] = key
        part["sector"] = label
        part["salary_nominal"] = df[key].astype(float)
        part["salary_real_2023"] = part["salary_nominal"] * index_2023 / part["price_index"]
        part["nominal_growth_pct"] = part["salary_nominal"].pct_change() * 100
        part["real_growth_pct"] = (
            part["salary_nominal"].div(part["salary_nominal"].shift())
            .div(1 + part["inflation_pct"] / 100).sub(1) * 100
        )
        stacked.append(part)
    return pd.concat(stacked, ignore_index=True)


def period_stats(data, sector_key, first_year, last_year):
    """Summarize nominal and CPI-adjusted changes between two selected years."""
    sector = data.loc[data["sector_key"] == sector_key].set_index("year")
    first = sector.loc[first_year]
    last = sector.loc[last_year]
    nom_growth = (last["salary_nominal"] / first["salary_nominal"] - 1) * 100
    real_growth = (last["salary_real_2023"] / first["salary_real_2023"] - 1) * 100
    return {
        "salary_start": float(first["salary_nominal"]),
        "salary_end": float(last["salary_nominal"]),
        "nominal_growth_pct": float(nom_growth),
        "real_growth_pct": float(real_growth),
        "real_factor": float(last["salary_real_2023"] / first["salary_real_2023"]),
    }


def unemployment_correlation(data, sector_key, start, end, exclude_shocks=False):
    """Pearson r of annual *changes*, excluding 2017 classification boundary.

    Filter years only after YoY computation, so period start still has a proper
    comparison to the preceding year, if available.
    """
    sector = data.loc[
        (data["sector_key"] == sector_key)
        & data["year"].between(start, end)
        & (data["year"] != 2017)
    ].copy()
    if exclude_shocks:
        sector = sector.loc[~sector["year"].isin([2009, 2020, 2022])]
    sector = sector.dropna(subset=["unemployment_change_pp", "real_growth_pct"])
    if len(sector) < 3 or sector["unemployment_change_pp"].nunique() < 2:
        return sector, np.nan
    r = sector["unemployment_change_pp"].corr(sector["real_growth_pct"])
    return sector, float(r)
