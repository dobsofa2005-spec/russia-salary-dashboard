"""Automated integrity checks for the dashboard data and calculations."""
import math

from analysis import calculate_indicators, load_raw, period_stats, unemployment_correlation


def test_complete_source():
    data = load_raw()
    assert len(data) == 24
    assert data.year.tolist() == list(range(2000, 2024))
    assert (data[['mining', 'manufacturing', 'education']] > 0).all().all()


def test_real_2023_equals_nominal():
    data = calculate_indicators(load_raw())
    subset = data[data.year == 2023]
    assert len(data) == 72
    assert all(abs(s.salary_real_2023 - s.salary_nominal) < 0.001 for s in subset.itertuples())


def test_2022_education_inflation_adjustment():
    data = calculate_indicators(load_raw())
    row = data[(data.year == 2022) & (data.sector_key == 'education')].iloc[0]
    expected = ((48400 / 43391) / (1 + 11.92 / 100) - 1) * 100
    assert math.isclose(row.real_growth_pct, expected, abs_tol=1e-10)
    assert row.real_growth_pct < 0


def test_growth_uses_previous_year_even_when_window_starts_later():
    data = calculate_indicators(load_raw())
    row = data[(data.year == 2009) & (data.sector_key == 'manufacturing')].iloc[0]
    assert math.isclose(row.nominal_growth_pct, (16583 / 16050 - 1) * 100, abs_tol=1e-8)
    stat = period_stats(data, 'education', 2000, 2023)
    assert stat['real_factor'] > 5


def test_correlation_ignores_2017_and_missing_first_year():
    data = calculate_indicators(load_raw())
    points, corr = unemployment_correlation(data, 'mining', 2000, 2023)
    assert len(points) == 22
    assert not (points.year == 2017).any()
    assert -1 <= corr <= 1
