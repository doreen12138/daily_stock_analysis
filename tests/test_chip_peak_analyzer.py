import numpy as np
import pandas as pd

from src.services.chip_peak_analyzer import analyze_chip_peak


def _bars(days: int = 120, center: float = 10.0) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    close = center + np.sin(np.arange(days) / 8.0) * 0.35
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.05,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "volume": np.full(days, 1_000_000.0),
            "amount": close * 1_000_000.0,
        }
    )


def test_exactly_120_valid_bars_produces_deterministic_distribution() -> None:
    data = _bars()

    first = analyze_chip_peak(data, "600519")
    second = analyze_chip_peak(data, "600519")

    assert first is not None
    assert second is not None
    assert first.to_dict() == second.to_dict()
    assert first.sample_days == 120
    assert first.source == "local_ohlcv_estimate"
    assert first.calculation_method == "ohlcv_volume_decay_v1"
    assert first.is_estimated is True
    assert first.cost_90_low <= first.cost_70_low <= first.avg_cost
    assert first.avg_cost <= first.cost_70_high <= first.cost_90_high
    assert first.cost_90_low <= first.peak_price <= first.cost_90_high
    assert 0 <= first.profit_ratio <= 1
    assert first.concentration_70 <= first.concentration_90


def test_more_than_120_bars_uses_latest_window() -> None:
    old = _bars(20, center=5.0)
    recent = _bars(120, center=20.0)
    recent["date"] = pd.bdate_range("2025-02-03", periods=120)
    combined = pd.concat([old, recent], ignore_index=True)

    result = analyze_chip_peak(combined, "000001")

    assert result is not None
    assert result.sample_days == 120
    assert result.avg_cost > 15
    assert result.peak_price > 15


def test_fewer_than_120_valid_bars_returns_none() -> None:
    assert analyze_chip_peak(_bars(119), "600519") is None


def test_zero_volume_invalid_prices_and_nulls_are_excluded() -> None:
    data = _bars(123).sample(frac=1.0, random_state=7).reset_index(drop=True)
    data.loc[0, "volume"] = 0
    data.loc[1, "close"] = np.nan
    data.loc[2, "high"] = -1

    result = analyze_chip_peak(data, "600519")

    assert result is not None
    assert result.sample_days == 120


def test_two_price_clusters_expose_secondary_peak() -> None:
    low_cluster = _bars(60, center=10.0)
    high_cluster = _bars(60, center=20.0)
    high_cluster["date"] = pd.bdate_range("2025-04-01", periods=60)
    data = pd.concat([low_cluster, high_cluster], ignore_index=True)

    result = analyze_chip_peak(data, "600519")

    assert result is not None
    assert result.secondary_peaks
    assert all(item["price"] > 0 for item in result.secondary_peaks)
    assert all(0 < item["strength"] <= 1 for item in result.secondary_peaks)


def test_profit_ratio_increases_when_current_price_is_above_cost_distribution() -> None:
    high_price = _bars()
    high_price.loc[119, ["high", "close"]] = [12.2, 12.0]
    low_price = _bars()
    low_price.loc[119, ["low", "close"]] = [7.8, 8.0]

    high_result = analyze_chip_peak(high_price, "600519")
    low_result = analyze_chip_peak(low_price, "600519")

    assert high_result is not None
    assert low_result is not None
    assert high_result.profit_ratio > low_result.profit_ratio
