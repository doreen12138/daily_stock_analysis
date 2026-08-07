from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_provider.realtime_types import ChipDistribution


REQUIRED_COLUMNS = ("date", "high", "low", "close", "volume")


def analyze_chip_peak(
    daily_data: pd.DataFrame,
    stock_code: str,
    window: int = 120,
    bins: int = 240,
) -> Optional[ChipDistribution]:
    if daily_data is None or not isinstance(daily_data, pd.DataFrame):
        return None
    if any(column not in daily_data.columns for column in REQUIRED_COLUMNS):
        return None
    if window < 2 or bins < 20:
        return None

    frame = daily_data.loc[:, REQUIRED_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[
        (frame["high"] > 0)
        & (frame["low"] > 0)
        & (frame["close"] > 0)
        & (frame["volume"] > 0)
        & (frame["high"] >= frame["low"])
        & (frame["close"] >= frame["low"])
        & (frame["close"] <= frame["high"])
    ]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if len(frame) < window:
        return None
    frame = frame.tail(window).reset_index(drop=True)

    price_min = float(frame["low"].min())
    price_max = float(frame["high"].max())
    if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
        return None

    prices = np.linspace(price_min, price_max, bins, dtype=float)
    weights = np.zeros(bins, dtype=float)
    rolling_volume = frame["volume"].rolling(20, min_periods=1).mean()

    for row_index, row in frame.iterrows():
        relative_volume = float(row["volume"] / rolling_volume.iloc[row_index])
        migration = float(np.clip(0.025 * relative_volume, 0.0025, 0.20))
        weights *= 1.0 - migration

        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        typical = (low + high + close) / 3.0
        active = (prices >= low) & (prices <= high)
        allocation = np.zeros(bins, dtype=float)
        if not np.any(active) or high == low:
            allocation[int(np.abs(prices - typical).argmin())] = 1.0
        else:
            span = max(high - low, np.finfo(float).eps)
            allocation[active] = np.maximum(
                0.05,
                1.0 - np.abs(prices[active] - typical) / span,
            )
        allocation_sum = float(allocation.sum())
        if allocation_sum <= 0:
            continue
        weights += migration * allocation / allocation_sum

    total_weight = float(weights.sum())
    if not np.isfinite(total_weight) or total_weight <= 0:
        return None
    weights /= total_weight

    cost_70_low = _weighted_quantile(prices, weights, 0.15)
    cost_70_high = _weighted_quantile(prices, weights, 0.85)
    cost_90_low = _weighted_quantile(prices, weights, 0.05)
    cost_90_high = _weighted_quantile(prices, weights, 0.95)
    avg_cost = float(np.dot(prices, weights))
    current_price = float(frame.iloc[-1]["close"])
    profit_ratio = float(weights[prices < current_price].sum())
    peak_index = int(weights.argmax())
    peak_price = float(prices[peak_index])
    peak_ratio = _peak_window_ratio(prices, weights, peak_price)
    positive_weights = weights[weights > 0]
    peak_strength = float(weights[peak_index] / positive_weights.mean())

    return ChipDistribution(
        code=stock_code,
        date=frame.iloc[-1]["date"].date().isoformat(),
        source="local_ohlcv_estimate",
        profit_ratio=round(profit_ratio, 6),
        avg_cost=round(avg_cost, 4),
        cost_90_low=round(cost_90_low, 4),
        cost_90_high=round(cost_90_high, 4),
        concentration_90=round(_concentration(cost_90_low, cost_90_high), 6),
        cost_70_low=round(cost_70_low, 4),
        cost_70_high=round(cost_70_high, 4),
        concentration_70=round(_concentration(cost_70_low, cost_70_high), 6),
        peak_price=round(peak_price, 4),
        peak_ratio=round(peak_ratio, 6),
        peak_strength=round(peak_strength, 4),
        secondary_peaks=_secondary_peaks(prices, weights, peak_index),
        sample_days=len(frame),
        calculation_method="ohlcv_volume_decay_v1",
        is_estimated=True,
    )


def _weighted_quantile(prices: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    cumulative = np.cumsum(weights)
    return float(np.interp(quantile, cumulative, prices))


def _concentration(low: float, high: float) -> float:
    denominator = low + high
    if denominator <= 0:
        return 0.0
    return float((high - low) / denominator)


def _peak_window_ratio(prices: np.ndarray, weights: np.ndarray, peak_price: float) -> float:
    tolerance = max(abs(peak_price) * 0.01, float(prices[1] - prices[0]))
    return float(weights[np.abs(prices - peak_price) <= tolerance].sum())


def _secondary_peaks(
    prices: np.ndarray,
    weights: np.ndarray,
    primary_index: int,
) -> list[dict[str, float]]:
    candidates = [
        index
        for index in range(1, len(weights) - 1)
        if weights[index] > weights[index - 1] and weights[index] >= weights[index + 1]
    ]
    minimum_distance = max(3, len(weights) // 30)
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: float(weights[item]), reverse=True):
        if abs(index - primary_index) < minimum_distance:
            continue
        if any(abs(index - existing) < minimum_distance for existing in selected):
            continue
        selected.append(index)
        if len(selected) == 2:
            break
    return [
        {
            "price": round(float(prices[index]), 4),
            "ratio": round(_peak_window_ratio(prices, weights, float(prices[index])), 6),
            "strength": round(float(weights[index] / weights[primary_index]), 4),
        }
        for index in selected
    ]
