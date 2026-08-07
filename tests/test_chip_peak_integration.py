from unittest.mock import patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    from tests.litellm_stub import ensure_litellm_stub

    ensure_litellm_stub()

from data_provider.realtime_types import ChipDistribution
from src.agent.executor import AgentExecutor
from src.agent.tools.data_tools import _handle_get_chip_distribution
from src.analyzer import GeminiAnalyzer
from src.schemas.report_schema import ChipStructure


def _local_chip() -> ChipDistribution:
    return ChipDistribution(
        code="600519",
        date="2026-08-07",
        source="local_ohlcv_estimate",
        profit_ratio=0.62,
        avg_cost=10.2,
        cost_90_low=9.4,
        cost_90_high=11.0,
        concentration_90=0.0784,
        cost_70_low=9.8,
        cost_70_high=10.6,
        concentration_70=0.0392,
        peak_price=10.1,
        peak_ratio=0.18,
        peak_strength=2.4,
        secondary_peaks=[{"price": 10.8, "ratio": 0.08, "strength": 0.6}],
        sample_days=120,
        calculation_method="ohlcv_volume_decay_v1",
        is_estimated=True,
    )


def test_legacy_prompt_contains_local_peak_cost_zones_and_estimate_notice() -> None:
    with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
        analyzer = GeminiAnalyzer()
    chip = _local_chip()
    context = {
        "code": "600519",
        "stock_name": "测试股票",
        "date": "2026-08-07",
        "today": {"close": 10.5},
        "chip": {**chip.to_dict(), "chip_status": "健康"},
    }

    with patch.object(analyzer, "_get_skill_prompt_sections", return_value=("", "", True)):
        prompt = analyzer._format_prompt(context, "测试股票")

    assert "主筹码峰价格" in prompt
    assert "9.4 - 11.0" in prompt
    assert "9.8 - 10.6" in prompt
    assert "120 个交易日" in prompt
    assert "local_ohlcv_estimate / ohlcv_volume_decay_v1" in prompt
    assert "不代表账户级真实持仓分布" in prompt


def test_agent_chip_tool_returns_peak_and_calculation_metadata() -> None:
    manager = type("Manager", (), {"get_chip_distribution": lambda self, code: _local_chip()})()

    with patch("src.agent.tools.data_tools._get_fetcher_manager", return_value=manager):
        payload = _handle_get_chip_distribution("600519")

    assert payload["peak_price"] == 10.1
    assert payload["cost_90_low"] == 9.4
    assert payload["sample_days"] == 120
    assert payload["calculation_method"] == "ohlcv_volume_decay_v1"
    assert payload["is_estimated"] is True


def test_agent_initial_message_contains_local_peak_metadata() -> None:
    message = AgentExecutor.__new__(AgentExecutor)._build_user_message(
        "分析股票",
        {
            "stock_code": "600519",
            "report_type": "detailed",
            "chip_distribution": _local_chip().to_dict(),
        },
    )

    assert '"peak_price": 10.1' in message
    assert '"sample_days": 120' in message
    assert '"source": "local_ohlcv_estimate"' in message
    assert '"is_estimated": true' in message


def test_report_schema_retains_extended_chip_fields() -> None:
    payload = ChipStructure.model_validate(
        {
            "peak_price": 10.1,
            "cost_70_low": 9.8,
            "cost_70_high": 10.6,
            "cost_90_low": 9.4,
            "cost_90_high": 11.0,
            "sample_days": 120,
            "calculation_method": "ohlcv_volume_decay_v1",
            "source": "local_ohlcv_estimate",
            "is_estimated": True,
        }
    ).model_dump()

    assert payload["peak_price"] == 10.1
    assert payload["sample_days"] == 120
    assert payload["source"] == "local_ohlcv_estimate"
    assert payload["is_estimated"] is True
