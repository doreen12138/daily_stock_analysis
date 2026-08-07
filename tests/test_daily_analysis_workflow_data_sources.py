from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT_DIR / ".github/workflows/00-daily-analysis.yml"


def test_daily_analysis_workflow_prefers_stable_free_daily_sources():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "DAILY_SOURCE_PRIORITY:" in workflow
    assert "tencent,baostock,akshare,pytdx,yfinance,efinance,tushare,tickflow" in workflow
    assert workflow.index("tencent,baostock") < workflow.index("efinance,tushare")
