import sys
from pathlib import Path

import pytest

# 让测试能 `from synth import ...`：把 synth_pipeline 根挂到路径上。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _sanitize_proxy_env(monkeypatch):
    # 冒烟测试的代理坑：裸 IPv6 CIDR 的 NO_PROXY 会让 httpx 建 client 时 InvalidURL。
    for key in ("ALL_PROXY", "all_proxy", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
