"""L1 冒烟：打真模型（deepseek-v4-flash-ga-260731 @ ARK /api/v3/chat/completions）。

离线默认跳过：需要 `EH_SMOKE=1` 且拿到 ARK key（.env 的 ARK_API_KEY / EH_API_KEY）。
这是【真实网络 + 真实扣费】测试，不做严格断言，只确认"能连通、模型回返非空文本 + 用量"。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent import Message, OpenAIChatModel

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def _load_env() -> None:
    """从仓库根读 .env（key 未导出到 shell，需要手动读）。"""
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        # .env 里 key 常带引号（python-dotenv 会剥，这边手动剥）
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _ark_key() -> str:
    return os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY") or ""


@pytest.fixture(autouse=True)
def _sanitize_proxy_env(monkeypatch):
    # shell 的 no_proxy 带裸 IPv6 CIDR（fe80::/10 等），个别 httpx 版本建 client 会 InvalidURL。
    # 清掉代理变量、只留安全 no_proxy；保留 HTTP_PROXY（内网代理用来中继到公网 ARK）。
    for key in ("ALL_PROXY", "all_proxy", "NO_PROXY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")


@pytest.mark.smoke
async def test_smoke_real_deepseek():
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    key = _ark_key()
    if not key:
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")

    model_name = os.getenv("EH_MODEL", DEEPSEEK)
    base_url = os.getenv("EH_BASE_URL", ARK_BASE)

    model = OpenAIChatModel(
        base_url=base_url,
        model=model_name,
        api_key=key,
        timeout=60.0,
        max_retries=2,
    )
    try:
        result = await model.complete([Message(role="user", content="用一句话介绍什么是强化学习")])
    finally:
        await model.aclose()

    assert result.content and result.content.strip()
    assert result.usage.total_tokens > 0
    print(f"\n[smoke] model={model_name} finish={result.finish_reason} "
          f"usage={result.usage.total_tokens}t -> {result.content!r}")