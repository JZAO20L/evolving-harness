"""L1 冒烟：真模型（deepseek-V4 @ ARK）跑通合成链路的小配额（2 math + 2 tooluse）。

断言只到「链路活着」：至少各生成 1 条通过机械校验的数据，产物能按 Task 解析。
离线默认跳过：需 EH_SMOKE=1 + ARK key。
"""

import json
import os
from pathlib import Path

import pytest
import synth
from synth import OpenAIChatModel, Task, synthesize

DEEPSEEK = "deepseek-v4-flash-ga-260731"
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"


def _load_env() -> None:
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def _need_key() -> str:
    _load_env()
    if os.getenv("EH_SMOKE") != "1":
        pytest.skip("EH_SMOKE != 1，离线跳过 L1 冒烟")
    key = os.getenv("ARK_API_KEY") or os.getenv("EH_API_KEY")
    if not key:
        pytest.skip("缺少 ARK_API_KEY / EH_API_KEY，跳过 L1 冒烟")
    return key


def _make_model(api_key: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        base_url=os.getenv("EH_BASE_URL", ARK_BASE),
        model=os.getenv("EH_MODEL", DEEPSEEK),
        api_key=api_key,
        timeout=120.0,
        max_retries=2,
    )


@pytest.mark.smoke
async def test_smoke_generates_valid_mixed_data(tmp_path):
    api_key = _need_key()
    model = _make_model(api_key)
    try:
        report = await synthesize(
            model,
            math_count=2,
            tooluse_count=2,
            seed=7,
            max_calls=20,
            out_dir=tmp_path,
            model_name=model.model,
            docs_wanted=4,
        )
    finally:
        await model.aclose()

    assert report.math, "真模型应至少产出 1 条通过校验的 math"
    assert report.tooluse, "真模型应至少产出 1 条通过校验的 tooluse"
    assert report.docs, "应有可用语料"
    assert report.calls <= 20

    for item in report.math:
        assert item["mode"] in {"mth-num", "mth-fmt"}
        assert synth._safe_eval(item["expr"]) == synth._normalize_answer(item["answer"])
    for item in report.tooluse:
        assert item["mode"] in {"tol-single", "tol-multi"}
        synth._validate_tooluse(
            {
                "statement": item["statement"],
                "answers": item["answers"],
                "evidence": item["evidence"],
                "forbid": item["forbid"],
            },
            report.docs,
        )

    for domain in ("math", "tooluse"):
        for name in ("train", "val", "test"):
            path = tmp_path / domain / f"{name}.jsonl"
            assert path.exists()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            for row in rows:
                assert set(row) == {"id", "statement", "checks"}
                Task(**row)
    corpus = json.loads((tmp_path / "tooluse" / "corpus.json").read_text(encoding="utf-8"))
    assert corpus["docs"]
    assert (tmp_path / "generation_log.jsonl").exists()

    print(f"\n[smoke] calls={report.calls} docs={len(report.docs)} "
          f"math={len(report.math)} tooluse={len(report.tooluse)}")
    for item in report.math[:2]:
        print(f"   math {item['mode']}: {item['statement']} -> {item['answer']}")
    for item in report.tooluse[:2]:
        print(f"   tooluse {item['mode']}: {item['statement']} -> {item['answers']}")
