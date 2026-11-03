"""L0 离线：ch14 版本库 / 组件 diff / 单基线合并 / VersionedRsi（全 Mock，不联网）。"""

from pathlib import Path

import pytest
from agent import ChatModel, Message, ModelResult, Tool
from rsi import (
    AgentConfig,
    MemoryPolicy,
    PromptPack,
    Task,
    TaskScore,
    VersionedRsi,
    update_memory_policy,
    update_structure,
)
from versions import (
    AgentRepo,
    BaseMismatch,
    ChangeSet,
    ConcurrentCommit,
    ImmutableChange,
    MergeConflict,
    VersionStoreError,
    config_diff,
    merge_changes,
)


def _tool(name: str = "echo") -> Tool:
    return Tool(name=name, description=name,
                parameters={"type": "object", "properties": {}},
                handler=lambda _args: f"ok:{name}")


def _cfg(**changes) -> AgentConfig:
    base = {"name": "base", "prompt": PromptPack(guidance="先算再答。"), "max_steps": 3,
            "memory": MemoryPolicy(recall_enabled=False, store_full_turns=False,
                                   profile_enabled=False)}
    base.update(changes)
    return AgentConfig(**base)


def _task(task_id: str = "t1", statement: str = "请计算 40+2") -> Task:
    return Task(id=task_id, statement=statement, checks=("42",))


def _scores(value: float, *ids: str) -> list[TaskScore]:
    return [TaskScore(task_id=task_id, mean=value, best=value, min_=value) for task_id in ids]


def _seed(repo: AgentRepo, config: AgentConfig, *, parent: str | None,
          train: list[TaskScore], val: list[TaskScore], verdict: str = "commit") -> str:
    return repo.commit(config, parent=parent, message="seed", train_scores=train,
                       val_scores=val, verdict=verdict)


def _repo(tmp_path: Path) -> AgentRepo:
    return AgentRepo(tmp_path / "repo")


class PromptAwareModel(ChatModel):
    """system 里带 MAGIC 才答对；用于让变更真实改变 train/val 分数。"""

    async def complete(self, messages: list[Message], tools=None) -> ModelResult:
        system = "\n".join(m.content for m in messages if m.role == "system")
        return ModelResult(content="42" if "MAGIC" in system else "7")


def _magic_proposer(candidate: AgentConfig):
    async def propose(current: AgentConfig, _scores: list[TaskScore]):
        return ChangeSet.from_config(current, candidate)

    return propose


# --------------------------------------------------------------------------- #
# 1. 版本库：提交 / checkout / log / ancestors / rollback / 原子性 / 并发
# --------------------------------------------------------------------------- #
def test_commit_checkout_round_trip_reinjects_handlers(tmp_path):
    tool = _tool("echo")
    config = _cfg(base_tools=(tool,))
    repo = _repo(tmp_path)
    version = _seed(repo, config, parent=None, train=_scores(1.0, "t1"),
                    val=_scores(1.0, "v1"))

    restored = repo.checkout(version, base_tools=(tool,))
    assert restored.hash() == version
    assert restored.enabled_tools()[0].handler({}) == "ok:echo"
    assert repo.head() == version


def test_duplicate_commit_is_idempotent_and_head_unchanged(tmp_path):
    repo = _repo(tmp_path)
    config = _cfg()
    version = _seed(repo, config, parent=None, train=_scores(0.0, "t1"),
                    val=_scores(0.0, "v1"))

    again = _seed(repo, config, parent=version, train=_scores(1.0, "t1"),
                  val=_scores(1.0, "v1"))
    assert again == version
    assert repo.head() == version
    assert len(list((tmp_path / "repo" / "versions").glob("*.json"))) == 1


def test_empty_repo_head_and_first_commit(tmp_path):
    repo = _repo(tmp_path)
    assert repo.head() is None
    version = _seed(repo, _cfg(), parent=None, train=_scores(0.0, "t1"),
                    val=_scores(0.0, "v1"))
    assert repo.head() == version


def test_log_ancestors_and_rollback_keep_old_versions(tmp_path):
    repo = _repo(tmp_path)
    v1 = _seed(repo, _cfg(), parent=None, train=_scores(0.0, "t1"), val=_scores(0.0, "v1"))
    v2 = _seed(repo, _cfg(prompt=PromptPack(guidance="A")), parent=v1,
               train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))
    v3 = _seed(repo, update_structure(_cfg(prompt=PromptPack(guidance="A")), max_steps=5),
               parent=v2, train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))

    assert [version.id for version in repo.log()] == [v3, v2, v1]
    assert [version.id for version in repo.ancestors()] == [v3, v2, v1]

    restored = repo.rollback()
    assert repo.head() == v2
    assert restored.hash() == v2
    assert v3 in {version.id for version in repo.log()}
    assert repo.checkout(v3).hash() == v3


def test_commit_is_atomic_and_head_contains_exact_id(tmp_path):
    repo = _repo(tmp_path)
    version = _seed(repo, _cfg(), parent=None, train=_scores(0.0, "t1"),
                    val=_scores(0.0, "v1"))
    assert repo.head_path.read_text(encoding="utf-8") == version
    assert not list((tmp_path / "repo").rglob("*.tmp"))


def test_rollback_to_explicit_version(tmp_path):
    repo = _repo(tmp_path)
    v1 = _seed(repo, _cfg(), parent=None, train=_scores(0.0, "t1"), val=_scores(0.0, "v1"))
    v2 = _seed(repo, _cfg(prompt=PromptPack(guidance="B")), parent=v1,
               train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))

    restored = repo.rollback(v2)
    assert repo.head() == v2 and restored.hash() == v2
    root = repo.rollback(v1)
    assert repo.head() == v1 and root.hash() == v1
    with pytest.raises(ValueError):
        repo.rollback()


def test_checkout_unknown_and_empty_repo_errors(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ValueError):
        repo.checkout()
    with pytest.raises(KeyError):
        repo.checkout("0" * 64)
    assert repo.get("0" * 64) is None


def test_stale_parent_raises_concurrent_commit(tmp_path):
    repo = _repo(tmp_path)
    v1 = _seed(repo, _cfg(), parent=None, train=_scores(0.0, "t1"), val=_scores(0.0, "v1"))
    v2 = _seed(repo, _cfg(prompt=PromptPack(guidance="A")), parent=v1,
               train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))

    repo.head_path.write_text(v1, encoding="utf-8")      # 另一个写者把 HEAD 挪回去了
    with pytest.raises(ConcurrentCommit):
        _seed(repo, _cfg(prompt=PromptPack(guidance="C")), parent=v2,
              train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))
    assert repo.head() == v1


def test_corrupted_object_raises_with_path(tmp_path):
    repo = _repo(tmp_path)
    version = _seed(repo, _cfg(), parent=None, train=_scores(0.0, "t1"),
                    val=_scores(0.0, "v1"))
    path = tmp_path / "repo" / "versions" / f"{version}.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(VersionStoreError) as exc:
        repo.get(version)
    assert str(path) in str(exc.value)
    with pytest.raises(VersionStoreError):
        repo.log()


def test_checkout_historical_version_moves_head_and_parent(tmp_path):
    repo = _repo(tmp_path)
    v1 = _seed(repo, _cfg(), parent=None, train=_scores(0.0, "t1"), val=_scores(0.0, "v1"))
    v2 = _seed(repo, _cfg(prompt=PromptPack(guidance="A")), parent=v1,
               train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))

    repo.checkout(v1, move_head=True)
    assert repo.head() == v1
    v3 = _seed(repo, _cfg(prompt=PromptPack(guidance="C")), parent=v1,
               train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))
    assert repo.get(v3).parents == (v1,)
    assert repo.get(v2).parents == (v1,)      # 旧分支仍在 log 里可查
    assert {version.id for version in repo.log()} == {v1, v2, v3}


def test_repo_diff_between_versions(tmp_path):
    repo = _repo(tmp_path)
    v1 = _seed(repo, _cfg(), parent=None, train=_scores(0.0, "t1"), val=_scores(0.0, "v1"))
    v2 = _seed(repo, update_structure(_cfg(), max_steps=5), parent=v1,
               train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))
    diff = repo.diff(v1, v2)
    assert diff.base_version == v1
    assert diff.components == ("structure",)
    assert diff.changed["structure"]["after"]["max_steps"] == 5


# --------------------------------------------------------------------------- #
# 2. diff / ChangeSet / 合并
# --------------------------------------------------------------------------- #
def test_config_diff_reports_changed_components_with_absolute_values():
    base = _cfg()
    cand = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"), max_steps=5)
    diff = config_diff(base, cand)
    assert set(diff.components) == {"prompt", "structure"}
    assert diff.changed["structure"]["before"]["max_steps"] == 3
    assert diff.changed["structure"]["after"]["max_steps"] == 5
    assert diff.changed["prompt"]["after"]["guidance"] == "MAGIC 先算再答。"


def test_changeset_round_trip_restores_candidate_hash():
    base = _cfg()
    cand = update_structure(base, max_steps=5)
    changeset = ChangeSet.from_config(base, cand)
    assert changeset.to_config(base_tools=base.base_tools).hash() == cand.hash()
    assert changeset.components == ("structure",)


def test_merge_disjoint_components():
    base = _cfg()
    prompt_cand = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    structure_cand = update_structure(base, max_steps=5)
    merged = merge_changes(base, [ChangeSet.from_config(base, prompt_cand),
                                  ChangeSet.from_config(base, structure_cand)])
    assert merged.components == ("prompt", "structure")
    expected = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"), max_steps=5)
    assert merged.to_config(base_tools=base.base_tools).hash() == expected.hash()


def test_merge_same_component_same_value_is_idempotent():
    base = _cfg()
    cand = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    changeset = ChangeSet.from_config(base, cand)
    merged = merge_changes(base, [changeset, changeset])
    assert merged.components == ("prompt",)
    assert merged.hash() == changeset.hash()


def test_merge_same_component_different_values_conflicts():
    base = _cfg()
    a = ChangeSet.from_changes(base, {"prompt": {"guidance": "A", "task_frame": {}, "final_format": ""}})
    b = ChangeSet.from_changes(base, {"prompt": {"guidance": "B", "task_frame": {}, "final_format": ""}})
    with pytest.raises(MergeConflict) as exc:
        merge_changes(base, [a, b])
    assert exc.value.component == "prompt"
    assert len(exc.value.variants) == 2


def test_base_mismatch_rejects_other_baseline():
    base = _cfg()
    other = _cfg(prompt=PromptPack(guidance="other"))
    changeset = ChangeSet.from_config(other, update_structure(other, max_steps=5))
    with pytest.raises(BaseMismatch):
        merge_changes(base, [changeset])


def test_changeset_hash_is_insertion_order_independent():
    base = _cfg()
    prompt = {"guidance": "MAGIC", "task_frame": {}, "final_format": ""}
    structure = {"max_steps": 5}
    first = ChangeSet.from_changes(base, {"prompt": prompt, "structure": structure})
    second = ChangeSet.from_changes(base, {"structure": structure, "prompt": prompt})
    assert first.hash() == second.hash()


def test_immutable_fields_rejected():
    base = _cfg(base_tools=(_tool("echo"),))
    with pytest.raises(ImmutableChange):
        ChangeSet.from_config(base, _cfg(name="renamed", base_tools=(_tool("echo"),)))
    with pytest.raises(ImmutableChange):
        ChangeSet.from_config(base, _cfg(base_tools=(_tool("other"),)))
    with pytest.raises(ImmutableChange):
        ChangeSet.from_changes(base, {"name": "renamed"})
    with pytest.raises(ImmutableChange):
        ChangeSet.from_changes(base, {"base_tools": []})


# --------------------------------------------------------------------------- #
# 3. VersionedRsi：合并 → 评估 → 门控 → 提交 / 回滚 / 冲突 / 历史基线
# --------------------------------------------------------------------------- #
TRAIN = [_task("t1", "请计算 40+2")]
VAL = [_task("v1", "请计算 20+22")]


def _versioned(config, repo, proposers, **kwargs) -> VersionedRsi:
    return VersionedRsi(config, PromptAwareModel(), repo=repo, train=TRAIN, val=VAL,
                        proposers=proposers, n_rollouts=1, max_rounds=2, **kwargs)


async def test_versioned_commits_improving_prompt(tmp_path):
    repo = _repo(tmp_path)
    base = _cfg()
    improved = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    report = await _versioned(base, repo, [_magic_proposer(improved)]).run()

    outcome = report.outcomes[0]
    assert outcome.verdict == "commit"
    assert outcome.version == report.head
    version = repo.get(report.head)
    assert version.verdict == "commit"
    assert version.parents and version.parents[0] != report.head
    assert "prompt" in version.meta["diff"]
    assert {record.id for record in repo.log()} == {version.parents[0], report.head}
    assert report.head_config.hash() == improved.hash()


async def test_versioned_merges_disjoint_proposals_in_one_commit(tmp_path):
    repo = _repo(tmp_path)
    base = _cfg()
    prompt_cand = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    structure_cand = update_structure(base, max_steps=5)
    report = await _versioned(base, repo, [_magic_proposer(prompt_cand),
                                           _magic_proposer(structure_cand)]).run()

    assert report.outcomes[0].verdict == "commit"
    version = repo.get(report.head)
    assert set(version.meta["diff"]) == {"prompt", "structure"}
    assert version.meta["sources"] == ["prompt", "structure"]
    assert report.head_config.hash() == _cfg(
        prompt=PromptPack(guidance="MAGIC 先算再答。"), max_steps=5).hash()


async def test_versioned_holds_when_scores_unchanged(tmp_path):
    repo = _repo(tmp_path)
    magic = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    neutral = update_memory_policy(magic, recall_top_k=9)
    report = await _versioned(magic, repo, [_magic_proposer(neutral)]).run()

    assert report.outcomes[0].verdict == "hold"
    assert report.outcomes[0].version is None
    assert repo.head() == report.head
    assert repo.get(report.head).verdict == "base"


async def test_versioned_rolls_back_when_scores_regress(tmp_path):
    repo = _repo(tmp_path)
    magic = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    worse = _cfg()
    report = await _versioned(magic, repo, [_magic_proposer(worse)]).run()

    assert report.outcomes[0].verdict == "rollback"
    assert report.outcomes[0].version is None
    assert repo.get(report.head).verdict == "base"
    assert len(repo.log()) == 1


async def test_versioned_conflict_does_not_evaluate_or_move_head(tmp_path):
    repo = _repo(tmp_path)
    base = _cfg()
    baseline = _seed(repo, base, parent=None, train=_scores(0.0, "t1"),
                     val=_scores(0.0, "v1"), verdict="base")

    def proposer(prompt_guidance: str):
        async def propose(current: AgentConfig, _scores: list[TaskScore]):
            payload = {"guidance": prompt_guidance, "task_frame": {}, "final_format": ""}
            return ChangeSet.from_changes(current, {"prompt": payload})

        return propose

    versioned = _versioned(base, repo, [proposer("A"), proposer("B")], base_version=baseline)
    report = await versioned.run()

    assert report.outcomes[0].verdict == "conflict"
    assert "prompt" in report.outcomes[0].error
    assert versioned.evals == 0
    assert repo.head() == baseline


async def test_versioned_reuses_cached_baseline_scores(tmp_path):
    repo = _repo(tmp_path)
    base = _cfg()
    baseline = _seed(repo, base, parent=None, train=_scores(0.0, "t1"),
                     val=_scores(0.0, "v1"), verdict="base")
    improved = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    report = await _versioned(base, repo, [_magic_proposer(improved)],
                              base_version=baseline).run()

    assert report.outcomes[0].verdict == "commit"
    assert report.outcomes[0].train_delta > 0 and report.outcomes[0].val_delta > 0
    assert report.outcomes[0].base_version == baseline


async def test_versioned_can_start_from_historical_version(tmp_path):
    repo = _repo(tmp_path)
    v1_config = _cfg()
    v1 = _seed(repo, v1_config, parent=None, train=_scores(0.0, "t1"),
               val=_scores(0.0, "v1"), verdict="base")
    v2 = _seed(repo, _cfg(prompt=PromptPack(guidance="OTHER")), parent=v1,
               train=_scores(1.0, "t1"), val=_scores(1.0, "v1"))
    assert repo.head() == v2

    improved = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    versioned = _versioned(v1_config, repo, [_magic_proposer(improved)], base_version=v1)
    report = await versioned.run()

    assert report.outcomes[0].verdict == "commit"
    assert repo.get(report.head).parents == (v1,)
    assert repo.get(v2).parents == (v1,)
    assert v1 in {version.id for version in repo.log()}


async def test_versioned_counts_evaluations_when_baseline_is_cached(tmp_path):
    repo = _repo(tmp_path)
    base = _cfg()
    baseline = _seed(repo, base, parent=None, train=_scores(0.0, "t1"),
                     val=_scores(0.0, "v1"), verdict="base")
    improved = _cfg(prompt=PromptPack(guidance="MAGIC 先算再答。"))
    versioned = _versioned(base, repo, [_magic_proposer(improved)], base_version=baseline)
    await versioned.run()

    assert versioned.evals == 2      # 候选 train + val；基线来自缓存
    assert repo.head() != baseline
