"""course/ch14 · versions —— agent 版本库、组件 diff 与单基线合并。

``AgentConfig`` 的 ``canonical()`` 已经是内容寻址的完整快照（含 skills 指令、memory 策略、
结构字段）。本章把它落成一个**可回放的版本库**：

```
<storage_dir>/
  versions/<id>.json        # 内容寻址对象（id == AgentConfig.hash()）
  HEAD                      # 当前生效版本 id（空仓库时不存在）
  .lock                     # fcntl.flock，保护“校验 parent + 前进 HEAD”
```

两个关键取舍：

1. **内容寻址**：同一 canonical → 同一 id；重复 ``commit`` 幂等返回旧 id，不重写对象。
   代价是“同一配置从不同基线到达”会折叠 lineage（parent 以首次登记为准）。
2. **handler 不可序列化**：canonical 只存 ``base_tools`` 名字；``checkout/rollback`` 由调用方
   注入 live handler（``AgentConfig.from_canonical(..., base_tools=...)``）。
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from rsi import AgentConfig, TaskScore, _mean, _sha256

COMPONENTS = ("prompt", "skills", "memory", "structure")
STRUCTURE_KEYS = ("max_steps", "tool_names", "subagent_enabled", "max_subagent_depth")
IMMUTABLE_KEYS = ("name", "base_tools")


class VersionStoreError(RuntimeError):
    """版本对象缺失/损坏/不可解析（带路径，便于定位）。"""


class ConcurrentCommit(RuntimeError):
    """``parent`` 与当前 HEAD 不一致：有别人先提交了，本次提交作废。"""


class ImmutableChange(ValueError):
    """``name`` / ``base_tools`` 不在可变面内。"""

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name
        super().__init__(f"字段 {field_name!r} 不可变，不能进入版本 diff")


class BaseMismatch(ValueError):
    """ChangeSet 基于另一份基线，不能参与本次合并。"""

    def __init__(self, expected: str, got: str) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"ChangeSet 基线是 {got}，本次合并基线是 {expected}")


class MergeConflict(ValueError):
    """同一组件出现两个不同的绝对新值。"""

    def __init__(self, component: str, variants: list) -> None:
        self.component = component
        self.variants = list(variants)
        super().__init__(f"合并冲突：组件 {component!r} 有 {len(self.variants)} 个不同新值")


@dataclass(frozen=True)
class Version:
    """一条版本记录：内容寻址 id + 谱系 + 配置快照 + 评估分数 + 裁决。"""

    id: str
    parents: tuple[str, ...]
    message: str
    author: str
    created_at: str
    canonical: dict
    train_scores: tuple[TaskScore, ...]
    val_scores: tuple[TaskScore, ...]
    aggregate_score: float
    verdict: str
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "parents": list(self.parents),
            "message": self.message,
            "author": self.author,
            "created_at": self.created_at,
            "canonical": self.canonical,
            "train_scores": [asdict(score) for score in self.train_scores],
            "val_scores": [asdict(score) for score in self.val_scores],
            "aggregate_score": self.aggregate_score,
            "verdict": self.verdict,
            "meta": self.meta,
        }

    @classmethod
    def from_record(cls, record: dict) -> Version:
        train = tuple(TaskScore(**row) for row in record.get("train_scores") or ())
        val = tuple(TaskScore(**row) for row in record.get("val_scores") or ())
        return cls(
            id=str(record["id"]),
            parents=tuple(str(p) for p in record.get("parents") or ()),
            message=str(record.get("message", "")),
            author=str(record.get("author", "")),
            created_at=str(record.get("created_at", "")),
            canonical=record["canonical"],
            train_scores=train,
            val_scores=val,
            aggregate_score=float(record.get("aggregate_score", 0.0)),
            verdict=str(record.get("verdict", "")),
            meta=dict(record.get("meta") or {}),
        )


def _atomic_write(path: Path, text: str) -> None:
    """先写 ``*.tmp`` 再 ``os.replace``：同目录原子替换，崩溃时旧文件仍完整。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class AgentRepo:
    """文件版本库：对象 + HEAD + 锁；支持 log / ancestors / checkout / commit / rollback / diff。"""

    def __init__(self, storage_dir: str | Path) -> None:
        self.root = Path(storage_dir)
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.head_path = self.root / "HEAD"
        self.lock_path = self.root / ".lock"
        self._last_us = 0

    @contextmanager
    def _lock(self):
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _object_path(self, version_id: str) -> Path:
        if not version_id or "/" in version_id or "\\" in version_id or version_id.startswith("."):
            raise KeyError(f"非法版本 id: {version_id!r}")
        return self.versions_dir / f"{version_id}.json"

    def head(self) -> str | None:
        if not self.head_path.exists():
            return None
        value = self.head_path.read_text(encoding="utf-8").strip()
        return value or None

    def get(self, version_id: str) -> Version | None:
        path = self._object_path(version_id)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            return Version.from_record(record)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise VersionStoreError(f"版本对象损坏：{path}（{exc}）") from exc

    def log(self, *, limit: int | None = None) -> list[Version]:
        """全部版本，新→旧；包括回滚后不在 HEAD 祖先链上的对象。"""
        versions: list[Version] = []
        for path in sorted(self.versions_dir.glob("*.json")):
            try:
                versions.append(Version.from_record(json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise VersionStoreError(f"版本对象损坏：{path}（{exc}）") from exc
        versions.sort(key=lambda version: (version.created_at, version.id), reverse=True)
        return versions[:limit] if limit else versions

    def ancestors(self, version_id: str | None = None) -> list[Version]:
        """沿 ``parents`` 走当前谱系（新→旧）；不包含其它分支/被遗弃的版本。"""
        current = version_id or self.head()
        chain: list[Version] = []
        seen: set[str] = set()
        while current and current not in seen:
            record = self.get(current)
            if record is None:
                break
            chain.append(record)
            seen.add(current)
            current = record.parents[0] if record.parents else None
        return chain

    def commit(self, config: AgentConfig, *, parent: str | None, message: str,
               author: str = "rsi", train_scores=(), val_scores=(),
               verdict: str = "commit", meta: dict | None = None) -> str:
        """两阶段原子提交：先落对象，再校验 parent 并推进 HEAD。

        同一 canonical 已存在 → 幂等返回旧 id（HEAD 不动）。
        """
        canonical = config.canonical()
        version_id = config.hash()
        path = self._object_path(version_id)
        train = tuple(train_scores)
        val = tuple(val_scores)
        with self._lock():
            if path.exists():
                return version_id
            head = self.head()
            if parent != head:
                raise ConcurrentCommit(f"parent={parent!r} 与 HEAD={head!r} 不一致")
            self._last_us = max(self._last_us + 1, time.time_ns() // 1000)
            created_at = datetime.fromtimestamp(
                self._last_us / 1_000_000, UTC).isoformat(timespec="microseconds")
            record = Version(
                id=version_id,
                parents=(parent,) if parent else (),
                message=message,
                author=author,
                created_at=created_at,
                canonical=canonical,
                train_scores=train,
                val_scores=val,
                aggregate_score=_mean([score.mean for score in train]),
                verdict=verdict,
                meta=dict(meta or {}),
            )
            _atomic_write(path, json.dumps(record.to_record(), ensure_ascii=False,
                                           sort_keys=True, indent=2))
            _atomic_write(self.head_path, version_id)
        return version_id

    def checkout(self, version_id: str | None = None, *, move_head: bool = False,
                 base_tools: tuple = ()) -> AgentConfig:
        """重建历史版本 config；``move_head=True`` 把它设为新基线（“从历史版本选择”）。"""
        version_id = version_id or self.head()
        if version_id is None:
            raise ValueError("空仓库没有可 checkout 的版本")
        record = self.get(version_id)
        if record is None:
            raise KeyError(f"没有版本 {version_id}")
        config = AgentConfig.from_canonical(record.canonical, base_tools=tuple(base_tools))
        if move_head:
            with self._lock():
                _atomic_write(self.head_path, version_id)
        return config

    def rollback(self, version_id: str | None = None, *, base_tools: tuple = ()) -> AgentConfig:
        """把 HEAD 移到指定版本；默认移到 HEAD 的 parent。对象不删除。"""
        current = self.head()
        if current is None:
            raise ValueError("空仓库没有可回滚的版本")
        if version_id is None:
            record = self.get(current)
            if record is None:
                raise KeyError(f"没有版本 {current}")
            if not record.parents:
                raise ValueError("HEAD 是根版本，没有父版本可回滚")
            version_id = record.parents[0]
        return self.checkout(version_id, move_head=True, base_tools=base_tools)

    def diff(self, a: str, b: str | None = None) -> ConfigDiff:
        base = self.get(a)
        if base is None:
            raise KeyError(f"没有版本 {a}")
        target_id = b or self.head()
        if target_id is None:
            raise ValueError("空仓库没有 HEAD，无法 diff")
        target = self.get(target_id)
        if target is None:
            raise KeyError(f"没有版本 {target_id}")
        _guard_immutable(base.canonical, target.canonical)
        return ConfigDiff(base.id, base.canonical, target.canonical)


def _payload(canonical: dict, component: str):
    if component == "prompt":
        return copy.deepcopy(canonical.get("prompt") or {})
    if component == "skills":
        return copy.deepcopy(canonical.get("skills") or [])
    if component == "memory":
        return copy.deepcopy(canonical.get("memory") or {})
    if component == "structure":
        return {key: copy.deepcopy(canonical.get(key)) for key in STRUCTURE_KEYS}
    raise ValueError(f"未知组件: {component!r}")


def _guard_immutable(base_canonical: dict, cand_canonical: dict) -> None:
    for key in IMMUTABLE_KEYS:
        if base_canonical.get(key) != cand_canonical.get(key):
            raise ImmutableChange(key)


@dataclass(frozen=True)
class ConfigDiff:
    """两份 config 的组件级差异；``before/after`` 都是绝对值（不是 patch）。"""

    base_version: str
    base_canonical: dict
    cand_canonical: dict

    @property
    def changed(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for component in COMPONENTS:
            before = _payload(self.base_canonical, component)
            after = _payload(self.cand_canonical, component)
            if before != after:
                out[component] = {"before": before, "after": after}
        return out

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(component for component in COMPONENTS if component in self.changed)

    def hash(self) -> str:
        body = {"base": self.base_version, "changed": self.changed}
        return _sha256(json.dumps(body, sort_keys=True, ensure_ascii=False))


def config_diff(base: AgentConfig, cand: AgentConfig) -> ConfigDiff:
    base_canonical, cand_canonical = base.canonical(), cand.canonical()
    _guard_immutable(base_canonical, cand_canonical)
    return ConfigDiff(base.hash(), base_canonical, cand_canonical)


@dataclass(frozen=True)
class ChangeSet:
    """一组“组件 → 绝对新值”的改动，绑定在某个基线上；只含真正变化的组件。"""

    base_version: str
    base_canonical: dict
    changes: dict = field(default_factory=dict)

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(component for component in COMPONENTS if component in self.changes)

    def apply(self) -> dict:
        merged = copy.deepcopy(self.base_canonical)
        for component, value in self.changes.items():
            if component == "structure":
                merged.update(copy.deepcopy(value))
            else:
                merged[component] = copy.deepcopy(value)
        return merged

    def to_config(self, base_tools: tuple = ()) -> AgentConfig:
        return AgentConfig.from_canonical(self.apply(), base_tools=tuple(base_tools))

    def hash(self) -> str:
        return _sha256(json.dumps(self.changes, sort_keys=True, ensure_ascii=False))

    @classmethod
    def from_config(cls, base: AgentConfig, cand: AgentConfig) -> ChangeSet:
        base_canonical, cand_canonical = base.canonical(), cand.canonical()
        _guard_immutable(base_canonical, cand_canonical)
        changes = {}
        for component in COMPONENTS:
            before = _payload(base_canonical, component)
            after = _payload(cand_canonical, component)
            if before != after:
                changes[component] = after
        return cls(base.hash(), base_canonical, changes)

    @classmethod
    def from_changes(cls, base: AgentConfig, changes: dict) -> ChangeSet:
        """从“组件 → 新值”构造；structure 允许只写变化的键，其余组件要求完整 payload。"""
        base_canonical = base.canonical()
        clean: dict = {}
        for component, value in changes.items():
            if component in IMMUTABLE_KEYS:
                raise ImmutableChange(component)
            if component not in COMPONENTS:
                raise ValueError(f"未知组件: {component!r}")
            candidate = copy.deepcopy(value)
            if component == "structure":
                unknown = set(candidate) - set(STRUCTURE_KEYS)
                if unknown:
                    raise ValueError(f"structure 含未知字段: {sorted(unknown)}")
                candidate = {**_payload(base_canonical, "structure"), **candidate}
            if candidate != _payload(base_canonical, component):
                clean[component] = candidate
        return cls(base.hash(), base_canonical, clean)


def merge_changes(base: AgentConfig, changesets: list[ChangeSet]) -> ChangeSet:
    """单基线合并：不同组件直接组合；同组件同值幂等；同组件异值 → ``MergeConflict``。"""
    base_version = base.hash()
    merged: dict = {}
    for changeset in changesets:
        if changeset.base_version != base_version:
            raise BaseMismatch(base_version, changeset.base_version)
        for component, value in changeset.changes.items():
            if component not in merged:
                merged[component] = copy.deepcopy(value)
            elif merged[component] != value:
                raise MergeConflict(component, [merged[component], copy.deepcopy(value)])
    return ChangeSet(base_version, base.canonical(), merged)


__all__ = [
    "COMPONENTS",
    "IMMUTABLE_KEYS",
    "STRUCTURE_KEYS",
    "AgentRepo",
    "BaseMismatch",
    "ChangeSet",
    "ConcurrentCommit",
    "ConfigDiff",
    "ImmutableChange",
    "MergeConflict",
    "Version",
    "VersionStoreError",
    "config_diff",
    "merge_changes",
]
