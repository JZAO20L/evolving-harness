from __future__ import annotations

from pathlib import Path

from .base import Tool, ToolIsolation

MAX_OUTPUT_BYTES = 64 * 1024
READ_ISOLATION = ToolIsolation(
    level="read",
    network="none",
    fs_scope="workspace",
    resources={"output_bytes": MAX_OUTPUT_BYTES},
)
WRITE_ISOLATION = ToolIsolation(
    level="read",
    network="none",
    fs_scope="workspace_writable",
    resources={"output_bytes": MAX_OUTPUT_BYTES},
)


def fs_tools(root: str | Path, protected: tuple[str, ...] = ()) -> list[Tool]:
    """Filesystem tools confined to `root`; every path resolves inside it.

    ``protected`` paths are readable but not writable through these tools (used
    for harness-owned state such as the memory files).
    """
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    protected_paths = [(root_path / item).resolve() for item in protected]

    def _resolve(relative: str) -> Path:
        target = (root_path / relative).resolve()
        if not target.is_relative_to(root_path):
            raise ValueError(f"path escapes workspace: {relative}")
        return target

    def _resolve_writable(relative: str) -> Path:
        target = _resolve(relative)
        if any(target == item or item in target.parents for item in protected_paths):
            raise ValueError(f"path is reserved for harness state: {relative}")
        return target

    def _cap(text: str) -> str:
        data = text.encode("utf-8")
        if len(data) <= MAX_OUTPUT_BYTES:
            return text
        return data[:MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore") + "\n...[truncated]"

    def read_file(path: str, offset: int = 0, limit: int = 400) -> str:
        target = _resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"not a file: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 2000))
        window = lines[offset : offset + limit]
        body = "\n".join(f"{index + offset + 1}\t{line}" for index, line in enumerate(window))
        header = f"[{path}] lines {offset + 1}-{offset + len(window)} of {len(lines)}\n"
        return _cap(header + body)

    def write_file(path: str, content: str) -> str:
        target = _resolve_writable(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content.encode('utf-8'))} bytes to {path}"

    def list_dir(path: str = ".") -> str:
        target = _resolve(path)
        if not target.exists():
            raise FileNotFoundError(f"not found: {path}")
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        rows = []
        for child in sorted(target.iterdir()):
            kind = "dir" if child.is_dir() else "file"
            size = child.stat().st_size if child.is_file() else 0
            rows.append(f"{kind}\t{size}\t{child.name}")
        return _cap("\n".join(rows) if rows else "(empty)")

    return [
        Tool(
            name="read_file",
            description="Read a text file inside the workspace. Returns numbered lines.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace root."},
                    "offset": {"type": "integer", "description": "0-based line offset."},
                    "limit": {"type": "integer", "description": "Max lines to return."},
                },
                "required": ["path"],
            },
            handler=read_file,
            isolation=READ_ISOLATION,
        ),
        Tool(
            name="write_file",
            description="Write a text file inside the workspace, creating parent directories.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace root."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
            },
            handler=write_file,
            isolation=WRITE_ISOLATION,
        ),
        Tool(
            name="list_dir",
            description="List directory entries inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory relative to workspace root."},
                },
                "required": [],
            },
            handler=list_dir,
            isolation=READ_ISOLATION,
        ),
    ]
