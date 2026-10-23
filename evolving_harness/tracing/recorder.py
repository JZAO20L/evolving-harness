from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

TRACE_VERSION = 1


class TraceRecorder:
    """Append-only JSONL trace. One event per line, flushed immediately.

    Event shape: {v, run_id, seq, ts, elapsed_ms, kind, data}
    """

    def __init__(
        self, run_id: str, trace_dir: str | Path = "traces", *, append: bool = False
    ) -> None:
        self.run_id = run_id
        self.path = Path(trace_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        if append and self.path.is_file():
            # A resumed run continues the same trace; keep the original events.
            with self.path.open(encoding="utf-8") as existing:
                self._seq = sum(1 for line in existing if line.strip())
        self._file = self.path.open("a" if append else "w", encoding="utf-8")
        self._started = time.time()

    def event(self, kind: str, **data: Any) -> None:
        if self._file.closed:
            return
        self._seq += 1
        record = {
            "v": TRACE_VERSION,
            "run_id": self.run_id,
            "seq": self._seq,
            "ts": round(time.time(), 6),
            "elapsed_ms": int((time.time() - self._started) * 1000),
            "kind": kind,
            "data": data,
        }
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
