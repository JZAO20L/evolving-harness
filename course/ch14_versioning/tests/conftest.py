import sys
from pathlib import Path

# 让测试能 `from agent import ...` / `from rsi import ...`：把章根挂到路径上。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
