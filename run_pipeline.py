"""Run the full Week-1 pipeline: clean+split -> baseline -> leakage check."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
for script in ["data.py", "baseline.py", "leakage_check.py"]:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(SRC / script)], check=True, cwd=SRC)
