# Run continuous-stream experiments across all batch sizes (N=50, 100, 300).
# Usage: python run_all_simu.py
# Set DIDO_DATA_ROOT env var to point to data folder (default: ../data/continuous).

import subprocess
import sys
import os
from pathlib import Path

python_exec = sys.executable

N_POINTS_LIST = [50, 100, 300]   # batch sizes from the paper

script_path = Path(__file__).parent / "main_run_flow.py"

for n_points in N_POINTS_LIST:
    print(f"\n=== Running N={n_points} ===")
    result = subprocess.run([python_exec, str(script_path), str(n_points)], check=False)
    if result.returncode != 0:
        print(f"[WARN] N={n_points} exited with error code {result.returncode}")

