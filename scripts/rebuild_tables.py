#!/usr/bin/env python
"""Rebuild the statistics, tables and figures of the paper from the stored per-query results. Needs no data or GPU.
Usage: python scripts/rebuild_tables.py    Output goes to generated/."""
import os, subprocess, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); exp = os.path.join(root, "experiments"); env = {**os.environ, "JIT_OUT": os.path.join(root, "generated")}
for s in ["e20_stats.py", "e21_cost.py", "e23_pareto.py", "e25_shared_check.py", "e21_tables.py"]:
    print("==", s, flush=True); subprocess.run([sys.executable, s], cwd=exp, env=env, check=True)
