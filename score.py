"""
Score a solution script against mle-bench.

Usage: python3 score.py solution.py [competition_id]

The solution must define a run() function that returns a DataFrame.
DATA_DIR is injected automatically.
"""

import sys, os, ray

os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_ENABLE_METRICS"] = "0"

ray.init(ignore_reinit_error=True, configure_logging=False)

from tinker_cookbook.recipes.ttt.env_mle_bench import verify_mle_bench

script = sys.argv[1]
comp = sys.argv[2] if len(sys.argv) > 2 else "spaceship-titanic"

with open(script) as f:
    code = f"```python\n{f.read()}\n```"

result = verify_mle_bench(code, step=0, competition_id=comp)
print(f"score={result['raw_score']} medal={result['medal']} correctness={result['correctness']}")
