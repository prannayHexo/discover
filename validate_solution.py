"""
Validate a solution from a search log by re-running it through verify_mle_bench.

Usage:
    python3 validate_solution.py /tmp/discover-search/run_20260211_141053/r02_s01.json
"""

import json
import os
import sys

os.environ.setdefault("RAY_DEDUP_LOGS", "1")
os.environ.setdefault("RAY_ENABLE_METRICS", "0")
os.environ.setdefault("RAY_METRICS_EXPORT_PORT", "0")

import ray

from tinker_cookbook.recipes.ttt.env_mle_bench import verify_mle_bench


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/discover-search/run_20260211_141053/r02_s01.json"

    with open(log_path) as f:
        log = json.load(f)

    parsed_code = log["parsed_code"]
    orig = log["verify_output"]

    print(f"Log: {log_path}")
    print(f"Original: score={orig['raw_score']} medal={orig['medal']} correctness={orig['correctness']}")
    print()

    ray.init(ignore_reinit_error=True, configure_logging=False)

    result = verify_mle_bench(
        generation=parsed_code,
        step=999,
        num_cpus_per_task=2,
        eval_timeout=300,
        log_path="/tmp/discover-search-validate",
        competition_id="spaceship-titanic",
    )

    print("=== Verification Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print()
    if result["correctness"] > 0:
        score_ok = orig["raw_score"] == result["raw_score"]
        medal_ok = orig["medal"] == result["medal"]
        print(f"Score: {orig['raw_score']} vs {result['raw_score']} -> {'MATCH' if score_ok else 'MISMATCH'}")
        print(f"Medal: {orig['medal']} vs {result['medal']} -> {'MATCH' if medal_ok else 'MISMATCH'}")
    else:
        print(f"FAILED: {result['msg']}")


if __name__ == "__main__":
    main()
