import tempfile
import os
from pathlib import Path

import pandas as pd

from tasks.base_reward_task import BaseRewardTask
from mlebench.registry import registry
from mlebench.grade import grade_csv


class MleBenchTask(BaseRewardTask):

    def __init__(self, config, log_path="", competition_id="spaceship-titanic"):
        super().__init__(config, log_path)
        self.competition_id = competition_id
        self.competition = registry.get_competition(competition_id)

    def get_function_name(self) -> str:
        return "run"

    def preprocess_generation(self, generation, *, step, state=None, **kwargs) -> str:
        # Inject data paths so the LLM code can find train/test data
        public_dir = str(self.competition.public_dir)
        header = f'DATA_DIR = "{public_dir}"\n\n'
        return header + generation

    def get_reward(self, result) -> float:
        # result is a pandas DataFrame (the submission)
        df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

        # Write to temp CSV for grade_csv
        tmp_dir = Path(self.log_dir) / "tmp" if self.log_dir else Path(tempfile.gettempdir())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = tmp_dir / f"submission_{os.getpid()}.csv"
        df.to_csv(csv_path, index=False)

        report = grade_csv(csv_path, self.competition)

        # Clean up
        csv_path.unlink(missing_ok=True)

        if not report.valid_submission or report.score is None:
            return 0.0

        # Normalize score: map medal thresholds to reward tiers
        # gold=1.0, silver=0.75, bronze=0.5, above_median=0.25, else proportional
        if report.gold_medal:
            return 1.0
        elif report.silver_medal:
            return 0.75
        elif report.bronze_medal:
            return 0.5
        elif report.above_median:
            return 0.25
        else:
            return 0.1  # valid submission but below median

    def verify(self, result, *, step, **kwargs) -> bool:
        if result is None:
            return False
        if isinstance(result, pd.DataFrame):
            return len(result) > 0
        if isinstance(result, dict):
            return len(result) > 0
        return False


if __name__ == "__main__":
    from types import SimpleNamespace
    import ray

    ray.init("auto")

    def dict_to_ns(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
        return d

    config = {
        "ttt_rm": {
            "num_cpus_per_task": 2,
            "rew_type": "linear",
            "fail_score": 0.0,
            "eval_timeout": 300,
            "worst_perf_log": -10000,
            "n_item": 200,
        }
    }
    config_ns = dict_to_ns(config)

    task = MleBenchTask(config_ns, competition_id="spaceship-titanic")

    generation = """```python
import pandas as pd

def run():
    train = pd.read_csv(DATA_DIR + "/train.csv")
    test = pd.read_csv(DATA_DIR + "/test.csv")
    # Dummy baseline: predict False for everyone
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Transported": [False] * len(test),
    })
    return submission
```"""

    @ray.remote
    def run_score(task, generation):
        return task.compute_score(generation, step=0)

    futures = [run_score.remote(task, generation)]
    scores = ray.get(futures)
    print([(score['performance'], score['msg']) for score in scores])
