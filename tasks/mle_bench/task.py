from pathlib import Path
import tempfile

from tasks.base_reward_task import BaseRewardTask
from tasks.mle_bench.verifier import verify_submission, grade_submission
from mlebench.registry import registry


class MleBenchTask(BaseRewardTask):

    def __init__(self, config, log_path="", competition_id="spaceship-titanic"):
        super().__init__(config, log_path)
        self.competition_id = competition_id
        self.competition = registry.get_competition(competition_id)
        self._last_report = None

    def get_function_name(self) -> str:
        return "run"

    def preprocess_generation(self, generation, *, step, state=None, **kwargs) -> str:
        public_dir = str(self.competition.public_dir)
        header = f'DATA_DIR = "{public_dir}"\n\n'
        return header + generation

    def get_reward(self, result) -> float:
        tmp_dir = Path(self.log_dir) / "tmp" if self.log_dir else Path(tempfile.gettempdir())
        score, report = grade_submission(result, self.competition, tmp_dir=tmp_dir)
        self._last_report = report
        return score

    def verify(self, result, *, step, **kwargs) -> bool:
        return verify_submission(result)


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
