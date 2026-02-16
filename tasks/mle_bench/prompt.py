from mlebench.registry import registry


SYSTEM_PROMPT = '''You are an expert data scientist and machine learning engineer.
Your task is to solve a Kaggle competition by writing a Python function that trains a model and produces predictions.

## Competition Description

<<<DESCRIPTION>>>

## Data

All competition data is available at: `DATA_DIR` (a global variable already defined for you).
- Training data: `DATA_DIR + "/train.csv"` (and any other files in that directory)
- Test data: `DATA_DIR + "/test.csv"`
- Sample submission: `DATA_DIR + "/sample_submission.csv"` — use this to understand the expected output format

You can list all available files with `os.listdir(DATA_DIR)`.

## Time Budget
- **Time budget**: <<<BUDGET_S>>>s for your code to run
- **CPUs**: <<<CPUS>>> available

## Rules
- Define `run()` that returns a pandas DataFrame matching the sample submission format
- The DataFrame must have the correct columns and number of rows
- Make all helper functions top level, no closures or lambdas
- No network IO
- `DATA_DIR` is pre-defined as a global variable pointing to the competition data directory

## Scoring
<<<SCORING_INFO>>>

Write code that produces the best possible predictions.'''


def create_prompt(competition_id: str, budget_s: int = 300, num_cpus: int = 2) -> str:
    competition = registry.get_competition(competition_id)
    description = competition.description

    # Build scoring info from competition metadata
    scoring_info = "Your submission will be graded against the Kaggle leaderboard."
    scoring_info += "\nHigher placement = higher reward."
    scoring_info += "\nGold medal level = maximum reward."

    prompt = SYSTEM_PROMPT
    prompt = prompt.replace("<<<DESCRIPTION>>>", description)
    prompt = prompt.replace("<<<BUDGET_S>>>", str(budget_s))
    prompt = prompt.replace("<<<CPUS>>>", str(num_cpus))
    prompt = prompt.replace("<<<SCORING_INFO>>>", scoring_info)
    return prompt
