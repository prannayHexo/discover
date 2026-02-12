import tempfile
import uuid
from pathlib import Path

import pandas as pd
from mlebench.grade import grade_csv


def verify_submission(result, competition) -> bool:
    """Structurally validate a submission against competition expectations."""
    if result is None:
        return False
    if not isinstance(result, (pd.DataFrame, dict)):
        return False

    df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    if len(df) == 0:
        return False

    # Load expected format from sample submission
    sample = pd.read_csv(competition.sample_submission)

    # Check required columns exist
    missing = set(sample.columns) - set(df.columns)
    if missing:
        return False

    # Check row count matches expected
    answers = pd.read_csv(competition.answers)
    if len(df) != len(answers):
        return False

    # Check for all-NaN in any required column
    for col in sample.columns:
        if df[col].isna().all():
            return False

    return True


def grade_submission(result, competition, tmp_dir=None):
    """Grade a submission against a competition's ground truth.

    Args:
        result: pandas DataFrame or dict-like with submission data.
        competition: mlebench Competition object.
        tmp_dir: directory for temp CSV files. Uses system temp if None.

    Returns:
        (score, report) where score is a float (higher=better, 0.0 on failure)
        and report is the raw mlebench grading report.
    """
    df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)

    if tmp_dir is None:
        tmp_dir = Path(tempfile.gettempdir())
    else:
        tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tmp_dir / f"submission_{uuid.uuid4().hex}.csv"
    df.to_csv(csv_path, index=False)

    report = grade_csv(csv_path, competition)

    csv_path.unlink(missing_ok=True)

    if not report.valid_submission or report.score is None:
        return 0.0, report

    raw_score = report.score
    if report.is_lower_better:
        raw_score = -raw_score

    return raw_score, report
