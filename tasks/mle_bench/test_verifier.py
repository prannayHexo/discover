import pandas as pd
from tasks.mle_bench.verifier import verify_submission, grade_submission
from mlebench.registry import registry

COMP = registry.get_competition("spaceship-titanic")
SAMPLE = pd.read_csv(COMP.sample_submission)
ANSWERS = pd.read_csv(COMP.answers)


def test_verify_valid_dataframe():
    assert verify_submission(SAMPLE, COMP) is True
    print("  pass: valid DataFrame")


def test_verify_valid_dict():
    d = SAMPLE.to_dict(orient="list")
    assert verify_submission(d, COMP) is True
    print("  pass: valid dict")


def test_verify_empty_dataframe():
    df = pd.DataFrame()
    assert verify_submission(df, COMP) is False
    print("  pass: empty DataFrame rejected")


def test_verify_none():
    assert verify_submission(None, COMP) is False
    print("  pass: None rejected")


def test_verify_wrong_type():
    assert verify_submission("hello", COMP) is False
    assert verify_submission(42, COMP) is False
    print("  pass: wrong types rejected")


def test_verify_wrong_columns():
    df = pd.DataFrame({"wrong_col": range(len(ANSWERS))})
    assert verify_submission(df, COMP) is False
    print("  pass: wrong column names rejected")


def test_verify_missing_target_column():
    df = pd.DataFrame({"PassengerId": ANSWERS["PassengerId"]})
    assert verify_submission(df, COMP) is False
    print("  pass: missing target column rejected")


def test_verify_wrong_row_count():
    df = SAMPLE.head(5).copy()
    assert verify_submission(df, COMP) is False
    print("  pass: wrong row count rejected")


def test_verify_all_nan_column():
    df = SAMPLE.copy()
    df["Transported"] = None
    assert verify_submission(df, COMP) is False
    print("  pass: all-NaN target column rejected")


def test_grade_valid_submission():
    score, report = grade_submission(SAMPLE, COMP)
    assert report.valid_submission, f"submission invalid: {report}"
    assert score != 0.0 or report.score == 0.0
    print(f"  pass: graded sample submission, score={score:.4f}")


def test_grade_bad_submission():
    bad = pd.DataFrame({"wrong_col": [1, 2, 3]})
    score, report = grade_submission(bad, COMP)
    assert score == 0.0
    print(f"  pass: bad submission scored 0.0")


if __name__ == "__main__":
    tests = [
        test_verify_valid_dataframe,
        test_verify_valid_dict,
        test_verify_empty_dataframe,
        test_verify_none,
        test_verify_wrong_type,
        test_verify_wrong_columns,
        test_verify_missing_target_column,
        test_verify_wrong_row_count,
        test_verify_all_nan_column,
        test_grade_valid_submission,
        test_grade_bad_submission,
    ]

    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{passed + failed}")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} test(s) failed")
