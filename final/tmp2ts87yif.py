import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# 1️⃣  Feature engineering helpers
# ----------------------------------------------------------------------


def _fill_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """Missing amenity values mean “did not spend” → 0."""
    amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in amenities:
        df[col] = df[col].fillna(0)
    return df


def _parse_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """Split Cabin (deck/num/side) into three separate columns."""
    cabin = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["CabinSide"] = cabin[2]
    return df


def _extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title (Mr, Mrs, Miss, …) from Name."""
    df["Title"] = df["Name"].str.extract(r"([A-Za-z]+)\.", expand=False)
    return df


def _add_total_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of all amenity‑spending columns."""
    amenities = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[amenities].sum(axis=1)
    return df


def _add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group identifier and size."""
    df["GroupID"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("GroupID")["GroupID"].transform("size")
    df["IsAlone"] = (df["GroupSize"] == 1).astype(int)
    return df


def _fill_age_and_cabin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing Age and CabinNum using the median of the passenger’s own group,
    then fall back to the global median.
    """
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df.groupby("GroupID")["Age"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Age"].fillna(df["Age"].median(), inplace=True)

    if "CabinNum" in df.columns:
        df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")
        df["CabinNum"] = df.groupby("GroupID")["CabinNum"].transform(
            lambda s: s.fillna(s.median())
        )
        df["CabinNum"].fillna(df["CabinNum"].median(), inplace=True)
    return df


def _add_numeric_bool(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric versions of CryoSleep and VIP – needed for group statistics."""
    df["CryoSleep_num"] = df["CryoSleep"].map({True: 1, False: 0})
    df["VIP_num"] = df["VIP"].map({True: 1, False: 0})
    return df


def _add_group_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Group level statistics for several numeric columns."""
    agg_cols = ["Age", "TotalSpend", "CabinNum", "CryoSleep_num", "VIP_num"]
    for col in agg_cols:
        df[f"Group_{col}_Mean"] = df.groupby("GroupID")[col].transform("mean")
    df["Group_Age_Std"] = df.groupby("GroupID")["Age"].transform("std")
    df["Group_TotalSpend_Sum"] = df.groupby("GroupID")["TotalSpend"].transform("sum")
    return df


def _add_group_mode_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    """
    For each categorical column compute the *mode* inside each group
    and a flag that tells whether the passenger follows that mode.
    """
    for col in cols:
        # treat missing as a regular category
        df[col] = df[col].fillna("Missing")
        mode_series = (
            df.groupby("GroupID")[col]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Missing")
        )
        df[f"{col}_GroupMode"] = df["GroupID"].map(mode_series)
        df[f"{col}_IsMode"] = (df[col] == df[f"{col}_GroupMode"]).astype(int)
    return df


def _add_misc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Age buckets and a few simple engineered columns."""
    bins = [0, 12, 18, 35, 60, 120]
    df["AgeBin"] = pd.cut(df["Age"], bins=bins, labels=False, right=False)
    # make the bucket categorical (string) – CatBoost works better with strings
    df["AgeBin"] = df["AgeBin"].astype(object).where(df["AgeBin"].notna(), "Missing")
    df["AgeBin"] = df["AgeBin"].astype(str)
    return df


def _add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Differences / ratios between personal and group statistics."""
    df["AgeMinusGroupMean"] = df["Age"] - df["Group_Age_Mean"]
    df["SpendMinusGroupMean"] = df["TotalSpend"] - df["Group_TotalSpend_Mean"]
    df["SpendPerPerson"] = df["TotalSpend"] / df["GroupSize"]
    return df


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete engineering pipeline – returns the enhanced dataframe (still
    containing the target column ``Transported`` if present).
    """
    df = _fill_amenities(df)
    df = _parse_cabin(df)
    df = _extract_title(df)
    df = _add_total_spend(df)
    df = _add_group_features(df)
    df = _fill_age_and_cabin(df)
    df = _add_numeric_bool(df)
    df = _add_group_aggregations(df)

    # categorical columns for which we want a group‑mode feature
    cat_cols = ["HomePlanet", "Destination", "CabinDeck", "CabinSide", "Title"]
    df = _add_group_mode_features(df, cat_cols)

    df = _add_misc_features(df)
    df = _add_diff_features(df)

    # -----  Categorical missing handling (sentinel)  -----
    for col in cat_cols + [f"{c}_GroupMode" for c in cat_cols]:
        df[col] = df[col].fillna("Missing")

    # CryoSleep and VIP: CatBoost expects them as strings (category)
    for col in ["CryoSleep", "VIP"]:
        df[col] = df[col].astype(str).replace("nan", "Missing")

    # drop columns that are no longer useful for modelling
    df.drop(columns=["Cabin", "Name"], inplace=True, errors="ignore")
    return df


# ----------------------------------------------------------------------
# 2️⃣  Simple helpers for the XGBoost fallback
# ----------------------------------------------------------------------


def _bool_to_int(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Map True/False → 1/0, keep NaNs untouched."""
    for c in cols:
        df[c] = df[c].map({True: 1, False: 0})
    return df


def _encode_categoricals(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Integer‑encode every object column (except the target and the
    ``is_train`` flag).  Missing values become the sentinel string ``Missing``.
    """
    obj = df.select_dtypes(include=["object"]).columns.tolist()
    for col in [target_col, "is_train"]:
        if col in obj:
            obj.remove(col)

    for col in obj:
        df[col] = df[col].fillna("Missing")
        df[col], _ = pd.factorize(df[col])
    return df


def _fill_numeric_missing(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Median imputation for all numeric columns."""
    for col in X_train.columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    return X_train, X_test


# ----------------------------------------------------------------------
# 3️⃣  Model‑building helpers
# ----------------------------------------------------------------------


def _train_catboost(
    X_tr, y_tr, X_va, y_va, cat_features, class_weights
):
    """
    Train CatBoost with early stopping, then retrain on the whole data using the
    best number of trees found.
    Returns the final model, the early‑stopping model (used for threshold search)
    and the optimal number of trees.
    """
    from catboost import CatBoostClassifier

    early = CatBoostClassifier(
        iterations=8000,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="Accuracy",
        early_stopping_rounds=300,
        random_seed=42,
        verbose=False,
        thread_count=2,
        class_weights=class_weights,
    )
    early.fit(
        X_tr,
        y_tr,
        cat_features=cat_features,
        eval_set=(X_va, y_va),
        use_best_model=True,
    )
    best_iter = early.get_best_iteration() + 1

    final = CatBoostClassifier(
        iterations=best_iter,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        thread_count=2,
        class_weights=class_weights,
    )
    final.fit(
        pd.concat([X_tr, X_va]),
        pd.concat([y_tr, y_va]),
        cat_features=cat_features,
    )
    return final, early, best_iter


def _train_xgb(
    X_tr, y_tr, X_va, y_va
):
    """Train XGBoost with early stopping and return final & early models."""
    from xgboost import XGBClassifier

    # class‑weighting for the imbalanced problem
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    scale_pos_weight = n_neg / max(1, n_pos)

    base_params = dict(
        n_estimators=5000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=2,
        random_state=42,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    early = XGBClassifier(**base_params)
    early.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=100,
        verbose=False,
    )
    best_n = early.best_iteration + 1 if early.best_iteration is not None else base_params["n_estimators"]
    final = XGBClassifier(**{**base_params, "n_estimators": best_n})
    final.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
    return final, early


# ----------------------------------------------------------------------
# 4️⃣  Main entry point
# ----------------------------------------------------------------------


def run() -> pd.DataFrame:
    """
    Loads the Spaceship Titanic data, trains a model and returns a submission DataFrame
    with columns ``PassengerId`` and ``Transported`` (boolean values).
    """
    # --------------------------------------------------------------
    # 1️⃣  Load data
    # --------------------------------------------------------------
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    test_ids = test_df["PassengerId"].copy()

    # --------------------------------------------------------------
    # 2️⃣  Combine for identical feature creation
    # --------------------------------------------------------------
    train_df["is_train"] = 1
    test_df["is_train"] = 0
    combined = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    # --------------------------------------------------------------
    # 3️⃣  Feature engineering (shared by both paths)
    # --------------------------------------------------------------
    combined = _preprocess(combined)

    # --------------------------------------------------------------
    # 4️⃣  Separate back into train / test
    # --------------------------------------------------------------
    train_mask = combined["is_train"] == 1

    # target handling – keep the original boolean column for now
    y = combined.loc[train_mask, "Transported"].astype(int)

    # drop the raw target – the model will see only engineered columns
    combined.drop(columns=["Transported"], inplace=True)

    # feature matrices (drop identifiers that would trivially leak)
    drop_cols = ["PassengerId", "is_train"]
    X = combined.loc[train_mask].drop(columns=drop_cols)
    X_test = combined.loc[~train_mask].drop(columns=drop_cols)

    # --------------------------------------------------------------
    # 5️⃣  Basic numeric imputation (median of the training part)
    # --------------------------------------------------------------
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns
    for c in num_cols:
        median_val = X[c].median()
        X[c].fillna(median_val, inplace=True)
        X_test[c].fillna(median_val, inplace=True)

    # --------------------------------------------------------------
    # 6️⃣  Try CatBoost first (handles categoricals natively)
    # --------------------------------------------------------------
    try:
        from catboost import CatBoostClassifier  # noqa: F401
        catboost_ok = True
    except Exception:  # pragma: no cover
        catboost_ok = False

    if catboost_ok:
        # ----- CatBoost specific preparation -----
        # – fill missing categoricals with the sentinel string
        obj_cols = X.select_dtypes(include=["object"]).columns
        for c in obj_cols:
            X[c] = X[c].fillna("Missing")
            X_test[c] = X_test[c].fillna("Missing")

        # – turn the original bool columns into strings so that CatBoost
        #   treats them as categorical (they already have dtype object after
        #   preprocessing, but we make sure)
        for c in ["CryoSleep", "VIP"]:
            if c in X.columns:
                X[c] = X[c].astype(str).replace("nan", "Missing")
                X_test[c] = X_test[c].astype(str).replace("nan", "Missing")

        # cat_features = list of column names with object dtype
        cat_features = list(X.select_dtypes(include=["object"]).columns)

        # ----- Train / validation split -----
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        # class‑weighting (positive class is the minority)
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        weight = neg / max(1, pos)
        class_weights = [1.0, weight]

        # ----- Model training with early stopping -----
        final_model, early_model, best_iter = _train_catboost(
            X_tr, y_tr, X_va, y_va, cat_features, class_weights
        )

        # ----- Threshold optimisation on the validation split -----
        val_proba = early_model.predict_proba(X_va)[:, 1]
        thresholds = np.arange(0.10, 0.90, 0.01)
        best_thr = 0.5
        best_acc = ((val_proba > best_thr).astype(int) == y_va.values).mean()
        for thr in thresholds:
            acc = ((val_proba > thr).astype(int) == y_va.values).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = thr

        # ----- Predict on the real test set -----
        test_proba = final_model.predict_proba(X_test)[:, 1]
        test_pred_bool = test_proba > best_thr

    else:
        # --------------------------------------------------------------
        # 7️⃣  Fallback: integer‑encode categoricals → XGBoost / RF
        # --------------------------------------------------------------
        # – Boolean columns become numeric (True/False → 1/0)
        X = _bool_to_int(X, ["CryoSleep", "VIP"])
        X_test = _bool_to_int(X_test, ["CryoSleep", "VIP"])

        # – factorise categoricals
        X = _encode_categoricals(X, target_col="Transported")
        X_test = _encode_categoricals(X_test, target_col="Transported")

        # – (re‑)fill any numeric NaNs that appeared after factorisation
        X, X_test = _fill_numeric_missing(X, X_test)

        # ----- Train / validation split -----
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )

        # ----- Try XGBoost, otherwise RandomForest -----
        try:
            final_model, early_model = _train_xgb(X_tr, y_tr, X_va, y_va)
            val_proba = early_model.predict_proba(X_va)[:, 1]
        except Exception:  # pragma: no cover
            from sklearn.ensemble import RandomForestClassifier

            final_model = RandomForestClassifier(
                n_estimators=800,
                max_depth=None,
                class_weight="balanced",
                n_jobs=2,
                random_state=42,
            )
            final_model.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
            val_proba = final_model.predict_proba(X_va)[:, 1]

        # ----- Threshold optimisation -----
        thresholds = np.arange(0.10, 0.90, 0.01)
        best_thr = 0.5
        best_acc = ((val_proba > best_thr).astype(int) == y_va.values).mean()
        for thr in thresholds:
            acc = ((val_proba > thr).astype(int) == y_va.values).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = thr

        # ----- Predict on test -----
        test_proba = final_model.predict_proba(X_test)[:, 1]
        test_pred_bool = test_proba > best_thr

    # --------------------------------------------------------------
    # 8️⃣  Build the submission DataFrame
    # --------------------------------------------------------------
    submission = pd.DataFrame(
        {"PassengerId": test_ids, "Transported": test_pred_bool}
    )
    # Ensure the column order matches the sample submission
    submission = submission[["PassengerId", "Transported"]]
    # Cast to proper boolean dtype
    submission["Transported"] = submission["Transported"].astype(bool)

    return submission