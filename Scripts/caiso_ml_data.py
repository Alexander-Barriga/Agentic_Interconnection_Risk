"""Data preparation for the CAISO Cluster 14-and-prior project-realization classifier.

Role in the pipeline
--------------------
This module turns the raw CAISO queue workbook into a leakage-safe modeling table.
It is imported by ``Notebooks/ML_Model_CAISO_Cluster_14_and_Prior_Interconnection_Requests.ipynb``
and, later, by the agent's ML tool so that training and scoring use identical logic.

Prediction question
-------------------
"For a project entering the CAISO interconnection queue, what is the probability that it
eventually reaches commercial operation before it withdraws?"

* ``reached_operation = 1``: the project appears on the *Completed Generation Projects* sheet.
* ``reached_operation = 0``: the project appears on the *Withdrawn Generation Projects* sheet.
* Projects on the *Grid GenerationQueue* (active) sheet have no final outcome yet. They are
  **never** used as labels. They are used only to measure label maturity (see
  :func:`maturity_table`) and, later, as scoring targets for the agent.

The target is not "approval versus rejection"; CAISO does not record such a label here.

Main safeguards
---------------
1. **Leakage control.** Only information known when the request entered the queue is used
   as a feature. Withdrawal dates, online dates, study milestones, and agreement status are
   recorded after queue entry and often encode the outcome, so they are excluded
   (:data:`LEAKAGE_COLUMNS`).
2. **Label maturity.** Withdrawals happen quickly but reaching operation takes years, so
   recent queue cohorts look artificially unsuccessful. :func:`choose_cutoff_year` picks the
   latest queue year whose outcomes are mostly resolved; later cohorts are excluded from the
   main model.
3. **Train-only fitting.** Anything learned from data (median imputation, clipping bounds,
   scaling, category encoding) lives inside scikit-learn pipelines built by
   :func:`build_preprocessor`, so it is fit on the training split only.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Workbook constants
# ---------------------------------------------------------------------------

#: Maps each workbook sheet to the outcome it represents.
SHEET_OUTCOME = {
    "Grid GenerationQueue": "active",
    "Completed Generation Projects": "operational",
    "Withdrawn Generation Projects": "withdrawn",
}

#: Short, readable names for the long CAISO column headers (same map as the EDA notebook).
RENAME_MAP = {
    "project_name_confidential": "project_name",
    "proposed_on_line_date_as_filed_with_ir": "proposed_online_date",
    "current_on_line_date": "current_online_date",
    "actual_on_line_date": "actual_online_date",
    "interconnection_request_receive_date": "ir_receive_date",
    "net_mws_to_grid": "net_mw_to_grid",
    "full_capacity_partial_or_energy_only_fc_p_eo": "service_type",
    "interconnection_agreement_status": "ia_status",
    "feasibility_study_or_supplemental_review": "feasibility_study",
    "system_impact_study_or_phase_i_cluster_study": "phase1_study",
    "facilities_study_fas_or_phase_ii_cluster_study": "phase2_study",
    "optional_study_os": "optional_study",
    "station_or_transmission_line": "poi_station",
    "off_peak_deliverability_and_economic_only": "off_peak_deliverability",
}

DATE_COLUMNS = [
    "ir_receive_date", "queue_date", "proposed_online_date",
    "current_online_date", "actual_online_date", "withdrawn_date",
]
NUMERIC_COLUMNS = ["mw_1", "mw_2", "mw_3", "net_mw_to_grid", "tpd_allocation_percentage"]

#: Unique project identifier. Kept for joins and holdout tracking, never used as a feature.
ID_COLUMN = "queue_position"
TARGET = "reached_operation"

#: Columns that must never become features, with the reason each one is excluded.
LEAKAGE_COLUMNS = {
    "withdrawn_date": "Only exists for withdrawn projects: directly reveals the outcome.",
    "reason_for_withdrawal": "Only exists for withdrawn projects: directly reveals the outcome.",
    "actual_online_date": "Only exists for operational projects: directly reveals the outcome.",
    "current_online_date": "Latest schedule, updated after queue entry; missing by sheet.",
    "ia_status": "Interconnection-agreement status is a later milestone, not known at entry.",
    "feasibility_study": "Study milestone reached after queue entry.",
    "phase1_study": "Study milestone reached after queue entry; missing mostly for withdrawals.",
    "phase2_study": "Study milestone reached after queue entry; missing mostly for withdrawals.",
    "optional_study": "Study milestone reached after queue entry.",
    "tpd_allocation_percentage": "Deliverability allocation happens after studies; missing on the withdrawn sheet.",
    "tpd_allocation_group": "Deliverability allocation happens after studies; missing on the withdrawn sheet.",
    "off_peak_deliverability": "Deliverability result assigned after studies.",
    "suspension_status": "Current status field; only present on some sheets.",
    "application_status": "Restates sheet membership, i.e. the outcome itself.",
    "source_sheet": "Restates the outcome itself.",
    "outcome": "The outcome itself (target source).",
    "project_name": "Free-text identifier; confidential for many projects.",
    "pto_study_region": "Missing for about 94% of projects and assigned during studies.",
    ID_COLUMN: "Identifier only.",
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def snake_case(value) -> str:
    """Convert a raw CAISO column header to ``snake_case`` (e.g. ``"Net MWs to Grid"`` -> ``"net_mws_to_grid"``)."""
    value = str(value).replace("\n", " ").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def load_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Load one workbook sheet and clean it exactly as the EDA notebook did.

    The CAISO export has three title rows above the header, a ``Report Run Date``
    column, and a disclaimer row at the bottom; all of these are removed.

    Parameters
    ----------
    path : Path
        Location of ``CASIO_Cluster_14_and_Prior_Interconnection_Requests.xlsx``.
    sheet_name : str
        One of the keys of :data:`SHEET_OUTCOME`.

    Returns
    -------
    pd.DataFrame
        One row per project with snake_case columns, plus ``source_sheet`` and ``outcome``.
    """
    frame = pd.read_excel(path, sheet_name=sheet_name, skiprows=3)
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Report Run Date")]
    first_col = frame.columns[0]
    is_project_row = frame[first_col].notna()
    is_disclaimer = frame[first_col].astype(str).str.startswith("The contents of these pages")
    frame = frame.loc[is_project_row & ~is_disclaimer].copy()
    frame.columns = [snake_case(col) for col in frame.columns]
    frame = frame.rename(columns=RENAME_MAP)
    frame["source_sheet"] = sheet_name
    frame["outcome"] = SHEET_OUTCOME[sheet_name]
    return frame


def load_workbook(path: Path) -> pd.DataFrame:
    """Load all three sheets into one table with parsed dates and numeric capacities.

    The active sheet is included so its projects can be counted in the label-maturity
    diagnostic. Callers must filter to ``outcome in {"operational", "withdrawn"}`` before
    building training labels (see :func:`resolved_projects`).
    """
    frames = [load_sheet(path, name) for name in SHEET_OUTCOME]
    projects = pd.concat(frames, ignore_index=True, sort=False)
    for col in DATE_COLUMNS:
        if col not in projects:
            projects[col] = pd.NaT
        projects[col] = pd.to_datetime(projects[col], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col not in projects:
            projects[col] = np.nan
        projects[col] = pd.to_numeric(projects[col], errors="coerce")
    projects["queue_year"] = projects["queue_date"].dt.year
    projects["is_solar"] = projects["fuel_1"].astype(str).str.strip().eq("Solar")
    return projects


def resolved_projects(projects: pd.DataFrame) -> pd.DataFrame:
    """Return only projects with a final outcome and attach the binary target.

    Active projects are dropped here because their outcome is unknown (right-censored);
    labeling them either way would inject false labels.
    """
    resolved = projects[projects["outcome"].isin(["operational", "withdrawn"])].copy()
    resolved[TARGET] = (resolved["outcome"] == "operational").astype(int)
    return resolved


def data_quality_report(projects: pd.DataFrame) -> pd.DataFrame:
    """Summarize duplicates and date-logic problems for each sheet.

    Checks:
    * duplicate project IDs within a sheet and across sheets (a project cannot be both
      operational and withdrawn);
    * a requested online date earlier than the queue date (impossible request);
    * a queue date earlier than the interconnection-request receipt date by more than a
      week (date-entry error).
    """
    rows = []
    cross_sheet_dupes = projects[ID_COLUMN].duplicated(keep=False)
    for sheet, frame in projects.groupby("source_sheet"):
        horizon = (frame["proposed_online_date"] - frame["queue_date"]).dt.days
        ir_gap = (frame["queue_date"] - frame["ir_receive_date"]).dt.days
        rows.append({
            "sheet": sheet,
            "projects": len(frame),
            "duplicate_ids_within_sheet": int(frame[ID_COLUMN].duplicated().sum()),
            "ids_also_on_other_sheet": int(cross_sheet_dupes.loc[frame.index].sum()),
            "missing_queue_date": int(frame["queue_date"].isna().sum()),
            "requested_online_before_queue": int((horizon < 0).sum()),
            "queue_date_before_ir_receipt": int((ir_gap < -7).sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Label maturity and the queue-year cutoff
# ---------------------------------------------------------------------------


def maturity_table(projects: pd.DataFrame) -> pd.DataFrame:
    """Count outcomes by queue year and measure how unresolved each cohort still is.

    Columns
    -------
    active, operational, withdrawn : project counts.
    total : all projects that entered the queue that year.
    active_share : ``active / total``; the fraction of the cohort whose outcome is still unknown.
    operational_rate_resolved : ``operational / (operational + withdrawn)``; the rate a
        classifier would learn from that year's resolved labels.

    A high ``active_share`` means many eventual successes have not happened yet, so the
    resolved labels for that year are biased toward withdrawal.
    """
    table = pd.crosstab(projects["queue_year"], projects["outcome"])
    for col in ["active", "operational", "withdrawn"]:
        if col not in table:
            table[col] = 0
    table = table[["active", "operational", "withdrawn"]]
    table["total"] = table.sum(axis=1)
    table["active_share"] = table["active"] / table["total"]
    resolved = table["operational"] + table["withdrawn"]
    table["operational_rate_resolved"] = table["operational"] / resolved.replace(0, np.nan)
    table.index = table.index.astype(int)
    return table


def choose_cutoff_year(table: pd.DataFrame, tolerance: float = 0.10, min_projects: int = 20) -> int:
    """Choose the latest queue year whose outcomes are mature enough to train on.

    Rule
    ----
    Consider only queue years with at least ``min_projects`` projects (years with a handful
    of projects produce unstable shares). The cutoff ``Y`` is the last such year whose
    ``active_share`` is at or below ``tolerance``, provided every later year exceeds the
    tolerance. In other words, ``Y`` is the point after which cohorts stay meaningfully
    unresolved.

    An isolated early year slightly above the tolerance (for example a small 2005 cohort
    with a few long-running projects) does not move the cutoff, because the question is
    where the *recent* immaturity begins, not whether every historical year is perfect.

    Returns
    -------
    int
        The cutoff queue year ``Y``. Resolved projects queued in or before ``Y`` form the
        mature modeling cohort.
    """
    eligible = table[table["total"] >= min_projects]
    within = eligible.index[eligible["active_share"] <= tolerance]
    if len(within) == 0:
        raise ValueError("No queue year meets the maturity tolerance.")
    cutoff = int(within.max())
    later = eligible.loc[eligible.index > cutoff, "active_share"]
    assert (later > tolerance).all(), "A later year re-enters the tolerance; review the rule."
    return cutoff


def split_by_cutoff(resolved: pd.DataFrame, cutoff_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split resolved projects into the mature modeling cohort and the post-cutoff cohort.

    Returns
    -------
    mature : resolved projects queued in or before ``cutoff_year`` (used for modeling).
    immature : resolved projects queued after ``cutoff_year`` (reported separately only;
        their labels over-represent quick withdrawals).
    """
    mature = resolved[resolved["queue_year"] <= cutoff_year].copy()
    immature = resolved[resolved["queue_year"] > cutoff_year].copy()
    return mature, immature


# ---------------------------------------------------------------------------
# Queue-entry feature engineering (deterministic, row-wise only)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "log_net_mw",
    "n_components",
    "poi_kv",
    "requested_horizon_years",
    "days_ir_to_queue",
    "queue_year",
]
BINARY_FEATURES = [
    "is_solar", "is_hybrid", "has_storage", "zero_mw",
    "horizon_invalid", "poi_kv_missing", "poi_is_line",
]
CATEGORICAL_FEATURES = ["fuel_1", "type_1", "service_type", "utility", "state", "county", "study_group"]

_SERIAL_PROCESSES = {"Serial LGIP", "AMEND 39", "Pre- Amend. 39"}
_FAST_TRACK_PROCESSES = {"SGIP", "SGIP-TC", "FT", "ISP"}
_TRANSITION_PROCESSES = {"TC"}


def study_group(process) -> str:
    """Group the 20+ CAISO study-process codes into four broad application paths.

    * ``cluster``: cluster studies ``C01``-``C14`` (the modern process).
    * ``serial``: the older one-at-a-time Large Generator Interconnection Procedure.
    * ``fast_track``: small-generator and fast-track procedures (SGIP, FT, ISP).
    * ``transition``: the transition cluster (TC).

    Individual cluster numbers are not used, because each cluster number is effectively a
    queue-date proxy and would duplicate ``queue_year``.
    """
    text = str(process).strip()
    if re.fullmatch(r"C\d+", text):
        return "cluster"
    if text in _SERIAL_PROCESSES:
        return "serial"
    if text in _FAST_TRACK_PROCESSES:
        return "fast_track"
    if text in _TRANSITION_PROCESSES:
        return "transition"
    return "other"


def parse_poi_kv(station) -> float:
    """Extract the highest voltage (kV) from a point-of-interconnection description.

    Examples: ``"Whirlwind Substation 230kV"`` -> 230; ``"Smyrna-Alpaugh 115kV line"`` -> 115.
    Returns NaN when no voltage is written.
    """
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*-?\s*k\s*v", str(station), flags=re.IGNORECASE)
    return max(float(m) for m in matches) if matches else np.nan


def _clean_category(series: pd.Series) -> pd.Series:
    """Strip and upper/normal-case text categories and label missing values explicitly.

    Missingness is kept as its own category ("Missing") because it can be informative
    (for example, service type was not filed), and it avoids imputing a false value.
    """
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA})
    return cleaned.fillna("Missing").astype(str)


def engineer_features(projects: pd.DataFrame) -> pd.DataFrame:
    """Build the queue-entry feature matrix from cleaned project records.

    Every feature uses only information filed when the project entered the queue, and each
    is computed row by row (no statistics across projects), so this step cannot leak
    information between training and test projects.

    Features
    --------
    log_net_mw : log(1 + requested MW to the grid). Capacity is right-skewed (a few
        projects request 1,000+ MW), so the log keeps large projects from dominating.
    n_components : number of generating components filed (1-3).
    poi_kv : voltage of the requested point of interconnection; higher voltages usually mean
        bulk-transmission connections.
    requested_horizon_years : years between queue entry and the online date requested in the
        application. Negative or > 15-year values are treated as data errors (NaN) and flagged.
    days_ir_to_queue : days between request receipt and queue entry (negative values are
        treated as data errors).
    queue_year : calendar year of queue entry (era proxy; tested by ablation in the notebook).
    is_solar, is_hybrid, has_storage, zero_mw, horizon_invalid, poi_kv_missing, poi_is_line :
        0/1 indicators.
    fuel_1, type_1, service_type, utility, state, county, study_group : categories.

    Returns
    -------
    pd.DataFrame indexed like ``projects`` with columns
    ``NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES``.
    """
    features = pd.DataFrame(index=projects.index)

    components = projects[["mw_1", "mw_2", "mw_3"]]
    net_mw = projects["net_mw_to_grid"].fillna(components.fillna(0).sum(axis=1))
    features["log_net_mw"] = np.log1p(net_mw.clip(lower=0))
    features["zero_mw"] = (net_mw <= 0).astype(int)
    features["n_components"] = projects[["type_1", "type_2", "type_3"]].notna().sum(axis=1).clip(lower=1)

    fuels = projects[["fuel_1", "fuel_2", "fuel_3"]].astype(str)
    types = projects[["type_1", "type_2", "type_3"]].astype(str)
    features["is_solar"] = projects["fuel_1"].astype(str).str.strip().eq("Solar").astype(int)
    features["is_hybrid"] = projects[["type_2", "type_3", "fuel_2", "fuel_3"]].notna().any(axis=1).astype(int)
    features["has_storage"] = (
        fuels.apply(lambda col: col.str.contains("Battery", case=False)).any(axis=1)
        | types.apply(lambda col: col.str.contains("Storage", case=False)).any(axis=1)
    ).astype(int)

    features["poi_kv"] = projects["poi_station"].map(parse_poi_kv)
    features["poi_kv_missing"] = features["poi_kv"].isna().astype(int)
    features["poi_is_line"] = projects["poi_station"].astype(str).str.contains(r"\bline\b", case=False).astype(int)

    horizon = (projects["proposed_online_date"] - projects["queue_date"]).dt.days / 365.25
    invalid = horizon.isna() | (horizon < 0) | (horizon > 15)
    features["requested_horizon_years"] = horizon.where(~invalid)
    features["horizon_invalid"] = invalid.astype(int)

    ir_gap = (projects["queue_date"] - projects["ir_receive_date"]).dt.days
    features["days_ir_to_queue"] = ir_gap.where(ir_gap >= 0)

    features["queue_year"] = projects["queue_year"].astype(float)

    for col in ["fuel_1", "type_1", "service_type", "utility", "state"]:
        features[col] = _clean_category(projects[col])
    features["county"] = _clean_category(projects["county"]).str.upper()
    features["study_group"] = projects["study_process"].map(study_group)

    return features[NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES]


def assert_no_leakage(features: pd.DataFrame) -> None:
    """Fail loudly if any excluded post-entry or outcome column reached the feature matrix."""
    leaked = sorted(set(features.columns) & set(LEAKAGE_COLUMNS))
    assert not leaked, f"Leakage columns found in features: {leaked}"
    assert TARGET not in features.columns, "Target column is in the feature matrix."


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def stratified_three_way_split(
    frame: pd.DataFrame,
    train_size: float = 0.60,
    test_size: float = 0.30,
    validation_size: float = 0.10,
    seed: int = 42,
) -> dict[str, pd.Index]:
    """Split projects into train / test / validation index sets, stratified by outcome and solar.

    Stratifying on both ``reached_operation`` and ``is_solar`` keeps the operational share
    and the solar share nearly identical across the three sets, which matters because the
    positive class is small.

    The validation set is carved out first and is meant to stay untouched until the full
    agent is evaluated.

    Returns
    -------
    dict with keys ``"train"``, ``"test"``, ``"validation"`` mapping to row indices of ``frame``.
    """
    assert abs(train_size + test_size + validation_size - 1.0) < 1e-9
    strata = frame[TARGET].astype(str) + "_" + frame["is_solar"].astype(int).astype(str)
    remaining, validation = train_test_split(
        frame.index, test_size=validation_size, stratify=strata, random_state=seed
    )
    test_share_of_remaining = test_size / (train_size + test_size)
    train, test = train_test_split(
        remaining, test_size=test_share_of_remaining, stratify=strata.loc[remaining], random_state=seed
    )
    splits = {"train": pd.Index(train), "test": pd.Index(test), "validation": pd.Index(validation)}
    ids = {name: set(frame.loc[idx, ID_COLUMN]) for name, idx in splits.items()}
    assert not ids["train"] & ids["test"], "Train/test overlap"
    assert not ids["train"] & ids["validation"], "Train/validation overlap"
    assert not ids["test"] & ids["validation"], "Test/validation overlap"
    return splits


# ---------------------------------------------------------------------------
# Train-only preprocessing
# ---------------------------------------------------------------------------


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to quantile bounds learned from the training data.

    Extreme values (for example a 2,000 MW request or a 15-year requested horizon) are
    legitimate but rare; clipping limits their pull on linear models without deleting the
    projects. Bounds are learned in ``fit`` (training split only) and reused unchanged
    for test, validation, and new projects.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(
    numeric: list[str],
    binary: list[str],
    categorical: list[str],
    scale: bool,
    min_frequency: int = 10,
) -> ColumnTransformer:
    """Create the column preprocessing used inside every model pipeline.

    * Numeric: median imputation, quantile clipping, and (for Logistic Regression only)
      standardization so coefficients are comparable and regularization is fair.
    * Binary: passed through unchanged.
    * Categorical: one-hot encoding. Categories with fewer than ``min_frequency`` training
      projects are pooled into an "infrequent" bucket, and categories never seen in training
      are mapped to that bucket at scoring time instead of raising an error.

    Because this object sits inside a ``Pipeline``, all of its statistics are learned from
    the training folds only.
    """
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper()),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("bin", "passthrough", binary),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_frequency,
                                  sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


class CatBoostCategoricalClassifier(ClassifierMixin, BaseEstimator):
    """Thin scikit-learn wrapper around ``catboost.CatBoostClassifier``.

    CatBoost copies its ``cat_features`` argument internally, which breaks
    ``sklearn.base.clone`` (used by ``RandomizedSearchCV`` and ``CalibratedClassifierCV``).
    This wrapper stores the constructor arguments untouched and builds the CatBoost model
    in ``fit``. Categorical columns are passed as text ("Missing" for blanks) and encoded
    by CatBoost's ordered target statistics, so no one-hot encoding is needed.
    """

    def __init__(self, cat_features=(), depth=4, learning_rate=0.05, iterations=500, l2_leaf_reg=3.0,
                 auto_class_weights="Balanced", random_seed=42, thread_count=4):
        self.cat_features = cat_features
        self.depth = depth
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_leaf_reg = l2_leaf_reg
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.thread_count = thread_count

    def fit(self, X, y):
        from catboost import CatBoostClassifier  # imported lazily so the module loads without CatBoost

        self.model_ = CatBoostClassifier(
            cat_features=list(self.cat_features), depth=self.depth, learning_rate=self.learning_rate,
            iterations=self.iterations, l2_leaf_reg=self.l2_leaf_reg,
            auto_class_weights=self.auto_class_weights, random_seed=self.random_seed,
            thread_count=self.thread_count, verbose=0, allow_writing_files=False,
        )
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Scoring helper for the agent
# ---------------------------------------------------------------------------


def risk_tier(probability: float, low_threshold: float, high_threshold: float) -> str:
    """Map a calibrated probability to the tier shown in the agent's report."""
    if probability >= high_threshold:
        return "higher realization likelihood"
    if probability < low_threshold:
        return "lower realization likelihood"
    return "moderate realization likelihood"


def score_projects(model, metadata: dict, projects: pd.DataFrame) -> pd.DataFrame:
    """Score raw project records with the saved model and return the agent tool contract.

    Parameters
    ----------
    model : fitted scikit-learn estimator with ``predict_proba``.
    metadata : contents of ``Models/model_metadata.json``.
    projects : cleaned project records (output of :func:`load_workbook`, any sheet).

    Returns
    -------
    pd.DataFrame with one row per project: probability of reaching operation before
    withdrawal, risk tier, model version, and caveats (including a warning when the project
    was queued after the training cutoff, i.e. outside the training period).
    """
    features = engineer_features(projects)[metadata["features"]]
    probabilities = model.predict_proba(features)[:, 1]
    cutoff = metadata["maturity_cutoff_year"]
    low, high = metadata["tier_thresholds"]["low"], metadata["tier_thresholds"]["high"]
    rows = []
    for idx, prob in zip(projects.index, probabilities):
        caveats = [
            "Probability of reaching commercial operation before withdrawal; not an approval forecast.",
            "Based on historical associations, not causes.",
        ]
        year = projects.at[idx, "queue_year"]
        if pd.notna(year) and year > cutoff:
            caveats.append(
                f"Project queued in {int(year)}, after the training cutoff ({cutoff}); "
                "the estimate extrapolates from older cohorts. Treat it as a relative ranking against "
                "peers rather than a literal probability: in time-based tests the model over-predicted "
                "later cohorts, largely because storage-paired projects became the norm."
            )
        rows.append({
            ID_COLUMN: projects.at[idx, ID_COLUMN],
            "probability_reaching_operation": float(prob),
            "risk_tier": risk_tier(prob, low, high),
            "model_version": metadata["model_version"],
            "caveats": caveats,
        })
    return pd.DataFrame(rows)
