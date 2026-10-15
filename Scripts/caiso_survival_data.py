"""Data preparation for competing-risks survival analysis of CAISO interconnection projects.

Role in the pipeline
--------------------
The earlier classifier (``caiso_ml_data.py``) asked a yes/no question: "did a project reach
operation before it withdrew?" and had to throw away recent projects because their outcome
was not yet known. Survival analysis asks a richer question: *when* does each project reach
one of two competing outcomes, and what do we know about projects that have not finished yet?

This module turns the two CAISO workbooks into a **survival table**, one row per project:

* ``duration_years``: time from queue entry (the time origin) to the end of observation;
* ``event``: what ended observation (the coding below).

Event coding (competing risks)
------------------------------
* ``0`` = **censored / still active**: the project had not reached either outcome by the
  snapshot date. We only know it survived at least ``duration_years`` (a lower bound).
* ``1`` = **reached commercial operation** (project is on the Completed sheet).
* ``2`` = **withdrew** (project is on a Withdrawn sheet).

Operation and withdrawal are *competing* events: once one happens, the other can never happen.

Inputs
------
* ``CASIO_Cluster_14_and_Prior_Interconnection_Requests.xlsx``: Active, Completed, and
  Withdrawn sheets. Each sheet states its snapshot in a ``Report Run Date`` cell.
* ``CASIO_Cluster_15_Interconnection_Requests.xlsx``: Active ("Cluster 15 ") and Withdrawn
  sheets, which use a different layout and carry no snapshot date. We assume it shares the
  Cluster 14 snapshot (decision agreed with the project owner).

Everything here is deterministic and row-wise. Nothing is learned from outcomes, so building
the table cannot leak information. Dated outcome fields (withdrawal date, actual online date)
are used ONLY to build the time and event columns, never as features.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import caiso_ml_data as cd  # reuse the Cluster 14 loader and the queue-entry feature logic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Event codes used in the survival table.
EVENT_CENSORED, EVENT_OPERATION, EVENT_WITHDRAWAL = 0, 1, 2
EVENT_LABELS = {EVENT_CENSORED: "active (censored)", EVENT_OPERATION: "operation", EVENT_WITHDRAWAL: "withdrawal"}
OUTCOME_TO_EVENT = {"active": EVENT_CENSORED, "operational": EVENT_OPERATION, "withdrawn": EVENT_WITHDRAWAL}

DAYS_PER_YEAR = 365.25

#: Snapshot assumed for the Cluster 15 workbook, which does not state one. The latest dated event
#: in that file is 2026-07-14, so the true export is on or after that date. Agreed assumption:
#: the same snapshot as the Cluster 14 workbook. The notebook shows the sensitivity to this choice.
C15_ASSUMED_SNAPSHOT = pd.Timestamp("2026-09-24")

#: Cluster 15 sheets and the outcome each one represents.
C15_SHEETS = {"Cluster 15 ": "active", "Withdrawn": "withdrawn"}

#: Maps Cluster 15 column headers (after ``snake_case``) to the Cluster 14 names used by
#: ``caiso_ml_data.engineer_features``.
C15_RENAME = {
    "queue_number": "queue_position",
    "project_name": "project_name",
    "net_mw_1": "mw_1",
    "net_mw_2": "mw_2",
    "net_mw_3": "mw_3",
    "net_mw_poi": "net_mw_to_grid",
    "project_county": "county",
    "project_state": "state",
    "pto": "utility",
    "poi": "poi_station",
    "requested_cod": "proposed_online_date",
    "application_date": "ir_receive_date",
    "withdrawal_date": "withdrawn_date",
}

#: Cluster 15 writes technology as ``Type/Fuel`` (e.g. ``Storage/Battery``); Cluster 14 stores
#: type and fuel in separate columns. Fuel names are aligned to the Cluster 14 spelling.
_C15_FUEL_ALIASES = {"Pumped-Storage Hydro": "Pumped-Storage hydro", "Wind": "Wind Turbine"}

#: Cluster 15 service types are long phrases; collapse them to the Cluster 14 vocabulary.
_C15_SERVICE_MAP = {
    "Full Capacity Deliverability Status Requested": "Full Capacity",
    "Merchant- Full Capacity Deliverability Status Requested": "Full Capacity",
    "Energy Only Requested": "Energy Only",
}

# ---------------------------------------------------------------------------
# Snapshot dates
# ---------------------------------------------------------------------------


def read_snapshot_dates(path: Path) -> dict[str, pd.Timestamp]:
    """Read the ``Report Run Date`` stated on each sheet of the Cluster 14 workbook.

    CAISO prints ``Report Run Date: MM/DD/YYYY`` in a header cell of every sheet. This is the
    date the queue was exported, so every project still on the active sheet was alive on that
    date and is *right-censored* there. The Cluster 14 loader in ``caiso_ml_data`` drops the
    column, so it is read directly here.

    Returns
    -------
    dict mapping sheet name to the snapshot ``Timestamp``.
    """
    snapshots = {}
    for sheet in cd.SHEET_OUTCOME:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        for value in raw.to_numpy().ravel():
            match = re.search(r"Report Run Date:\s*(\d{2}/\d{2}/\d{4})", str(value))
            if match:
                snapshots[sheet] = pd.to_datetime(match.group(1), format="%m/%d/%Y")
                break
        else:
            raise ValueError(f"No 'Report Run Date' found on sheet {sheet!r}.")
    return snapshots


# ---------------------------------------------------------------------------
# Loading and harmonizing
# ---------------------------------------------------------------------------


def _cohort_label(study_process) -> str:
    """Cluster label (``C01``..``C14``) from the study-process code, else ``Pre-cluster``.

    Serial-process, fast-track, and transition projects have no cluster number; they are
    pooled as ``Pre-cluster`` so that cluster-level analyses stay readable.
    """
    text = str(study_process).strip()
    match = re.fullmatch(r"C(\d+)", text)
    return f"C{int(match.group(1)):02d}" if match else "Pre-cluster"


def load_cluster14(path: Path) -> pd.DataFrame:
    """Load the three Cluster 14-and-prior sheets with survival bookkeeping columns added.

    Adds ``project_id`` (string), ``cohort`` (cluster label), ``source`` (``"C14 workbook"``),
    and ``snapshot_date`` (the sheet's stated snapshot). Dates are normalized to midnight so
    durations are whole days.
    """
    snapshots = read_snapshot_dates(path)
    projects = cd.load_workbook(path)
    projects["project_id"] = projects[cd.ID_COLUMN].astype(str)
    projects["cohort"] = projects["study_process"].map(_cohort_label)
    projects["source"] = "C14 workbook"
    projects["snapshot_date"] = projects["source_sheet"].map(snapshots)
    for col in cd.DATE_COLUMNS:
        projects[col] = projects[col].dt.normalize()
    return projects


def _split_type_fuel(value) -> tuple[object, object]:
    """Split a Cluster 15 ``Type/Fuel`` string into (type, fuel); blanks stay NaN."""
    if pd.isna(value):
        return np.nan, np.nan
    kind, _, fuel = str(value).partition("/")
    fuel = _C15_FUEL_ALIASES.get(fuel.strip(), fuel.strip())
    return kind.strip(), fuel


def load_cluster15(path: Path, snapshot: pd.Timestamp = C15_ASSUMED_SNAPSHOT) -> pd.DataFrame:
    """Load Cluster 15 (Active and Withdrawn sheets) in the Cluster 14 column layout.

    Cluster 15 differs from Cluster 14 in four ways, all handled here:

    * technology is a ``Type/Fuel`` string, split into ``type_i`` and ``fuel_i``;
    * there is no study-process, interconnection-agreement, or study-milestone information
      (``study_process`` is set to ``C15``, since every project is in the cluster process);
    * the interconnection-request date is called *Application Date* (stored as
      ``ir_receive_date``) and the requested online date is called *Requested COD*;
    * service type and county are written differently (``Full Capacity Deliverability Status
      Requested`` and ``Merced County``), so they are normalized.

    Parameters
    ----------
    snapshot : the assumed export date, used as the censoring date for active projects.
    """
    frames = []
    for sheet, outcome in C15_SHEETS.items():
        frame = pd.read_excel(path, sheet_name=sheet)
        frame.columns = [cd.snake_case(c) for c in frame.columns]
        frame = frame[frame["queue_number"].notna()].copy()  # drop any blank trailing rows
        frame = frame.rename(columns=C15_RENAME)
        frame["source_sheet"] = f"C15 {sheet.strip()}"
        frame["outcome"] = outcome
        frames.append(frame)
    projects = pd.concat(frames, ignore_index=True, sort=False)

    # Technology: "Photovoltaic/Solar" -> type "Photovoltaic", fuel "Solar" for components 1-3.
    for i in (1, 2, 3):
        pieces = projects[f"generation_fuel_{i}"].map(_split_type_fuel)
        projects[f"type_{i}"] = pieces.map(lambda t: t[0])
        projects[f"fuel_{i}"] = pieces.map(lambda t: t[1])

    for col in ["ir_receive_date", "queue_date", "proposed_online_date", "withdrawn_date"]:
        projects[col] = pd.to_datetime(projects.get(col), errors="coerce").dt.normalize()
    projects["actual_online_date"] = pd.NaT  # Cluster 15 cannot have operational projects yet
    for col in ["mw_1", "mw_2", "mw_3", "net_mw_to_grid", "voltage_kv"]:
        projects[col] = pd.to_numeric(projects[col], errors="coerce")

    projects["service_type"] = projects["service_type"].map(_C15_SERVICE_MAP).fillna(projects["service_type"])
    projects["county"] = (projects["county"].astype("string").str.replace(r"\s+County$", "", regex=True)
                          .str.strip())
    projects["study_process"] = "C15"
    projects["queue_year"] = projects["queue_date"].dt.year
    projects["is_solar"] = projects["fuel_1"].astype(str).str.strip().eq("Solar")
    projects["project_id"] = projects["queue_position"].astype(str)
    projects["cohort"] = "C15"
    projects["source"] = "C15 workbook"
    projects["snapshot_date"] = snapshot
    return projects


def load_all_projects(c14_path: Path, c15_path: Path, c15_snapshot: pd.Timestamp = C15_ASSUMED_SNAPSHOT) -> pd.DataFrame:
    """Load Cluster 14-and-prior plus Cluster 15 into one table (all outcomes, all cohorts).

    Asserts that project IDs are unique across the two workbooks (a project cannot appear in
    both) before combining them.
    """
    c14 = load_cluster14(c14_path)
    c15 = load_cluster15(c15_path, snapshot=c15_snapshot)
    overlap = set(c14["project_id"]) & set(c15["project_id"])
    assert not overlap, f"Project IDs appear in both workbooks: {sorted(overlap)[:5]}"
    combined = pd.concat([c14, c15], ignore_index=True, sort=False)
    assert combined["project_id"].is_unique, "Duplicate project IDs in the combined table."
    return combined


# ---------------------------------------------------------------------------
# The survival table
# ---------------------------------------------------------------------------


def build_survival_table(projects: pd.DataFrame) -> pd.DataFrame:
    """Build one survival row per project: time origin, end date, duration, and event code.

    Definitions
    -----------
    * **Time origin**: ``queue_date`` (the day the project entered the queue). All projects
      in a cluster share one queue date, which keeps the clock comparable within a cluster.
    * **End date**: the withdrawal date, the actual online date, or the snapshot date for
      active projects (their observation is cut off there).
    * **Event**: see :data:`EVENT_LABELS`.

    Data-quality flags (added, nothing is silently repaired):
    * ``event_date_missing``: the project has an outcome but no dated event, so its true
      duration is unknown (``duration_days`` is NaN);
    * ``duration_nonpositive``: the end date is on or before the queue date;
    * ``event_after_snapshot``: a dated event falls after the snapshot (must be zero).

    Dated outcome columns are consumed here and must never be used as features.
    """
    table = projects[["project_id", cd.ID_COLUMN, "cohort", "source", "outcome", "queue_date", "queue_year",
                      "snapshot_date"]].copy()
    table["event"] = projects["outcome"].map(OUTCOME_TO_EVENT).astype(int)

    # The date the project reached its outcome; active projects have none.
    event_date = pd.Series(pd.NaT, index=projects.index, dtype="datetime64[ns]")
    event_date = event_date.where(projects["outcome"] != "operational", projects["actual_online_date"])
    event_date = event_date.where(projects["outcome"] != "withdrawn", projects["withdrawn_date"])
    table["event_date"] = event_date

    # Active projects are observed until the snapshot; resolved projects until their event.
    end_date = event_date.where(projects["outcome"] != "active", projects["snapshot_date"])
    table["end_date"] = end_date
    table["duration_days"] = (end_date - table["queue_date"]).dt.days.astype(float)
    table["duration_years"] = table["duration_days"] / DAYS_PER_YEAR

    table["event_date_missing"] = (table["event"] > 0) & table["event_date"].isna()
    table["duration_nonpositive"] = table["duration_days"] <= 0
    table["event_after_snapshot"] = (table["event"] > 0) & (table["event_date"] > table["snapshot_date"])
    return table


def repair_durations(table: pd.DataFrame, min_days: float = 1.0) -> pd.DataFrame:
    """Set non-positive durations to ``min_days`` and record which rows were changed.

    Survival estimators need strictly positive times. A handful of projects have an event dated
    on or before their queue date (a recording error, or an event backdated to the application).
    Treating them as events at a tiny positive time keeps them in the data without letting a
    zero or negative time break the estimators. The count is small; the notebook shows it.
    """
    table = table.copy()
    bad = table["duration_days"].notna() & (table["duration_days"] < min_days)
    table["duration_repaired"] = bad
    table.loc[bad, "duration_days"] = min_days
    table["duration_years"] = table["duration_days"] / DAYS_PER_YEAR
    return table


def resolve_missing_event_dates(table: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Handle resolved projects whose event date is missing.

    Strategies
    ----------
    ``"drop"`` : remove them (the main, simplest choice; they number about 50 and are almost
        all from before 2008).
    ``"impute_median"`` : keep them, assigning the median duration of resolved projects with
        the same event type and cohort (falling back to the same event type overall). This is a
        sensitivity check: results should not change materially under either choice.
    ``"keep"`` : leave the NaN durations untouched (for counting only).

    The ``duration_imputed`` column records which rows were filled.
    """
    table = table.copy()
    table["duration_imputed"] = False
    missing = table["event_date_missing"]
    if strategy == "keep":
        return table
    if strategy == "drop":
        return table.loc[~missing].copy()
    if strategy == "impute_median":
        known = table.loc[~missing & (table["event"] > 0)]
        by_cohort = known.groupby(["event", "cohort"])["duration_days"].median()
        by_event = known.groupby("event")["duration_days"].median()
        for idx in table.index[missing]:
            key = (table.at[idx, "event"], table.at[idx, "cohort"])
            table.at[idx, "duration_days"] = by_cohort.get(key, by_event[table.at[idx, "event"]])
            table.at[idx, "duration_imputed"] = True
        table["duration_years"] = table["duration_days"] / DAYS_PER_YEAR
        return table
    raise ValueError(f"Unknown strategy {strategy!r}")


def add_entry_time(table: pd.DataFrame, window_start: pd.Timestamp | None) -> pd.DataFrame:
    """Add ``entry_years``: the project age at which observation of it began (left truncation).

    If the data only contain projects that were still alive on ``window_start`` (older projects
    that ended before it were never recorded), then a project queued before ``window_start`` was
    only *observable* from the age it had on that date. Survival estimators handle this by
    counting the project in the at-risk group only from ``entry_years`` onward ("delayed entry").
    Projects queued on or after ``window_start`` enter at age 0.

    Pass ``window_start=None`` to ignore truncation (all entries at age 0).

    ``ends_before_window`` flags projects whose end date precedes ``window_start``. Under the
    truncation assumption such projects should not exist in the data at all (they would have left
    the queue before records began), so a non-zero count means the chosen window starts too late.
    Callers should drop them (see :func:`apply_window`).
    """
    table = table.copy()
    if window_start is None:
        table["entry_years"] = 0.0
        table["ends_before_window"] = False
    else:
        start = pd.Timestamp(window_start)
        gap = (start - table["queue_date"]).dt.days.clip(lower=0)
        table["entry_years"] = gap / DAYS_PER_YEAR
        table["ends_before_window"] = table["end_date"].notna() & (table["end_date"] < start)
    return table


def apply_window(table: pd.DataFrame, window_start: pd.Timestamp | None, method: str = "delayed_entry") -> pd.DataFrame:
    """Apply an observation-window choice to a survival table.

    Methods
    -------
    ``"none"`` : ignore truncation; every project enters at age 0 (the naive treatment).
    ``"restrict"`` : keep only projects queued on or after ``window_start``; all enter at age 0.
        Simple and safe, at the price of discarding the older projects.
    ``"delayed_entry"`` : keep older projects but count them as at risk only from the age they
        had on ``window_start`` (``entry_years``). Projects that ended before ``window_start`` are
        dropped, because the data design says they could not have been recorded.
    """
    if method == "none":
        return add_entry_time(table, None)
    if method == "restrict":
        kept = table[table["queue_date"] >= pd.Timestamp(window_start)]
        return add_entry_time(kept, None)
    if method == "delayed_entry":
        flagged = add_entry_time(table, window_start)
        return flagged.loc[~flagged["ends_before_window"]].copy()
    raise ValueError(f"Unknown method {method!r}")


def assert_survival_table(table: pd.DataFrame) -> None:
    """Fail loudly if the survival table violates the basic rules of the data design.

    Rules
    -----
    * every project ID is unique;
    * events are coded 0/1/2 and active projects (and only they) have event 0;
    * no dated event falls after the snapshot;
    * no active project has an event date;
    * every project with a known duration has a strictly positive one;
    * the entry time (if present) never exceeds the duration.
    """
    assert table["project_id"].is_unique, "Duplicate project IDs."
    assert set(table["event"].unique()) <= {0, 1, 2}, "Unexpected event codes."
    assert ((table["event"] == 0) == (table["outcome"] == "active")).all(), "Event code disagrees with outcome."
    assert not table["event_after_snapshot"].any(), "An event is dated after the snapshot."
    assert table.loc[table["event"] == 0, "event_date"].isna().all(), "An active project has an event date."
    known = table["duration_days"].notna()
    assert (table.loc[known, "duration_days"] > 0).all(), "Non-positive duration remains; call repair_durations."
    if "entry_years" in table:
        # A project cannot be observed before it enters; rows ending before the window should have
        # been removed by apply_window.
        assert (table.loc[known, "entry_years"] <= table.loc[known, "duration_years"] + 1e-9).all(), \
            "A project's entry time exceeds its duration; apply_window should drop projects ending before the window."


# ---------------------------------------------------------------------------
# Queue-entry features for survival
# ---------------------------------------------------------------------------

#: Features usable in survival models. ``days_ir_to_queue`` is excluded (sheet-specific recording
#: artifact, see the classifier notebook) and so is ``queue_year`` (era proxy, handled via cohort).
SURVIVAL_NUMERIC = ["log_net_mw", "n_components", "poi_kv", "requested_horizon_years"]
SURVIVAL_BINARY = ["is_solar", "is_hybrid", "has_storage", "zero_mw", "horizon_invalid", "poi_kv_missing", "poi_is_line"]
SURVIVAL_CATEGORICAL = ["fuel_1", "type_1", "service_type", "utility", "state", "county", "study_group"]
SURVIVAL_FEATURES = SURVIVAL_NUMERIC + SURVIVAL_BINARY + SURVIVAL_CATEGORICAL


def engineer_survival_features(projects: pd.DataFrame) -> pd.DataFrame:
    """Queue-entry features for both workbooks on a common footing.

    Starts from :func:`caiso_ml_data.engineer_features` (same definitions as the classifier) and
    fixes the two places where Cluster 15 needs different handling:

    * **Voltage.** Cluster 14 gives only a station string, from which kV is parsed. Cluster 15
      also has an explicit ``Voltage kV`` column, which is used wherever the parsed value is missing.
    * **Line versus substation.** Cluster 14 marks transmission-line taps with the word "line";
      Cluster 15 writes lines as ``STATION A - STATION B``. For Cluster 15, a hyphen in the POI
      name is taken to mean a line tap (a heuristic, reported in the notebook).

    Every column is computed from information filed at queue entry.
    """
    features = cd.engineer_features(projects)
    if "voltage_kv" in projects:
        from_column = projects["voltage_kv"]
        features["poi_kv"] = features["poi_kv"].fillna(from_column)
        features["poi_kv_missing"] = features["poi_kv"].isna().astype(int)
    is_c15 = projects["cohort"].eq("C15")
    hyphenated = projects["poi_station"].astype(str).str.contains(r"\s-\s|\w-\w", regex=True).astype(int)
    features.loc[is_c15, "poi_is_line"] = hyphenated[is_c15]
    cd.assert_no_leakage(features)
    return features[SURVIVAL_FEATURES]


# ---------------------------------------------------------------------------
# Nonparametric survival estimators (exact with ties, support delayed entry)
# ---------------------------------------------------------------------------
# These are small, transparent implementations of the standard estimators so that the notebook can
# show exactly what they compute. Our data have very many tied times (all projects of a cluster
# share a queue date and many withdraw on the same day), and the exact tie-handling below avoids the
# random jitter some libraries add. They are cross-checked against ``lifelines`` in the notebook.
#
# Conventions: ``duration`` is the age (years since queue entry) at the end of observation,
# ``event`` is 0/1/2 as in the survival table, and ``entry`` is the age at which observation began
# (0 unless the project is left-truncated). A project is "at risk" at age t if entry < t <= duration.


def _risk_table(duration, event, entry=None, flag=None):
    """Event times with the number at risk and the number of events at each time.

    ``flag`` is the 0/1 indicator that counts as "an event" for this table (default: any event,
    i.e. ``event > 0``). Returns (times, n_risk, event_index) where ``event_index`` gives, for each
    flagged observation, the position of its time in ``times``.
    """
    d = np.asarray(duration, dtype=float)
    ent = np.zeros_like(d) if entry is None else np.asarray(entry, dtype=float)
    flagged = (np.asarray(event) > 0) if flag is None else np.asarray(flag).astype(bool)
    times = np.unique(d[flagged])
    # At risk at t: entered before t and not yet ended before t  ->  #(entry < t) - #(duration < t).
    n_risk = (np.searchsorted(np.sort(ent), times, side="left")
              - np.searchsorted(np.sort(d), times, side="left"))
    event_index = np.searchsorted(times, d[flagged])
    return times, n_risk, event_index, flagged


def km_survival(duration, event_flag, entry=None) -> pd.DataFrame:
    """Kaplan-Meier survival curve S(t) for an arbitrary 0/1 ``event_flag``.

    ``S(t)`` is the probability of still being event-free at age ``t``. Everyone who has not had
    the flagged event by their last observation (including projects that had a *different* event) is
    treated as censored. Used for: time in the queue (flag = any event), the naive per-cause curve
    (flag = one cause), and the follow-up distribution (reverse Kaplan-Meier: flag = censored).
    """
    times, n_risk, idx, flagged = _risk_table(duration, event_flag, entry, flag=event_flag)
    deaths = np.bincount(idx, minlength=len(times))
    survival = np.cumprod(1.0 - deaths / n_risk)
    out = pd.DataFrame({"survival": survival, "n_risk": n_risk, "n_events": deaths}, index=times)
    out.index.name = "age_years"
    return pd.concat([pd.DataFrame({"survival": [1.0], "n_risk": [np.nan], "n_events": [0]},
                                   index=pd.Index([0.0], name="age_years")), out])


def aj_cif(duration, event, cause: int, entry=None) -> pd.DataFrame:
    """Aalen-Johansen cumulative incidence of ``cause`` in the presence of competing events.

    CIF_k(t) = P(the first event to occur is cause k, and it happens by age t). At each event time
    the estimator adds ``S(t-) * d_k(t) / n(t)``: the probability of having survived (no event of any
    kind) to just before t, times the share of those still at risk who have a cause-k event at t.
    Unlike ``1 - KM`` with other events censored, the CIFs of all causes plus S(t) add to 1.
    """
    event = np.asarray(event)
    times, n_risk, idx, flagged = _risk_table(duration, event, entry)
    d_all = np.bincount(idx, minlength=len(times))
    d_cause = np.bincount(idx[event[flagged] == cause], minlength=len(times))
    survival = np.cumprod(1.0 - d_all / n_risk)
    survival_before = np.concatenate([[1.0], survival[:-1]])
    cif = np.cumsum(survival_before * d_cause / n_risk)
    out = pd.DataFrame({"cif": cif, "survival_any": survival, "n_risk": n_risk,
                        "n_cause_events": d_cause, "n_any_events": d_all}, index=times)
    out.index.name = "age_years"
    first = pd.DataFrame({"cif": [0.0], "survival_any": [1.0], "n_risk": [np.nan], "n_cause_events": [0],
                          "n_any_events": [0]}, index=pd.Index([0.0], name="age_years"))
    return pd.concat([first, out])


def step_value(curve: pd.Series, at) -> np.ndarray:
    """Value of a right-continuous step curve at the given ages (curve index must be sorted)."""
    at = np.atleast_1d(np.asarray(at, dtype=float))
    pos = np.searchsorted(curve.index.to_numpy(), at, side="right") - 1
    values = curve.to_numpy()
    return np.where(pos >= 0, values[np.clip(pos, 0, None)], 0.0 if curve.iloc[0] == 0 else 1.0)


def cif_at(duration, event, cause: int, at, entry=None) -> np.ndarray:
    """Aalen-Johansen CIF of ``cause`` evaluated at the ages in ``at``."""
    return step_value(aj_cif(duration, event, cause, entry)["cif"], at)


def bootstrap_cif_at(duration, event, cause: int, at, entry=None, n_boot: int = 300,
                     seed: int = 42, alpha: float = 0.05) -> pd.DataFrame:
    """Percentile bootstrap confidence intervals for the CIF at fixed ages.

    Projects are resampled with replacement. The intervals treat projects as independent, so they
    are optimistic: projects in the same cluster are correlated (they share a queue date and
    cluster rules), which widens the true uncertainty.
    """
    d = np.asarray(duration, dtype=float)
    e = np.asarray(event)
    ent = None if entry is None else np.asarray(entry, dtype=float)
    at = np.atleast_1d(np.asarray(at, dtype=float))
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(at)))
    for b in range(n_boot):
        pick = rng.integers(0, len(d), len(d))
        draws[b] = cif_at(d[pick], e[pick], cause, at, None if ent is None else ent[pick])
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    return pd.DataFrame({"age_years": at, "estimate": cif_at(d, e, cause, at, ent),
                         "ci_lower": lower, "ci_upper": upper})


def piecewise_hazard(duration, event, cause: int, bins, entry=None) -> pd.DataFrame:
    """Cause-specific hazard per person-year within age bands (a "life-table" hazard estimate).

    For each age band ``[lo, hi)``: events of the chosen cause that occurred in the band, divided
    by the total person-years the projects spent at risk inside the band. The hazard is the
    instantaneous rate of the event among projects still in the queue, which is why it can be read
    like a per-year "churn rate". Competing events and censoring simply end a project's exposure.
    """
    d = np.asarray(duration, dtype=float)
    e = np.asarray(event)
    ent = np.zeros_like(d) if entry is None else np.asarray(entry, dtype=float)
    bins = np.asarray(bins, dtype=float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        exposure = np.clip(np.minimum(d, hi) - np.maximum(ent, lo), 0, None).sum()
        events = int(((e == cause) & (d > lo) & (d <= hi) & (ent < d)).sum())
        rows.append({"age_from": lo, "age_to": hi, "events": events, "person_years": exposure,
                     "hazard_per_year": events / exposure if exposure > 0 else np.nan})
    return pd.DataFrame(rows)


def cause_logrank(duration, event, group, cause: int, entry=None) -> dict:
    """Two-group log-rank test for the cause-specific hazard of ``cause``, with delayed entry.

    The log-rank test asks: "among projects at risk at each moment, does group 1 have this event
    more (or less) often than it would if both groups shared one hazard?" At every event time of
    the chosen cause it compares the observed number of group-1 events with the number expected
    from group 1's share of the risk set, and pools the differences. The competing event and
    censoring only remove a project from the risk set. Delayed entry is respected through the
    risk set (a project is at risk at t only if ``entry < t <= duration``).

    Returns the observed and expected group-1 events, the chi-square statistic (1 d.f.) and the p-value.
    """
    d = np.asarray(duration, dtype=float)
    e = np.asarray(event)
    g = np.asarray(group).astype(bool)
    ent = np.zeros_like(d) if entry is None else np.asarray(entry, dtype=float)
    times = np.unique(d[e == cause])
    n_all = np.searchsorted(np.sort(ent), times, side="left") - np.searchsorted(np.sort(d), times, side="left")
    n_one = (np.searchsorted(np.sort(ent[g]), times, side="left")
             - np.searchsorted(np.sort(d[g]), times, side="left"))
    idx_all = np.searchsorted(times, d[e == cause])
    d_all = np.bincount(idx_all, minlength=len(times))
    d_one = np.bincount(np.searchsorted(times, d[(e == cause) & g]), minlength=len(times))
    expected = d_all * n_one / n_all
    variance = np.where(n_all > 1, d_all * (n_one / n_all) * (1 - n_one / n_all) * (n_all - d_all) / np.maximum(n_all - 1, 1), 0.0)
    statistic = (d_one.sum() - expected.sum()) ** 2 / variance.sum() if variance.sum() > 0 else np.nan
    from scipy import stats as _stats
    return {"observed_group1": int(d_one.sum()), "expected_group1": float(expected.sum()),
            "chi2": float(statistic), "p_value": float(_stats.chi2.sf(statistic, 1))}
