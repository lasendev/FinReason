"""Shared, leakage-resistant evaluation helpers for FinReason evidence alerts."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import pandas as pd


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def transaction_ids_from_evidence(
    evidence: pd.DataFrame,
    *,
    column: str = "transaction_ids",
) -> set[str]:
    """Return non-empty transaction IDs from pipe-delimited or iterable evidence."""

    if column not in evidence.columns:
        raise ValueError(f"evidence is missing {column!r}")
    transaction_ids: set[str] = set()
    for value in evidence[column]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        parts = value.split("|") if isinstance(value, str) else value
        transaction_ids.update(str(item) for item in parts if str(item).strip())
    return transaction_ids


def evaluate_transaction_ids(
    predicted_transaction_ids: Iterable[str],
    transactions: pd.DataFrame,
    *,
    split: str,
    label_column: str,
    id_column: str = "transaction_id",
) -> dict[str, Any]:
    """Evaluate a frozen set of transaction predictions at transaction grain."""

    required = {id_column, "split", label_column}
    missing = required - set(transactions.columns)
    if missing:
        raise ValueError(f"transaction truth is missing columns: {sorted(missing)}")
    if transactions[id_column].astype(str).duplicated().any():
        raise ValueError(f"{id_column} must be unique at transaction grain")

    target = transactions.loc[
        transactions["split"].eq(split), [id_column, label_column]
    ].copy()
    numeric_labels = pd.to_numeric(target[label_column], errors="coerce")
    known = target.loc[numeric_labels.isin([0, 1])].copy()
    known[label_column] = numeric_labels.loc[known.index].astype(int)

    known_ids = set(known[id_column].astype(str))
    positive_ids = set(known.loc[known[label_column].eq(1), id_column].astype(str))
    predicted_ids = {str(value) for value in predicted_transaction_ids} & known_ids
    true_positive = len(predicted_ids & positive_ids)
    false_positive = len(predicted_ids - positive_ids)
    false_negative = len(positive_ids - predicted_ids)
    true_negative = len(known_ids) - true_positive - false_positive - false_negative
    precision = _safe_rate(true_positive, true_positive + false_positive)
    recall = _safe_rate(true_positive, true_positive + false_negative)

    return {
        "evaluation_unit": "transaction",
        "split": split,
        "label_column": label_column,
        "population": len(known_ids),
        "expected_positives": len(positive_ids),
        "predicted_positives": len(predicted_ids),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "prevalence": _safe_rate(len(positive_ids), len(known_ids)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_rate(2 * precision * recall, precision + recall),
        "false_positive_rate": _safe_rate(false_positive, false_positive + true_negative),
    }


def evaluate_scenario_ids(
    predicted_transaction_ids: Iterable[str],
    scenario_transactions: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    split: str,
    label_column: str,
) -> dict[str, Any]:
    """Evaluate scenario detection when any member transaction is evidenced."""

    membership_required = {"scenario_id", "transaction_id"}
    scenario_required = {"scenario_id", "split", label_column}
    if membership_required - set(scenario_transactions.columns):
        raise ValueError("scenario membership is missing scenario_id or transaction_id")
    if scenario_required - set(scenarios.columns):
        raise ValueError("scenario truth is missing identity, split or label columns")
    if scenarios["scenario_id"].astype(str).duplicated().any():
        raise ValueError("scenario_id must be unique at scenario grain")

    target = scenarios.loc[
        scenarios["split"].eq(split), ["scenario_id", label_column]
    ].copy()
    numeric_labels = pd.to_numeric(target[label_column], errors="coerce")
    known = target.loc[numeric_labels.isin([0, 1])].copy()
    known[label_column] = numeric_labels.loc[known.index].astype(int)

    known_ids = set(known["scenario_id"].astype(str))
    positive_ids = set(known.loc[known[label_column].eq(1), "scenario_id"].astype(str))
    predicted_transaction_ids = {str(value) for value in predicted_transaction_ids}
    predicted_ids = set(
        scenario_transactions.loc[
            scenario_transactions["transaction_id"].astype(str).isin(predicted_transaction_ids),
            "scenario_id",
        ].astype(str)
    ) & known_ids
    true_positive = len(predicted_ids & positive_ids)
    false_positive = len(predicted_ids - positive_ids)
    false_negative = len(positive_ids - predicted_ids)
    true_negative = len(known_ids) - true_positive - false_positive - false_negative
    precision = _safe_rate(true_positive, true_positive + false_positive)
    recall = _safe_rate(true_positive, true_positive + false_negative)

    return {
        "evaluation_unit": "scenario",
        "split": split,
        "label_column": label_column,
        "population": len(known_ids),
        "expected_positives": len(positive_ids),
        "predicted_positives": len(predicted_ids),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "prevalence": _safe_rate(len(positive_ids), len(known_ids)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_rate(2 * precision * recall, precision + recall),
        "false_positive_rate": _safe_rate(false_positive, false_positive + true_negative),
        "scenario_detection_rate": recall,
    }


def evaluate_audit_completeness(
    evidence: pd.DataFrame,
    known_transaction_ids: Iterable[str],
    *,
    required_columns: Sequence[str] = (
        "alert_id",
        "detector_family",
        "alert_type",
        "subject_id",
        "window_start",
        "window_end",
        "transaction_ids",
        "explanation",
    ),
) -> dict[str, Any]:
    """Measure required-field completeness and source-record traceability."""

    missing = set(required_columns) - set(evidence.columns)
    if missing:
        raise ValueError(f"evidence is missing audit columns: {sorted(missing)}")
    known = {str(value) for value in known_transaction_ids}
    complete_alerts = 0
    traceable_alerts = 0
    referenced_ids: set[str] = set()

    for row in evidence[list(required_columns)].itertuples(index=False, name=None):
        values = dict(zip(required_columns, row))
        required_present = all(
            value is not None
            and not (isinstance(value, float) and pd.isna(value))
            and bool(str(value).strip())
            for value in values.values()
        )
        if required_present:
            complete_alerts += 1
        raw_ids = values["transaction_ids"]
        parts = raw_ids.split("|") if isinstance(raw_ids, str) else raw_ids
        alert_ids = {str(item) for item in parts if str(item).strip()}
        referenced_ids.update(alert_ids)
        if alert_ids and alert_ids <= known:
            traceable_alerts += 1

    alert_count = len(evidence)
    return {
        "alerts": alert_count,
        "complete_alerts": complete_alerts,
        "traceable_alerts": traceable_alerts,
        "unique_evidence_transactions": len(referenced_ids),
        "audit_completeness_rate": _safe_rate(complete_alerts, alert_count),
        "alert_traceability_rate": _safe_rate(traceable_alerts, alert_count),
        "unresolved_transaction_ids": len(referenced_ids - known),
    }
