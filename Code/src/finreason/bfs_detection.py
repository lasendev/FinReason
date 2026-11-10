"""Bounded temporal BFS and explainable fan-out/fan-in detectors."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import networkx as nx
import pandas as pd


Direction = Literal["forward", "reverse"]


@dataclass(frozen=True)
class BFSAlert:
    alert_id: str
    detector: str
    pattern_type: str
    root_account: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    distinct_counterparties: int
    transaction_count: int
    total_amount: float
    counterparties: tuple[str, ...]
    transaction_ids: tuple[str, ...]
    maximum_depth: int
    explanation: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["window_start"] = self.window_start.isoformat()
        record["window_end"] = self.window_end.isoformat()
        record["counterparties"] = "|".join(self.counterparties)
        record["transaction_ids"] = "|".join(self.transaction_ids)
        return record


def _utc(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def bounded_temporal_bfs(
    graph: nx.MultiDiGraph,
    root_account: str,
    *,
    direction: Direction = "forward",
    maximum_depth: int = 2,
    start: Any | None = None,
    end: Any | None = None,
) -> dict[str, Any]:
    """Traverse breadth-first while respecting direction, time and depth."""
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be 'forward' or 'reverse'")
    if maximum_depth < 0:
        raise ValueError("maximum_depth must be non-negative")
    start_time = _utc(start) if start is not None else pd.Timestamp.min.tz_localize("UTC")
    end_time = _utc(end) if end is not None else pd.Timestamp.max.tz_localize("UTC")
    if start_time >= end_time:
        raise ValueError("start must be earlier than end")

    visited_depth: dict[str, int] = {root_account: 0}
    visited_by_depth: dict[int, list[str]] = {0: [root_account]}
    evidence_ids: list[str] = []
    queue: deque[tuple[str, int, pd.Timestamp]] = deque([
        (root_account, 0, start_time if direction == "forward" else end_time)
    ])

    while queue:
        account, depth, path_time = queue.popleft()
        if depth >= maximum_depth or account not in graph:
            continue
        if direction == "forward":
            candidates = list(graph.out_edges(account, keys=True, data=True))
            candidates.sort(key=lambda edge: (edge[3]["timestamp"], str(edge[2])))
        else:
            candidates = list(graph.in_edges(account, keys=True, data=True))
            candidates.sort(key=lambda edge: (edge[3]["timestamp"], str(edge[2])), reverse=True)

        for source, target, key, data in candidates:
            timestamp = _utc(data["timestamp"])
            if timestamp < start_time or timestamp >= end_time:
                continue
            if direction == "forward" and timestamp < path_time:
                continue
            if direction == "reverse" and timestamp > path_time:
                continue
            neighbour = target if direction == "forward" else source
            next_depth = depth + 1
            evidence_ids.append(str(key))
            if next_depth < visited_depth.get(neighbour, maximum_depth + 1):
                visited_depth[neighbour] = next_depth
                visited_by_depth.setdefault(next_depth, []).append(neighbour)
                queue.append((neighbour, next_depth, timestamp))

    return {
        "root_account": root_account,
        "direction": direction,
        "maximum_depth": maximum_depth,
        "visited_by_depth": {
            depth: sorted(accounts) for depth, accounts in sorted(visited_by_depth.items())
        },
        "visited_account_count": len(visited_depth),
        "transaction_ids": tuple(dict.fromkeys(evidence_ids)),
    }


def _account_events(
    graph: nx.MultiDiGraph,
    account: str,
    *,
    direction: Direction,
    excluded_transaction_types: set[str],
) -> list[dict[str, Any]]:
    edges = (
        graph.out_edges(account, keys=True, data=True)
        if direction == "forward"
        else graph.in_edges(account, keys=True, data=True)
    )
    events = []
    for source, target, key, data in edges:
        transaction_type = str(data.get("transaction_type") or "").upper()
        if transaction_type in excluded_transaction_types:
            continue
        events.append({
            "counterparty": target if direction == "forward" else source,
            "transaction_id": str(key),
            "timestamp": _utc(data["timestamp"]),
            "amount": float(data["amount"]),
        })
    return sorted(events, key=lambda event: (event["timestamp"], event["transaction_id"]))


def detect_fan_patterns(
    graph: nx.MultiDiGraph,
    *,
    direction: Direction,
    window_hours: float = 24.0,
    minimum_distinct_counterparties: int = 4,
    excluded_transaction_types: Iterable[str] = (),
) -> list[BFSAlert]:
    """Detect non-overlapping star bursts using depth-one BFS adjacency."""
    if minimum_distinct_counterparties < 2:
        raise ValueError("minimum_distinct_counterparties must be at least two")
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    excluded = {value.upper() for value in excluded_transaction_types}
    pattern_type = "FAN_OUT" if direction == "forward" else "FAN_IN"
    detector = "BFS" if direction == "forward" else "BFS_REVERSE"
    duration = pd.Timedelta(hours=window_hours)
    alerts: list[BFSAlert] = []

    for account in sorted(graph.nodes):
        events = _account_events(
            graph,
            account,
            direction=direction,
            excluded_transaction_types=excluded,
        )
        left = 0
        right = 0
        counterparty_counts: Counter[str] = Counter()
        while left < len(events):
            limit = events[left]["timestamp"] + duration
            while right < len(events) and events[right]["timestamp"] < limit:
                counterparty_counts[events[right]["counterparty"]] += 1
                right += 1
            if len(counterparty_counts) >= minimum_distinct_counterparties:
                selected = events[left:right]
                transaction_ids = tuple(event["transaction_id"] for event in selected)
                sorted_counterparties = tuple(sorted(counterparty_counts))
                alerts.append(BFSAlert(
                    alert_id="", detector=detector, pattern_type=pattern_type,
                    root_account=account,
                    window_start=selected[0]["timestamp"],
                    window_end=selected[-1]["timestamp"],
                    distinct_counterparties=len(sorted_counterparties),
                    transaction_count=len(selected),
                    total_amount=round(sum(event["amount"] for event in selected), 2),
                    counterparties=sorted_counterparties,
                    transaction_ids=transaction_ids,
                    maximum_depth=1,
                    explanation=(
                        f"{pattern_type.lower().replace('_', '-')} detected: {account} connected to "
                        f"{len(sorted_counterparties)} distinct counterparties through {len(selected)} "
                        f"transactions within {window_hours:g} hours."
                    ),
                ))
                left = right
                right = left
                counterparty_counts.clear()
            else:
                counterparty = events[left]["counterparty"]
                counterparty_counts[counterparty] -= 1
                if counterparty_counts[counterparty] == 0:
                    del counterparty_counts[counterparty]
                left += 1

    ordered = sorted(alerts, key=lambda alert: (alert.window_start, alert.root_account, alert.pattern_type))
    return [
        BFSAlert(**{**asdict(alert), "alert_id": f"{alert.detector}-{index:06d}"})
        for index, alert in enumerate(ordered, start=1)
    ]


def detect_fan_out(
    graph: nx.MultiDiGraph,
    *,
    window_hours: float = 24.0,
    minimum_distinct_counterparties: int = 4,
    exclude_payroll: bool = True,
) -> list[BFSAlert]:
    return detect_fan_patterns(
        graph,
        direction="forward",
        window_hours=window_hours,
        minimum_distinct_counterparties=minimum_distinct_counterparties,
        excluded_transaction_types={"PAYROLL"} if exclude_payroll else set(),
    )


def detect_fan_in(
    graph: nx.MultiDiGraph,
    *,
    window_hours: float = 24.0,
    minimum_distinct_counterparties: int = 4,
    exclude_merchant_purchases: bool = True,
) -> list[BFSAlert]:
    return detect_fan_patterns(
        graph,
        direction="reverse",
        window_hours=window_hours,
        minimum_distinct_counterparties=minimum_distinct_counterparties,
        excluded_transaction_types={"PURCHASE"} if exclude_merchant_purchases else set(),
    )


def evaluate_scenario_alerts(
    alerts: list[BFSAlert],
    expected_alerts: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    detector: str,
    split: str,
    account_prefix: str = "FR:",
) -> dict[str, Any]:
    """Match predicted alert windows to expected scenarios without using labels in detection."""
    expected = expected_alerts.loc[expected_alerts["expected_detector"] == detector].merge(
        scenarios[["scenario_id", "start_timestamp", "end_timestamp", "split"]],
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    expected = expected.loc[expected["split"] == split].copy()
    expected["start_timestamp"] = pd.to_datetime(expected["start_timestamp"], utc=True)
    expected["end_timestamp"] = pd.to_datetime(expected["end_timestamp"], utc=True)
    used_alerts: set[str] = set()
    matched_scenarios: list[str] = []

    for row in expected.sort_values(["start_timestamp", "scenario_id"]).itertuples(index=False):
        expected_account = row.target_account if detector == "BFS_REVERSE" else row.root_account
        if pd.isna(expected_account):
            expected_account = row.root_account if pd.notna(row.root_account) else row.target_account
        root_account = account_prefix + str(expected_account)
        candidates = [
            alert
            for alert in alerts
            if alert.alert_id not in used_alerts
            and alert.pattern_type == row.alert_type
            and alert.root_account == root_account
            and alert.window_start <= row.end_timestamp
            and alert.window_end >= row.start_timestamp
        ]
        if candidates:
            match = min(candidates, key=lambda alert: (abs(alert.window_start - row.start_timestamp), alert.alert_id))
            used_alerts.add(match.alert_id)
            matched_scenarios.append(str(row.scenario_id))

    true_positive = len(matched_scenarios)
    predicted = len(alerts)
    expected_count = len(expected)
    catalogue_precision = true_positive / predicted if predicted else 0.0
    catalogue_recall = true_positive / expected_count if expected_count else 0.0
    catalogue_f1 = (
        2 * catalogue_precision * catalogue_recall / (catalogue_precision + catalogue_recall)
        if catalogue_precision + catalogue_recall else 0.0
    )
    unmatched_alerts = predicted - true_positive
    return {
        "detector": detector,
        "split": split,
        "expected_scenarios": expected_count,
        "predicted_alerts": predicted,
        "matched_alerts": true_positive,
        # The expected-alert catalogue is a minimum positive set, not an
        # exhaustive negative set.  An unmatched alert therefore cannot be
        # called a false positive without joining label-aligned truth.
        "unmatched_top_level_catalog_alerts": unmatched_alerts,
        "false_positive_alerts": unmatched_alerts,
        "missed_scenarios": expected_count - true_positive,
        "catalogue_match_precision": catalogue_precision,
        "catalogue_match_recall": catalogue_recall,
        "catalogue_match_f1": catalogue_f1,
        # Backward-compatible aliases.  Callers should label these as
        # catalogue-match rates, not classifier precision/recall.
        "precision": catalogue_precision,
        "recall": catalogue_recall,
        "f1": catalogue_f1,
        "matched_scenario_ids": matched_scenarios,
        "unmatched_alert_ids": sorted({alert.alert_id for alert in alerts} - used_alerts),
    }
