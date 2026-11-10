"""Time-respecting depth-first search for layering paths and circular flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

import networkx as nx
import pandas as pd


PatternType = Literal["LAYERING_CHAIN", "CIRCULAR_FLOW"]


@dataclass(frozen=True)
class DFSAlert:
    alert_id: str
    detector: str
    pattern_type: PatternType
    root_account: str
    target_account: str
    depth: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    duration_minutes: float
    accounts: tuple[str, ...]
    transaction_ids: tuple[str, ...]
    amounts: tuple[float, ...]
    minimum_step_retention: float
    end_to_start_retention: float
    explanation: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["window_start"] = self.window_start.isoformat()
        record["window_end"] = self.window_end.isoformat()
        record["accounts"] = "|".join(self.accounts)
        record["transaction_ids"] = "|".join(self.transaction_ids)
        record["amounts"] = "|".join(f"{amount:.2f}" for amount in self.amounts)
        return record


@dataclass(frozen=True)
class _PathCandidate:
    accounts: tuple[str, ...]
    transaction_ids: tuple[str, ...]
    timestamps: tuple[pd.Timestamp, ...]
    amounts: tuple[float, ...]


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _eligible_adjacency(
    graph: nx.MultiDiGraph,
    excluded_transaction_types: Iterable[str],
) -> dict[str, list[tuple[str, str, pd.Timestamp, float]]]:
    excluded = {value.upper() for value in excluded_transaction_types}
    adjacency: dict[str, list[tuple[str, str, pd.Timestamp, float]]] = {}
    for source, target, key, data in graph.edges(keys=True, data=True):
        transaction_type = str(data.get("transaction_type") or "").upper()
        if transaction_type in excluded:
            continue
        amount = float(data["amount"])
        if amount <= 0:
            continue
        adjacency.setdefault(source, []).append((target, str(key), _utc(data["timestamp"]), amount))
    for edges in adjacency.values():
        edges.sort(key=lambda edge: (edge[2], edge[1], edge[0]))
    return adjacency


def _amount_step_valid(previous: float, current: float, minimum_retention: float, maximum_growth: float) -> bool:
    ratio = current / previous
    return minimum_retention <= ratio <= maximum_growth


def _is_contiguous_subsequence(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    if len(shorter) >= len(longer):
        return False
    width = len(shorter)
    return any(longer[index:index + width] == shorter for index in range(len(longer) - width + 1))


def _suppress_layering_subpaths(candidates: list[_PathCandidate]) -> list[_PathCandidate]:
    kept: list[_PathCandidate] = []
    exact_seen: set[tuple[str, ...]] = set()
    for candidate in sorted(
        candidates,
        key=lambda path: (-len(path.transaction_ids), path.timestamps[0], path.transaction_ids),
    ):
        if candidate.transaction_ids in exact_seen:
            continue
        if any(_is_contiguous_subsequence(candidate.transaction_ids, item.transaction_ids) for item in kept):
            continue
        kept.append(candidate)
        exact_seen.add(candidate.transaction_ids)
    return sorted(kept, key=lambda path: (path.timestamps[0], path.accounts[0], path.transaction_ids))


def _search_temporal_paths(
    graph: nx.MultiDiGraph,
    *,
    pattern_type: PatternType,
    minimum_depth: int,
    maximum_depth: int,
    window_hours: float,
    minimum_amount_retention: float,
    maximum_amount_growth: float,
    excluded_transaction_types: Iterable[str],
    maximum_candidates: int,
) -> list[_PathCandidate]:
    if minimum_depth < 2 or maximum_depth < minimum_depth:
        raise ValueError("DFS depths must satisfy 2 <= minimum_depth <= maximum_depth")
    if window_hours <= 0:
        raise ValueError("window_hours must be positive")
    if not 0 < minimum_amount_retention <= 1:
        raise ValueError("minimum_amount_retention must be in (0, 1]")
    if maximum_amount_growth < 1:
        raise ValueError("maximum_amount_growth must be at least one")

    adjacency = _eligible_adjacency(graph, excluded_transaction_types)
    duration = pd.Timedelta(hours=window_hours)
    candidates: list[_PathCandidate] = []

    def append_candidate(
        accounts: tuple[str, ...],
        transaction_ids: tuple[str, ...],
        timestamps: tuple[pd.Timestamp, ...],
        amounts: tuple[float, ...],
    ) -> None:
        candidates.append(_PathCandidate(accounts, transaction_ids, timestamps, amounts))
        if len(candidates) > maximum_candidates:
            raise RuntimeError(f"DFS candidate safeguard exceeded ({maximum_candidates})")

    def explore(
        start_account: str,
        current_account: str,
        accounts: tuple[str, ...],
        transaction_ids: tuple[str, ...],
        timestamps: tuple[pd.Timestamp, ...],
        amounts: tuple[float, ...],
    ) -> None:
        depth = len(transaction_ids)
        if depth >= maximum_depth:
            if pattern_type == "LAYERING_CHAIN" and depth >= minimum_depth:
                append_candidate(accounts, transaction_ids, timestamps, amounts)
            return

        extensions = []
        cycle_closure_available = False
        if pattern_type == "LAYERING_CHAIN":
            for target, transaction_id, timestamp, amount in adjacency.get(current_account, []):
                if target != start_account or transaction_id in transaction_ids:
                    continue
                combined = sorted(
                    [*zip(timestamps, amounts), (timestamp, amount)], key=lambda item: item[0]
                )
                combined_times = [item[0] for item in combined]
                combined_amounts = [item[1] for item in combined]
                if combined_times[-1] - combined_times[0] > duration:
                    continue
                if len(set(combined_times)) != len(combined_times):
                    continue
                if all(
                    _amount_step_valid(previous, current, minimum_amount_retention, maximum_amount_growth)
                    for previous, current in zip(combined_amounts, combined_amounts[1:])
                ) and combined_amounts[-1] / combined_amounts[0] >= minimum_amount_retention:
                    cycle_closure_available = True
                    break
        for target, transaction_id, timestamp, amount in adjacency.get(current_account, []):
            if timestamp <= timestamps[-1] or timestamp - timestamps[0] > duration:
                continue
            if not _amount_step_valid(
                amounts[-1], amount, minimum_amount_retention, maximum_amount_growth
            ):
                continue
            next_depth = depth + 1
            if target == start_account:
                cycle_closure_available = True
                if (
                    pattern_type == "CIRCULAR_FLOW"
                    and next_depth >= minimum_depth
                    and amount / amounts[0] >= minimum_amount_retention
                ):
                    append_candidate(
                        accounts + (target,),
                        transaction_ids + (transaction_id,),
                        timestamps + (timestamp,),
                        amounts + (amount,),
                    )
                continue
            if target in accounts:
                cycle_closure_available = True
                continue
            extensions.append((target, transaction_id, timestamp, amount))

        for target, transaction_id, timestamp, amount in extensions:
            explore(
                start_account,
                target,
                accounts + (target,),
                transaction_ids + (transaction_id,),
                timestamps + (timestamp,),
                amounts + (amount,),
            )
        if (
            pattern_type == "LAYERING_CHAIN"
            and not extensions
            and not cycle_closure_available
            and depth >= minimum_depth
        ):
            append_candidate(accounts, transaction_ids, timestamps, amounts)

    for start_account in sorted(adjacency):
        for target, transaction_id, timestamp, amount in adjacency[start_account]:
            explore(
                start_account,
                target,
                (start_account, target),
                (transaction_id,),
                (timestamp,),
                (amount,),
            )

    if pattern_type == "LAYERING_CHAIN":
        return _suppress_layering_subpaths(candidates)
    unique = {candidate.transaction_ids: candidate for candidate in candidates}
    return sorted(unique.values(), key=lambda path: (path.timestamps[0], path.accounts[0], path.transaction_ids))


def _to_alerts(candidates: list[_PathCandidate], pattern_type: PatternType) -> list[DFSAlert]:
    alerts = []
    for index, candidate in enumerate(candidates, start=1):
        step_retention = tuple(
            current / previous for previous, current in zip(candidate.amounts, candidate.amounts[1:])
        )
        depth = len(candidate.transaction_ids)
        root = candidate.accounts[0]
        target = candidate.accounts[-1]
        duration_minutes = (candidate.timestamps[-1] - candidate.timestamps[0]).total_seconds() / 60
        readable = "layering chain" if pattern_type == "LAYERING_CHAIN" else "circular flow"
        alerts.append(DFSAlert(
            alert_id=f"DFS-{index:06d}",
            detector="DFS",
            pattern_type=pattern_type,
            root_account=root,
            target_account=target,
            depth=depth,
            window_start=candidate.timestamps[0],
            window_end=candidate.timestamps[-1],
            duration_minutes=duration_minutes,
            accounts=candidate.accounts,
            transaction_ids=candidate.transaction_ids,
            amounts=candidate.amounts,
            minimum_step_retention=min(step_retention),
            end_to_start_retention=candidate.amounts[-1] / candidate.amounts[0],
            explanation=(
                f"{readable} detected from {root} to {target}: {depth} time-increasing edges "
                f"over {duration_minutes:.1f} minutes; minimum step retention "
                f"{min(step_retention):.3f}."
            ),
        ))
    return alerts


def detect_layering_paths(
    graph: nx.MultiDiGraph,
    *,
    minimum_depth: int = 4,
    maximum_depth: int = 6,
    window_hours: float = 24.0,
    minimum_amount_retention: float = 0.95,
    maximum_amount_growth: float = 1.05,
    exclude_invoice_payments: bool = True,
    maximum_candidates: int = 50_000,
) -> list[DFSAlert]:
    candidates = _search_temporal_paths(
        graph,
        pattern_type="LAYERING_CHAIN",
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        window_hours=window_hours,
        minimum_amount_retention=minimum_amount_retention,
        maximum_amount_growth=maximum_amount_growth,
        excluded_transaction_types={"INVOICE_PAYMENT"} if exclude_invoice_payments else set(),
        maximum_candidates=maximum_candidates,
    )
    return _to_alerts(candidates, "LAYERING_CHAIN")


def detect_circular_flows(
    graph: nx.MultiDiGraph,
    *,
    minimum_depth: int = 3,
    maximum_depth: int = 6,
    window_hours: float = 24.0,
    minimum_amount_retention: float = 0.95,
    maximum_amount_growth: float = 1.05,
    exclude_invoice_payments: bool = True,
    maximum_candidates: int = 50_000,
) -> list[DFSAlert]:
    candidates = _search_temporal_paths(
        graph,
        pattern_type="CIRCULAR_FLOW",
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        window_hours=window_hours,
        minimum_amount_retention=minimum_amount_retention,
        maximum_amount_growth=maximum_amount_growth,
        excluded_transaction_types={"INVOICE_PAYMENT"} if exclude_invoice_payments else set(),
        maximum_candidates=maximum_candidates,
    )
    return _to_alerts(candidates, "CIRCULAR_FLOW")


def evaluate_dfs_scenarios(
    alerts: list[DFSAlert],
    expected_alerts: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    alert_type: PatternType,
    split: str,
    account_prefix: str = "FR:",
) -> dict[str, Any]:
    expected = expected_alerts.loc[
        expected_alerts["expected_detector"].eq("DFS")
        & expected_alerts["alert_type"].eq(alert_type)
    ].merge(
        scenarios[["scenario_id", "start_timestamp", "end_timestamp", "split"]],
        on="scenario_id",
        how="left",
        validate="many_to_one",
    )
    expected = expected.loc[expected["split"].eq(split)].copy()
    expected["start_timestamp"] = pd.to_datetime(expected["start_timestamp"], utc=True)
    expected["end_timestamp"] = pd.to_datetime(expected["end_timestamp"], utc=True)
    used_alerts: set[str] = set()
    matched_scenarios: list[str] = []

    for row in expected.sort_values(["start_timestamp", "scenario_id"]).itertuples(index=False):
        root = account_prefix + str(row.root_account)
        target = account_prefix + str(row.target_account)
        minimum_depth = int(row.minimum_depth)
        maximum_depth = int(row.maximum_depth)
        candidates = [
            alert
            for alert in alerts
            if alert.alert_id not in used_alerts
            and alert.pattern_type == alert_type
            and alert.root_account == root
            and alert.target_account == target
            and minimum_depth <= alert.depth <= maximum_depth
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
        "detector": "DFS",
        "alert_type": alert_type,
        "split": split,
        "expected_scenarios": expected_count,
        "predicted_alerts": predicted,
        "matched_alerts": true_positive,
        "unmatched_top_level_catalog_alerts": unmatched_alerts,
        "false_positive_alerts": unmatched_alerts,
        "missed_scenarios": expected_count - true_positive,
        "catalogue_match_precision": catalogue_precision,
        "catalogue_match_recall": catalogue_recall,
        "catalogue_match_f1": catalogue_f1,
        "precision": catalogue_precision,
        "recall": catalogue_recall,
        "f1": catalogue_f1,
        "matched_scenario_ids": matched_scenarios,
        "unmatched_alert_ids": sorted({alert.alert_id for alert in alerts} - used_alerts),
    }
