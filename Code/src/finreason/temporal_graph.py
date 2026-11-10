"""Directed temporal multigraph construction for traceable transaction search."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import networkx as nx
import pandas as pd

from .preprocessing import COMMON_COLUMNS, clean_harmonized


EDGE_ATTRIBUTES = (
    "transaction_id",
    "timestamp",
    "amount",
    "payment_currency",
    "receiving_currency",
    "payment_format",
    "transaction_type",
    "source",
    "is_anomaly",
    "is_laundering",
    "scenario_id",
    "split",
)


def _python_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalise_boundary(value: Any | None) -> pd.Timestamp | None:
    if value is None:
        return None
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    return result


def add_transaction_frame(
    graph: nx.MultiDiGraph,
    frame: pd.DataFrame,
    *,
    start: Any | None = None,
    end: Any | None = None,
    split: str | None = None,
) -> dict[str, int]:
    """Clean and add selected transaction rows to a directed multigraph."""
    missing = set(COMMON_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Harmonized frame is missing columns: {sorted(missing)}")
    clean, diagnostics = clean_harmonized(frame)
    start_time = _normalise_boundary(start)
    end_time = _normalise_boundary(end)
    selected = clean
    if start_time is not None:
        selected = selected.loc[selected["timestamp"] >= start_time]
    if end_time is not None:
        selected = selected.loc[selected["timestamp"] < end_time]
    if split is not None:
        selected = selected.loc[selected["split"] == split]

    for row in selected.itertuples(index=False):
        values = row._asdict()
        transaction_id = str(values["transaction_id"])
        attributes = {
            attribute: _python_value(values[attribute])
            for attribute in EDGE_ATTRIBUTES
        }
        graph.add_edge(
            str(values["from_account"]),
            str(values["to_account"]),
            key=transaction_id,
            **attributes,
        )
    return {
        "input_rows": diagnostics["input_rows"],
        "clean_rows": len(clean),
        "dropped_rows": diagnostics["dropped_rows"],
        "selected_rows": len(selected),
    }


def build_temporal_multidigraph(
    chunks: Iterable[pd.DataFrame],
    *,
    start: Any | None = None,
    end: Any | None = None,
    split: str | None = None,
    name: str = "temporal_transactions",
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    """Stream harmonized chunks into one bounded NetworkX MultiDiGraph."""
    graph = nx.MultiDiGraph(name=name)
    graph.graph.update({
        "window_start": _normalise_boundary(start),
        "window_end": _normalise_boundary(end),
        "split": split,
        "directed": True,
        "parallel_edges_preserved": True,
    })
    totals: Counter[str] = Counter()
    chunks_scanned = 0
    for frame in chunks:
        totals.update(add_transaction_frame(graph, frame, start=start, end=end, split=split))
        chunks_scanned += 1
    build_stats = {
        "chunks_scanned": chunks_scanned,
        **{key: int(value) for key, value in totals.items()},
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
    }
    if build_stats.get("selected_rows", 0) != graph.number_of_edges():
        raise RuntimeError("Selected transaction rows do not match graph edge count")
    return graph, build_stats


def graph_summary(graph: nx.MultiDiGraph) -> dict[str, Any]:
    timestamps: list[pd.Timestamp] = []
    anomaly_positive = 0
    laundering_positive = 0
    source_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    self_loops = 0
    for source, target, _key, data in graph.edges(keys=True, data=True):
        timestamp = data.get("timestamp")
        if timestamp is not None:
            timestamps.append(pd.Timestamp(timestamp))
        anomaly_positive += int(data.get("is_anomaly") == 1)
        laundering_positive += int(data.get("is_laundering") == 1)
        source_counts[str(data.get("source"))] += 1
        pair_counts[(source, target)] += 1
        self_loops += int(source == target)
    parallel_counts = [count for count in pair_counts.values() if count > 1]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "timestamp_min": min(timestamps).isoformat() if timestamps else None,
        "timestamp_max": max(timestamps).isoformat() if timestamps else None,
        "self_loop_edges": self_loops,
        "parallel_account_pairs": len(parallel_counts),
        "parallel_edge_excess": sum(count - 1 for count in parallel_counts),
        "maximum_parallel_edges": max(parallel_counts, default=1 if graph.number_of_edges() else 0),
        "anomaly_positive_edges": anomaly_positive,
        "laundering_positive_edges": laundering_positive,
        "source_edge_counts": dict(source_counts),
    }


def validate_graph_traceability(graph: nx.MultiDiGraph) -> dict[str, Any]:
    transaction_ids: set[str] = set()
    missing_attributes = 0
    key_mismatches = 0
    invalid_timestamps = 0
    duplicate_transaction_ids = 0
    for _source, _target, key, data in graph.edges(keys=True, data=True):
        missing_attributes += int(any(attribute not in data for attribute in EDGE_ATTRIBUTES))
        transaction_id = str(data.get("transaction_id"))
        key_mismatches += int(str(key) != transaction_id)
        invalid_timestamps += int(pd.isna(data.get("timestamp")))
        duplicate_transaction_ids += int(transaction_id in transaction_ids)
        transaction_ids.add(transaction_id)
    checks = {
        "edge_count": graph.number_of_edges(),
        "unique_transaction_ids": len(transaction_ids),
        "missing_attribute_edges": missing_attributes,
        "edge_key_mismatches": key_mismatches,
        "invalid_timestamp_edges": invalid_timestamps,
        "duplicate_transaction_ids": duplicate_transaction_ids,
    }
    checks["traceability_valid"] = all(
        checks[key] == 0
        for key in (
            "missing_attribute_edges",
            "edge_key_mismatches",
            "invalid_timestamp_edges",
            "duplicate_transaction_ids",
        )
    ) and checks["edge_count"] == checks["unique_transaction_ids"]
    return checks


def parallel_edge_examples(graph: nx.MultiDiGraph, *, pair_limit: int = 3, edge_limit: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs_seen = 0
    for source, targets in graph.adj.items():
        for target, keyed_edges in targets.items():
            if len(keyed_edges) <= 1:
                continue
            pairs_seen += 1
            for transaction_id, data in sorted(
                keyed_edges.items(), key=lambda item: (item[1]["timestamp"], str(item[0]))
            )[:edge_limit]:
                rows.append({
                    "from_account": source,
                    "to_account": target,
                    "transaction_id": transaction_id,
                    "timestamp": data["timestamp"],
                    "amount": data["amount"],
                })
            if pairs_seen >= pair_limit:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def iter_outgoing_temporal_edges(
    graph: nx.MultiDiGraph,
    account: str,
    *,
    start: Any | None = None,
    end: Any | None = None,
) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
    """Yield outgoing edges in deterministic chronological order."""
    if account not in graph:
        return
    start_time = _normalise_boundary(start)
    end_time = _normalise_boundary(end)
    edges = []
    for source, target, key, data in graph.out_edges(account, keys=True, data=True):
        timestamp = pd.Timestamp(data["timestamp"])
        if start_time is not None and timestamp < start_time:
            continue
        if end_time is not None and timestamp >= end_time:
            continue
        edges.append((source, target, str(key), data))
    yield from sorted(edges, key=lambda edge: (edge[3]["timestamp"], edge[2]))


def graph_report_path(root: Path) -> Path:
    """Small helper retained for consistent output naming."""
    return root / "section4_temporal_graph_report.json"

