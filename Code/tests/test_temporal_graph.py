from __future__ import annotations

import unittest

import networkx as nx
import pandas as pd

from finreason.preprocessing import COMMON_COLUMNS
from finreason.temporal_graph import (
    build_temporal_multidigraph,
    graph_summary,
    iter_outgoing_temporal_edges,
    validate_graph_traceability,
)


def fixture_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-02T00:00:00Z"]
    )
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3"],
        "timestamp": timestamps,
        "event_step": pd.Series([pd.NA] * 3, dtype="Int64"),
        "from_account": ["A", "A", "B"],
        "to_account": ["B", "B", "A"],
        "amount": [10.0, 20.0, 30.0],
        "payment_currency": ["USD"] * 3,
        "receiving_currency": ["USD"] * 3,
        "payment_format": ["ACH"] * 3,
        "transaction_type": ["TRANSFER"] * 3,
        "source": ["fixture"] * 3,
        "is_anomaly": pd.Series([0, 1, 0], dtype="Int8"),
        "is_laundering": pd.Series([0, 1, 0], dtype="Int8"),
        "scenario_id": pd.Series([pd.NA, "S1", pd.NA], dtype="string"),
        "split": ["train", "train", "test"],
    })[COMMON_COLUMNS]


class TemporalGraphTests(unittest.TestCase):
    def test_parallel_transactions_are_preserved(self) -> None:
        graph, stats = build_temporal_multidigraph([fixture_frame()], split="train")
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertEqual(graph.number_of_edges("A", "B"), 2)
        self.assertEqual(set(graph["A"]["B"]), {"T1", "T2"})
        self.assertEqual(stats["selected_rows"], 2)

    def test_direction_is_preserved(self) -> None:
        graph, _ = build_temporal_multidigraph([fixture_frame()])
        self.assertEqual(graph.number_of_edges("A", "B"), 2)
        self.assertEqual(graph.number_of_edges("B", "A"), 1)

    def test_window_is_start_inclusive_end_exclusive(self) -> None:
        graph, _ = build_temporal_multidigraph(
            [fixture_frame()], start="2025-01-01T01:00:00Z", end="2025-01-02T00:00:00Z"
        )
        self.assertEqual(list(key for *_edge, key in graph.edges(keys=True)), ["T2"])

    def test_traceability_validation(self) -> None:
        graph, _ = build_temporal_multidigraph([fixture_frame()])
        checks = validate_graph_traceability(graph)
        self.assertTrue(checks["traceability_valid"])
        self.assertEqual(checks["unique_transaction_ids"], 3)

    def test_parallel_summary_and_labels(self) -> None:
        graph, _ = build_temporal_multidigraph([fixture_frame()])
        summary = graph_summary(graph)
        self.assertEqual(summary["parallel_account_pairs"], 1)
        self.assertEqual(summary["parallel_edge_excess"], 1)
        self.assertEqual(summary["anomaly_positive_edges"], 1)

    def test_temporal_adjacency_is_sorted(self) -> None:
        frame = fixture_frame().iloc[[1, 0, 2]]
        graph, _ = build_temporal_multidigraph([frame])
        identifiers = [edge[2] for edge in iter_outgoing_temporal_edges(graph, "A")]
        self.assertEqual(identifiers, ["T1", "T2"])


if __name__ == "__main__":
    unittest.main()
