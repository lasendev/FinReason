from __future__ import annotations

import unittest

import pandas as pd

from finreason.dfs_detection import detect_circular_flows, detect_layering_paths
from finreason.preprocessing import COMMON_COLUMNS
from finreason.temporal_graph import build_temporal_multidigraph


def make_graph(edges: list[tuple[str, str, str, str, float, str]]):
    rows = []
    for transaction_id, timestamp, source, target, amount, transaction_type in edges:
        rows.append({
            "transaction_id": transaction_id, "timestamp": pd.Timestamp(timestamp),
            "event_step": pd.NA, "from_account": source, "to_account": target,
            "amount": amount, "payment_currency": "USD", "receiving_currency": "USD",
            "payment_format": "ACH", "transaction_type": transaction_type,
            "source": "fixture", "is_anomaly": 0, "is_laundering": 0,
            "scenario_id": pd.NA, "split": "train",
        })
    frame = pd.DataFrame(rows)[COMMON_COLUMNS]
    frame["event_step"] = frame["event_step"].astype("Int64")
    frame["is_anomaly"] = frame["is_anomaly"].astype("Int8")
    frame["is_laundering"] = frame["is_laundering"].astype("Int8")
    frame["scenario_id"] = frame["scenario_id"].astype("string")
    return build_temporal_multidigraph([frame])[0]


class DFSDetectionTests(unittest.TestCase):
    def test_layering_path_is_detected(self) -> None:
        graph = make_graph([
            ("T1", "2025-01-01T00:00:00Z", "A", "B", 100.0, "TRANSFER"),
            ("T2", "2025-01-01T01:00:00Z", "B", "C", 99.0, "TRANSFER"),
            ("T3", "2025-01-01T02:00:00Z", "C", "D", 98.0, "TRANSFER"),
            ("T4", "2025-01-01T03:00:00Z", "D", "E", 97.0, "TRANSFER"),
        ])
        alerts = detect_layering_paths(graph)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].accounts, ("A", "B", "C", "D", "E"))
        self.assertEqual(alerts[0].depth, 4)

    def test_non_increasing_time_is_rejected(self) -> None:
        graph = make_graph([
            ("T1", "2025-01-01T00:00:00Z", "A", "B", 100.0, "TRANSFER"),
            ("T2", "2024-12-31T23:00:00Z", "B", "C", 99.0, "TRANSFER"),
            ("T3", "2025-01-01T02:00:00Z", "C", "D", 98.0, "TRANSFER"),
            ("T4", "2025-01-01T03:00:00Z", "D", "E", 97.0, "TRANSFER"),
        ])
        self.assertEqual(detect_layering_paths(graph), [])

    def test_amount_retention_is_required(self) -> None:
        graph = make_graph([
            ("T1", "2025-01-01T00:00:00Z", "A", "B", 100.0, "TRANSFER"),
            ("T2", "2025-01-01T01:00:00Z", "B", "C", 50.0, "TRANSFER"),
            ("T3", "2025-01-01T02:00:00Z", "C", "D", 49.0, "TRANSFER"),
            ("T4", "2025-01-01T03:00:00Z", "D", "E", 48.0, "TRANSFER"),
        ])
        self.assertEqual(detect_layering_paths(graph), [])

    def test_cycle_is_detected_and_not_reported_as_layering(self) -> None:
        graph = make_graph([
            ("T1", "2025-01-01T00:00:00Z", "A", "B", 100.0, "TRANSFER"),
            ("T2", "2025-01-01T01:00:00Z", "B", "C", 99.0, "TRANSFER"),
            ("T3", "2025-01-01T02:00:00Z", "C", "A", 98.0, "TRANSFER"),
        ])
        cycles = detect_circular_flows(graph)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0].accounts, ("A", "B", "C", "A"))
        self.assertEqual(detect_layering_paths(graph, minimum_depth=2), [])

    def test_layering_subpaths_are_suppressed(self) -> None:
        graph = make_graph([
            (f"T{i}", f"2025-01-01T0{i}:00:00Z", source, target, 101.0 - i, "TRANSFER")
            for i, (source, target) in enumerate(
                [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F")], start=1
            )
        ])
        alerts = detect_layering_paths(graph)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].depth, 5)

    def test_invoice_supply_chain_is_contextually_excluded(self) -> None:
        graph = make_graph([
            (f"T{i}", f"2025-01-0{i}T00:00:00Z", source, target, 100.0, "INVOICE_PAYMENT")
            for i, (source, target) in enumerate(
                [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")], start=1
            )
        ])
        self.assertEqual(detect_layering_paths(graph, window_hours=120), [])
        self.assertEqual(
            len(detect_layering_paths(graph, window_hours=120, exclude_invoice_payments=False)), 1
        )


if __name__ == "__main__":
    unittest.main()
