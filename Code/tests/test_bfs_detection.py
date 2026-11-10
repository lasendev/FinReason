from __future__ import annotations

import unittest

import pandas as pd

from finreason.bfs_detection import (
    bounded_temporal_bfs,
    detect_fan_in,
    detect_fan_out,
    evaluate_scenario_alerts,
)
from finreason.preprocessing import COMMON_COLUMNS
from finreason.temporal_graph import build_temporal_multidigraph


def make_frame(edges: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    rows = []
    for transaction_id, timestamp, source, target, transaction_type in edges:
        rows.append({
            "transaction_id": transaction_id,
            "timestamp": pd.Timestamp(timestamp),
            "event_step": pd.NA,
            "from_account": source,
            "to_account": target,
            "amount": 100.0,
            "payment_currency": "USD",
            "receiving_currency": "USD",
            "payment_format": "ACH",
            "transaction_type": transaction_type,
            "source": "fixture",
            "is_anomaly": 0,
            "is_laundering": 0,
            "scenario_id": pd.NA,
            "split": "train",
        })
    frame = pd.DataFrame(rows)[COMMON_COLUMNS]
    frame["event_step"] = frame["event_step"].astype("Int64")
    frame["is_anomaly"] = frame["is_anomaly"].astype("Int8")
    frame["is_laundering"] = frame["is_laundering"].astype("Int8")
    frame["scenario_id"] = frame["scenario_id"].astype("string")
    return frame


class BFSDetectionTests(unittest.TestCase):
    def test_forward_bfs_respects_depth_and_time(self) -> None:
        frame = make_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", "B", "TRANSFER"),
            ("T2", "2025-01-01T01:00:00Z", "B", "C", "TRANSFER"),
            ("T3", "2025-01-01T02:00:00Z", "C", "D", "TRANSFER"),
        ])
        graph, _ = build_temporal_multidigraph([frame])
        result = bounded_temporal_bfs(
            graph, "A", maximum_depth=2,
            start="2025-01-01T00:00:00Z", end="2025-01-02T00:00:00Z",
        )
        self.assertEqual(result["visited_by_depth"], {0: ["A"], 1: ["B"], 2: ["C"]})
        self.assertNotIn("T3", result["transaction_ids"])

    def test_reverse_bfs_walks_incoming_edges_backwards(self) -> None:
        frame = make_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", "B", "TRANSFER"),
            ("T2", "2025-01-01T01:00:00Z", "B", "C", "TRANSFER"),
        ])
        graph, _ = build_temporal_multidigraph([frame])
        result = bounded_temporal_bfs(graph, "C", direction="reverse", maximum_depth=2)
        self.assertEqual(result["visited_by_depth"], {0: ["C"], 1: ["B"], 2: ["A"]})

    def test_fan_out_requires_distinct_counterparties(self) -> None:
        edges = [
            (f"T{i}", f"2025-01-01T0{i}:00:00Z", "ROOT", target, "TRANSFER")
            for i, target in enumerate(["B", "B", "C", "D", "E"])
        ]
        graph, _ = build_temporal_multidigraph([make_frame(edges)])
        alerts = detect_fan_out(graph, minimum_distinct_counterparties=4)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].distinct_counterparties, 4)
        self.assertEqual(alerts[0].transaction_count, 5)

    def test_payroll_fan_out_is_excluded(self) -> None:
        edges = [
            (f"T{i}", f"2025-01-01T0{i}:00:00Z", "PAYROLL", f"E{i}", "PAYROLL")
            for i in range(5)
        ]
        graph, _ = build_temporal_multidigraph([make_frame(edges)])
        self.assertEqual(detect_fan_out(graph), [])
        self.assertEqual(len(detect_fan_out(graph, exclude_payroll=False)), 1)

    def test_merchant_fan_in_is_excluded(self) -> None:
        edges = [
            (f"T{i}", f"2025-01-01T0{i}:00:00Z", f"C{i}", "MERCHANT", "PURCHASE")
            for i in range(5)
        ]
        graph, _ = build_temporal_multidigraph([make_frame(edges)])
        self.assertEqual(detect_fan_in(graph), [])
        self.assertEqual(len(detect_fan_in(graph, exclude_merchant_purchases=False)), 1)

    def test_transactions_outside_window_do_not_combine(self) -> None:
        edges = [
            (f"T{i}", f"2025-01-{1 + i:02d}T00:00:00Z", "ROOT", f"C{i}", "TRANSFER")
            for i in range(4)
        ]
        graph, _ = build_temporal_multidigraph([make_frame(edges)])
        self.assertEqual(detect_fan_out(graph, window_hours=24), [])

    def test_reverse_evaluation_uses_target_account(self) -> None:
        edges = [
            (f"T{i}", f"2025-01-01T0{i}:00:00Z", f"C{i}", "TARGET", "TRANSFER")
            for i in range(4)
        ]
        graph, _ = build_temporal_multidigraph([make_frame(edges)])
        alerts = detect_fan_in(graph)
        expected = pd.DataFrame([{
            "scenario_id": "S1", "expected_detector": "BFS_REVERSE", "alert_type": "FAN_IN",
            "root_account": pd.NA, "target_account": "TARGET",
        }])
        scenarios = pd.DataFrame([{
            "scenario_id": "S1", "start_timestamp": "2025-01-01T00:00:00Z",
            "end_timestamp": "2025-01-01T04:00:00Z", "split": "train",
        }])
        metrics = evaluate_scenario_alerts(
            alerts, expected, scenarios, detector="BFS_REVERSE", split="train", account_prefix=""
        )
        self.assertEqual(metrics["matched_alerts"], 1)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["unmatched_top_level_catalog_alerts"], 0)
        self.assertEqual(metrics["catalogue_match_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
