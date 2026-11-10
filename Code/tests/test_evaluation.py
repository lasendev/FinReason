import unittest

import pandas as pd

from finreason.evaluation import (
    evaluate_audit_completeness,
    evaluate_scenario_ids,
    evaluate_transaction_ids,
    transaction_ids_from_evidence,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.transactions = pd.DataFrame({
            "transaction_id": ["T1", "T2", "T3", "T4"],
            "split": ["validation", "validation", "validation", "test"],
            "is_laundering": [1, 0, 1, 1],
        })
        self.scenarios = pd.DataFrame({
            "scenario_id": ["S1", "S2", "S3"],
            "split": ["validation", "validation", "test"],
            "is_laundering": [1, 0, 1],
        })
        self.membership = pd.DataFrame({
            "scenario_id": ["S1", "S1", "S2", "S3"],
            "transaction_id": ["T1", "T3", "T2", "T4"],
        })

    def test_transaction_metrics_use_the_requested_split(self):
        result = evaluate_transaction_ids(
            {"T1", "T2", "T4"}, self.transactions,
            split="validation", label_column="is_laundering",
        )
        self.assertEqual(result["population"], 3)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertAlmostEqual(result["f1"], 0.5)

    def test_scenario_detection_uses_any_member_transaction(self):
        result = evaluate_scenario_ids(
            {"T3"}, self.membership, self.scenarios,
            split="validation", label_column="is_laundering",
        )
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["scenario_detection_rate"], 1.0)

    def test_evidence_parser_deduplicates_pipe_delimited_ids(self):
        evidence = pd.DataFrame({"transaction_ids": ["T1|T2", "T2|T3"]})
        self.assertEqual(transaction_ids_from_evidence(evidence), {"T1", "T2", "T3"})

    def test_audit_completeness_requires_fields_and_traceability(self):
        evidence = pd.DataFrame([{
            "alert_id": "A1", "detector_family": "GRAPH", "alert_type": "FAN_OUT",
            "subject_id": "X", "window_start": "2025-01-01T00:00:00Z",
            "window_end": "2025-01-01T01:00:00Z", "transaction_ids": "T1|T2",
            "explanation": "traceable evidence",
        }])
        result = evaluate_audit_completeness(evidence, {"T1", "T2"})
        self.assertEqual(result["audit_completeness_rate"], 1.0)
        self.assertEqual(result["alert_traceability_rate"], 1.0)
        self.assertEqual(result["unresolved_transaction_ids"], 0)


if __name__ == "__main__":
    unittest.main()
