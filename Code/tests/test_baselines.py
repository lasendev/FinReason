import unittest

import pandas as pd

from finreason.baselines import (
    AccountCurrencyZScoreBaseline,
    AccountDailyFrequencyIQRBaseline,
    AccountRollingFrequencyBaseline,
    CurrencyLogAmountZScoreBaseline,
    evaluate_scenario_evidence,
    evaluate_transaction_evidence,
)


def transaction_frame(rows):
    return pd.DataFrame(rows, columns=[
        "transaction_id", "timestamp", "from_account", "amount", "payment_currency", "split",
        "is_anomaly",
    ])


class StatisticalBaselineTests(unittest.TestCase):
    @staticmethod
    def rolling_frame(rows):
        return pd.DataFrame(rows, columns=[
            "transaction_id", "timestamp", "from_account", "amount",
            "payment_currency", "split", "is_anomaly", "transaction_type",
        ])

    def test_zscore_fit_uses_only_training_rows(self):
        frame = transaction_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", 10, "GBP", "train", 0),
            ("T2", "2025-01-02T00:00:00Z", "A", 20, "GBP", "train", 0),
            ("T3", "2025-02-01T00:00:00Z", "A", 1_000_000, "GBP", "validation", 1),
        ])
        model = AccountCurrencyZScoreBaseline(minimum_history=2).fit(frame)
        self.assertEqual(model.fit_metadata["fit_rows"], 2)
        self.assertEqual(model.global_mean, 15)
        self.assertEqual(model.fit_metadata["label_columns_used"], [])

    def test_zscore_uses_account_currency_then_currency_fallback(self):
        frame = transaction_frame([
            (f"T{i}", f"2025-01-0{i}T00:00:00Z", "A", amount, "GBP", "train", 0)
            for i, amount in enumerate([10, 11, 12, 13, 14], start=1)
        ] + [
            ("U1", "2025-02-01T00:00:00Z", "B", 100, "GBP", "validation", 1),
        ])
        model = AccountCurrencyZScoreBaseline(z_threshold=3, minimum_history=5).fit(frame)
        scored = model.score_transactions(frame)
        self.assertEqual(scored.loc[scored.transaction_id.eq("T1"), "statistics_level"].iloc[0], "ACCOUNT_CURRENCY")
        self.assertEqual(scored.loc[scored.transaction_id.eq("U1"), "statistics_level"].iloc[0], "CURRENCY")

    def test_zscore_prediction_has_exact_transaction_provenance(self):
        frame = transaction_frame([
            (f"T{i}", f"2025-01-{i:02d}T00:00:00Z", "A", amount, "GBP", "train", 0)
            for i, amount in enumerate([10, 11, 12, 13, 14], start=1)
        ] + [("V1", "2025-02-01T00:00:00Z", "A", 100, "GBP", "validation", 1)])
        model = AccountCurrencyZScoreBaseline(z_threshold=3, minimum_history=5).fit(frame)
        alerts = model.predict(frame)
        alert = next(item for item in alerts if item.transaction_ids == ("V1",))
        self.assertEqual(alert.fit_split, "train")
        self.assertIn("standard deviations", alert.explanation)

    def test_iqr_frequency_detects_only_above_upper_fence(self):
        rows = []
        transaction_id = 0
        for day, count in enumerate([1, 2, 2, 3, 4], start=1):
            for minute in range(count):
                transaction_id += 1
                rows.append((f"T{transaction_id}", f"2025-01-{day:02d}T00:{minute:02d}:00Z", "A", 10, "GBP", "train", 0))
        for minute in range(10):
            transaction_id += 1
            rows.append((f"V{transaction_id}", f"2025-02-01T00:{minute:02d}:00Z", "A", 10, "GBP", "validation", 1))
        frame = transaction_frame(rows)
        model = AccountDailyFrequencyIQRBaseline(minimum_active_days=5).fit(frame)
        alerts = model.predict(frame)
        validation_alert = next(item for item in alerts if item.split == "validation")
        self.assertEqual(len(validation_alert.transaction_ids), 10)
        self.assertGreater(validation_alert.observed_value, validation_alert.threshold)

    def test_frequency_fit_excludes_validation_day(self):
        frame = transaction_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", 10, "GBP", "train", 0),
            ("T2", "2025-01-02T00:00:00Z", "A", 10, "GBP", "train", 0),
            ("V1", "2025-02-01T00:00:00Z", "A", 10, "GBP", "validation", 0),
            ("V2", "2025-02-01T00:01:00Z", "A", 10, "GBP", "validation", 1),
        ])
        model = AccountDailyFrequencyIQRBaseline(minimum_active_days=2).fit(frame)
        self.assertEqual(model.fit_metadata["fit_account_days"], 2)
        self.assertEqual(model.fit_metadata["label_columns_used"], [])

    def test_rolling_frequency_crosses_midnight_with_causal_evidence(self):
        frame = self.rolling_frame([
            ("T1", "2025-01-01T23:00:00Z", "A", 10, "GBP", "train", 0, "TRANSFER"),
            ("T2", "2025-01-02T01:00:00Z", "A", 10, "GBP", "train", 1, "TRANSFER"),
            ("T3", "2025-01-02T09:00:00Z", "A", 10, "GBP", "train", 1, "TRANSFER"),
            ("T4", "2025-01-02T20:00:00Z", "A", 10, "GBP", "train", 1, "TRANSFER"),
        ])
        model = AccountRollingFrequencyBaseline(window_hours=48, minimum_count=4).fit(frame)
        alerts = model.predict(frame)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].transaction_ids, ("T1", "T2", "T3", "T4"))
        self.assertLessEqual(alerts[0].window_end, pd.Timestamp("2025-01-02T20:00:00Z"))

    def test_rolling_frequency_rejects_sparse_and_context_transactions(self):
        frame = self.rolling_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", 10, "GBP", "train", 0, "TRANSFER"),
            ("T2", "2025-01-02T00:00:00Z", "A", 10, "GBP", "train", 0, "TRANSFER"),
            ("T3", "2025-01-03T01:00:00Z", "A", 10, "GBP", "train", 0, "TRANSFER"),
            ("P1", "2025-01-03T01:01:00Z", "A", 10, "GBP", "train", 0, "PAYROLL"),
        ])
        model = AccountRollingFrequencyBaseline(window_hours=48, minimum_count=4).fit(
            pd.concat([frame, self.rolling_frame([
                ("F1", "2025-01-04T00:00:00Z", "B", 10, "GBP", "train", 0, "TRANSFER"),
                ("F2", "2025-01-04T01:00:00Z", "B", 10, "GBP", "train", 0, "TRANSFER"),
                ("F3", "2025-01-04T02:00:00Z", "B", 10, "GBP", "train", 0, "TRANSFER"),
                ("F4", "2025-01-04T03:00:00Z", "B", 10, "GBP", "train", 0, "TRANSFER"),
            ])], ignore_index=True)
        )
        alerts = model.predict(frame)
        self.assertEqual(alerts, [])

    def test_rolling_frequency_resets_state_at_split_boundary(self):
        frame = self.rolling_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", 10, "GBP", "train", 0, "TRANSFER"),
            ("T2", "2025-01-01T01:00:00Z", "A", 10, "GBP", "train", 0, "TRANSFER"),
            ("T3", "2025-01-01T02:00:00Z", "A", 10, "GBP", "train", 0, "TRANSFER"),
            ("V1", "2025-01-01T03:00:00Z", "A", 10, "GBP", "validation", 1, "TRANSFER"),
        ])
        fit_frame = self.rolling_frame([
            (f"F{i}", f"2024-12-01T0{i}:00:00Z", "B", 10, "GBP", "train", 0, "TRANSFER")
            for i in range(4)
        ])
        model = AccountRollingFrequencyBaseline(window_hours=48, minimum_count=4).fit(fit_frame)
        self.assertEqual(model.predict(frame), [])

    def test_log_amount_model_is_upper_tail_and_label_free(self):
        frame = transaction_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", 10, "GBP", "train", 0),
            ("T2", "2025-01-02T00:00:00Z", "A", 11, "GBP", "train", 0),
            ("T3", "2025-01-03T00:00:00Z", "A", 12, "GBP", "train", 0),
            ("T4", "2025-01-04T00:00:00Z", "A", 13, "GBP", "train", 0),
            ("V1", "2025-02-01T00:00:00Z", "A", 1000, "GBP", "validation", 1),
        ])
        model = CurrencyLogAmountZScoreBaseline(
            z_threshold=2, minimum_currency_history=4
        ).fit(frame)
        alert_ids = {item.transaction_ids for item in model.predict(frame)}
        self.assertIn(("V1",), alert_ids)
        self.assertEqual(model.fit_metadata["amount_transform"], "log1p")
        self.assertEqual(model.fit_metadata["label_columns_used"], [])

    def test_evaluation_contract_counts_transaction_evidence(self):
        frame = transaction_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", 10, "GBP", "train", 0),
            ("T2", "2025-01-01T01:00:00Z", "A", 100, "GBP", "train", 1),
        ])
        model = AccountCurrencyZScoreBaseline(z_threshold=0.5, minimum_history=2).fit(frame)
        alerts = model.predict(frame)
        metrics = evaluate_transaction_evidence(alerts, frame, split="train", label_column="is_anomaly")
        self.assertEqual(metrics["known_transactions"], 2)
        self.assertEqual(metrics["predicted_positive_transactions"], 2)
        self.assertEqual(metrics["true_positive_transactions"], 1)

    def test_scenario_evaluation_is_separate_from_transaction_evaluation(self):
        frame = transaction_frame([
            ("T1", "2025-01-01T00:00:00Z", "A", 10, "GBP", "train", 0),
            ("T2", "2025-01-01T01:00:00Z", "A", 100, "GBP", "train", 1),
        ])
        model = AccountCurrencyZScoreBaseline(z_threshold=0.5, minimum_history=2).fit(frame)
        alerts = model.predict(frame)
        membership = pd.DataFrame({"scenario_id": ["S1"], "transaction_id": ["T2"]})
        scenarios = pd.DataFrame({"scenario_id": ["S1", "S2"], "split": ["train", "train"], "is_anomaly": [1, 0]})
        metrics = evaluate_scenario_evidence(alerts, membership, scenarios, split="train")
        self.assertEqual(metrics["true_positive_scenarios"], 1)
        self.assertEqual(metrics["evaluation_unit"], "scenario")


if __name__ == "__main__":
    unittest.main()
