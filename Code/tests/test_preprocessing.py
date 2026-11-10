from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from finreason.preprocessing import (
    assign_temporal_split,
    clean_harmonized,
    create_exact_paysim_sample,
    derive_temporal_cutoffs,
    harmonize_ibm,
    harmonize_paysim,
    iter_paysim,
    parse_ibm_patterns,
)


class PreprocessingTests(unittest.TestCase):
    def test_exact_paysim_sample_is_deterministic_and_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paysim.csv"
            rows = ["step,type,amount,nameOrig,nameDest,isFraud,isFlaggedFraud"]
            rows.extend(f"{i % 10 + 1},PAYMENT,{i + 1},C{i},M{i},0,0" for i in range(101))
            source.write_text("\n".join(rows) + "\n", encoding="utf-8")
            output = root / "sample.csv.gz"
            metadata = root / "sample.json"
            first = create_exact_paysim_sample(source, output, metadata, fraction=0.10, seed=42, chunksize=17)
            selected_first = pd.read_csv(output)["_source_row"].tolist()
            second = create_exact_paysim_sample(source, output, metadata, fraction=0.10, seed=42, chunksize=11)
            selected_second = pd.read_csv(output)["_source_row"].tolist()
            self.assertEqual(first["sample_rows"], 10)
            self.assertEqual(selected_first, selected_second)
            self.assertEqual(second["status"], "REUSED_CACHE")

    def test_raw_paysim_ids_remain_unique_across_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paysim.csv"
            pd.DataFrame({
                "step": [1, 2, 3], "type": ["PAYMENT"] * 3, "amount": [1.0, 2.0, 3.0],
                "nameOrig": ["C1", "C2", "C3"], "nameDest": ["M1", "M2", "M3"],
                "isFraud": [0, 0, 1],
            }).to_csv(path, index=False)
            identifiers = [
                identifier
                for chunk in iter_paysim(path, chunksize=2)
                for identifier in chunk["transaction_id"].tolist()
            ]
            self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_paysim_fraud_is_not_relabelled_as_laundering(self) -> None:
        raw = pd.DataFrame({
            "step": [1, 743], "type": ["PAYMENT", "TRANSFER"], "amount": [10.0, 20.0],
            "nameOrig": ["C1", "C2"], "nameDest": ["M1", "C3"], "isFraud": [0, 1],
        })
        result = harmonize_paysim(raw)
        self.assertEqual(result["is_anomaly"].tolist(), [0, 1])
        self.assertTrue(result["is_laundering"].isna().all())
        self.assertEqual(result["split"].tolist(), ["train", "test"])

    def test_ibm_accounts_include_bank_and_currency_is_normalised(self) -> None:
        raw = pd.DataFrame({
            "Timestamp": ["2022/09/01 00:20"], "From Bank": ["010"], "Account": ["A1"],
            "To Bank": ["001"], "Account.1": ["B1"], "Amount Paid": [12.0],
            "Payment Currency": ["US Dollar"], "Receiving Currency": ["Euro"],
            "Payment Format": ["ACH"], "Is Laundering": [1],
        })
        result = harmonize_ibm(raw)
        self.assertEqual(result.loc[0, "from_account"], "IBM:010:A1")
        self.assertEqual(result.loc[0, "to_account"], "IBM:001:B1")
        self.assertEqual(result.loc[0, "payment_currency"], "USD")
        self.assertEqual(result.loc[0, "receiving_currency"], "EUR")

    def test_temporal_split_keeps_equal_values_together(self) -> None:
        values = pd.Series([1, 6, 6, 8, 10])
        result = assign_temporal_split(values, 1, 10)
        self.assertEqual(result.iloc[1], result.iloc[2])
        self.assertEqual(result.iloc[-1], "test")

    def test_count_balanced_cutoffs_are_chronological(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            pd.DataFrame({"step": list(range(1, 11))}).to_csv(path, index=False)
            cutoffs = derive_temporal_cutoffs(path, "step", chunksize=3)
            self.assertEqual(cutoffs["train_cutoff"], 6)
            self.assertEqual(cutoffs["validation_cutoff"], 8)
            self.assertEqual(cutoffs["expected_split_rows"], {"train": 6, "validation": 2, "test": 2})

    def test_cleaning_removes_invalid_graph_edges(self) -> None:
        raw = pd.DataFrame({
            "transaction_id": ["a", "b"], "timestamp": pd.to_datetime(["2025-01-01", None], utc=True),
            "event_step": pd.Series([pd.NA, pd.NA], dtype="Int64"), "from_account": ["A", "B"],
            "to_account": ["C", "D"], "amount": [10.0, -1.0], "payment_currency": ["USD", "USD"],
            "receiving_currency": ["USD", "USD"], "payment_format": ["ACH", "ACH"],
            "transaction_type": ["TRANSFER", "TRANSFER"], "source": ["x", "x"],
            "is_anomaly": pd.Series([0, 1], dtype="Int8"),
            "is_laundering": pd.Series([0, 1], dtype="Int8"),
            "scenario_id": pd.Series([pd.NA, pd.NA], dtype="string"), "split": ["train", "test"],
        })
        clean, diagnostics = clean_harmonized(raw)
        self.assertEqual(len(clean), 1)
        self.assertEqual(diagnostics["dropped_rows"], 1)
        self.assertEqual(diagnostics["invalid_timestamp_rows"], 1)
        self.assertEqual(diagnostics["invalid_amount_rows"], 1)

    def test_ibm_pattern_parser_reports_typologies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patterns.txt"
            path.write_text(
                "BEGIN LAUNDERING ATTEMPT - CYCLE: Max 3 hops\n"
                "2022/09/01 00:00,010,A,011,B,1.00,Euro,1.00,Euro,ACH,1\n"
                "END LAUNDERING ATTEMPT - CYCLE\n",
                encoding="utf-8",
            )
            report = parse_ibm_patterns(path)
            self.assertEqual(report["attempt_count"], 1)
            self.assertEqual(report["transaction_rows"], 1)
            self.assertEqual(report["attempts_by_typology"], {"CYCLE": 1})
            self.assertIsNone(report["unclosed_attempt"])


if __name__ == "__main__":
    unittest.main()
