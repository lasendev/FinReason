from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from finreason.acquisition import (
    AcquisitionError,
    acquire_finreason,
    acquire_ibm_aml,
    acquire_paysim,
    find_csv,
    safe_extract_zip,
    sha256_file,
    validate_file,
    verify_checksum_manifest,
)


class AcquisitionTests(unittest.TestCase):
    def test_sha256_and_expected_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"finreason")
            expected = hashlib.sha256(b"finreason").hexdigest()
            self.assertEqual(sha256_file(path), expected)
            self.assertEqual(validate_file(path, expected_sha256=expected), path)

    def test_html_download_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "download.zip"
            path.write_text("<!doctype html><title>Google Drive</title>", encoding="utf-8")
            with self.assertRaises(AcquisitionError):
                validate_file(path)

    def test_partial_file_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.csv"
            path.write_bytes(b"x" * 120)
            with self.assertRaises(AcquisitionError):
                validate_file(path, expected_bytes=121)

    def test_zip_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "unsafe")
            with self.assertRaises(AcquisitionError):
                safe_extract_zip(archive, root / "output")

    def test_valid_zip_and_csv_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "valid.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("nested/data.csv", "step,type,amount,nameOrig,nameDest,isFraud\n1,PAYMENT,10,A,B,0\n")
            extracted = root / "extracted"
            safe_extract_zip(archive, extracted)
            match = find_csv(extracted, {"step", "nameOrig", "nameDest", "isFraud"})
            self.assertEqual(match.name, "data.csv")

    def test_embedded_finreason_checksums(self) -> None:
        manifest = PROJECT_ROOT / "finreason_synthetic_dataset" / "data" / "checksums.sha256"
        result = verify_checksum_manifest(manifest)
        self.assertTrue(result)
        self.assertTrue(all(result.values()))

    def test_finreason_uses_bundled_dataset_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            dataset_root, records = acquire_finreason(
                project_root=PROJECT_ROOT,
                raw_dir=raw,
                file_id="unused",
                archive_name="unused.zip",
                expected_sha256="unused",
            )
            self.assertEqual(dataset_root, PROJECT_ROOT / "finreason_synthetic_dataset")
            self.assertEqual(records[0]["status"], "REUSED_BUNDLED_DATASET")
            self.assertFalse((raw / "unused.zip").exists())

    def test_paysim_reuses_extracted_cache_without_gdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            extracted = raw / "paysim"
            extracted.mkdir()
            csv_path = extracted / "PaySim.csv"
            csv_path.write_text(
                "step,type,amount,nameOrig,nameDest,isFraud\n1,PAYMENT,10,A,B,0\n",
                encoding="utf-8",
            )
            resolved, records = acquire_paysim(
                raw_dir=raw, file_id="unused", download_name="unused", force=False
            )
            self.assertEqual(resolved, csv_path)
            self.assertEqual(records[0]["status"], "REUSED_EXTRACTED_CACHE")

    def test_ibm_reuses_extracted_cache_without_gdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            destination = raw / "ibm_aml"
            destination.mkdir()
            transactions = destination / "HI-Small_Trans.csv"
            transactions.write_text(
                "Timestamp,Amount Paid,Payment Currency,Is Laundering\n"
                + "2022-09-01 00:00,100.0,US Dollar,0\n" * 4,
                encoding="utf-8",
            )
            patterns = destination / "HI-Small_Patterns.txt"
            patterns.write_text("synthetic pattern fixture\n" * 10, encoding="utf-8")
            paths, _ = acquire_ibm_aml(
                raw_dir=raw,
                file_id="unused",
                download_name="unused",
                transaction_file=transactions.name,
                pattern_file=patterns.name,
                minimum_free_bytes=10**15,
            )
            self.assertEqual(paths["transactions"], transactions)
            self.assertEqual(paths["patterns"], patterns)

    def test_ibm_drive_bundle_must_be_a_zip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            payload = raw / "ibm_bundle"
            payload.write_text("not a ZIP bundle" * 20, encoding="utf-8")
            with patch(
                "finreason.acquisition.download_google_drive",
                return_value=payload,
            ) as download:
                with self.assertRaisesRegex(AcquisitionError, "invalid after a clean re-download"):
                    acquire_ibm_aml(
                        raw_dir=raw,
                        file_id="unused",
                        download_name=payload.name,
                        transaction_file="HI-Small_Trans.csv",
                        pattern_file="HI-Small_Patterns.txt",
                        minimum_free_bytes=1,
                    )
            self.assertEqual(download.call_count, 2)

    def test_ibm_drive_zip_is_extracted_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            payload = raw / "ibm_bundle"
            with zipfile.ZipFile(payload, "w") as bundle:
                bundle.writestr(
                    "IBM_AML_Dataset/HI-Small_Trans.csv",
                    "Timestamp,Amount Paid,Payment Currency,Is Laundering\n"
                    + "2022-09-01 00:00,100.0,US Dollar,0\n" * 4,
                )
                bundle.writestr(
                    "IBM_AML_Dataset/HI-Small_Patterns.txt",
                    "synthetic pattern fixture\n" * 10,
                )
            paths, records = acquire_ibm_aml(
                raw_dir=raw,
                file_id="unused",
                download_name=payload.name,
                transaction_file="HI-Small_Trans.csv",
                pattern_file="HI-Small_Patterns.txt",
                minimum_free_bytes=1,
            )
            self.assertEqual(paths["transactions"].name, "HI-Small_Trans.csv")
            self.assertEqual(paths["patterns"].name, "HI-Small_Patterns.txt")
            self.assertEqual(records[0]["source"], "ibm_aml_drive")

    def test_ibm_corrupt_resume_is_replaced_with_clean_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            valid_archive = raw / "valid.zip"
            with zipfile.ZipFile(valid_archive, "w") as bundle:
                bundle.writestr(
                    "IBM_AML_Dataset/HI-Small_Trans.csv",
                    "Timestamp,Amount Paid,Payment Currency,Is Laundering\n"
                    + "2022-09-01 00:00,100.0,US Dollar,0\n" * 4,
                )
                bundle.writestr(
                    "IBM_AML_Dataset/HI-Small_Patterns.txt",
                    "synthetic pattern fixture\n" * 10,
                )

            valid_bytes = valid_archive.read_bytes()
            payload = raw / "ibm_bundle"
            corrupt_bytes = bytearray(valid_bytes)
            first_header = corrupt_bytes.find(b"PK\x03\x04")
            self.assertGreaterEqual(first_header, 0)
            corrupt_bytes[first_header:first_header + 4] = b"NOPE"
            payload.write_bytes(corrupt_bytes)
            self.assertTrue(zipfile.is_zipfile(payload))

            def fake_download(_file_id, destination, *, force=False):
                if force:
                    destination.write_bytes(valid_bytes)
                return destination

            with patch("finreason.acquisition.download_google_drive", side_effect=fake_download) as download:
                paths, records = acquire_ibm_aml(
                    raw_dir=raw,
                    file_id="unused",
                    download_name=payload.name,
                    transaction_file="HI-Small_Trans.csv",
                    pattern_file="HI-Small_Patterns.txt",
                    minimum_free_bytes=1,
                )

            self.assertEqual(download.call_count, 2)
            self.assertEqual(paths["transactions"].name, "HI-Small_Trans.csv")
            statuses = {record["status"] for record in records if "status" in record}
            self.assertIn("REDOWNLOADED_AFTER_CORRUPT_CACHE", statuses)


if __name__ == "__main__":
    unittest.main()
