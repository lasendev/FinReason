"""Safe, cache-aware acquisition helpers for the FinReason data sources."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class AcquisitionError(RuntimeError):
    """Raised when a downloaded or cached source is unsafe or invalid."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looks_like_html(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(2048).lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html") or b"<title>google drive" in prefix


def validate_file(
    path: Path,
    *,
    minimum_bytes: int = 1,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    require_zip: bool = False,
) -> Path:
    if not path.is_file():
        raise AcquisitionError(f"Required file does not exist: {path}")
    if path.stat().st_size < minimum_bytes:
        raise AcquisitionError(f"File is unexpectedly small ({path.stat().st_size} bytes): {path}")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise AcquisitionError(
            f"File-size mismatch for {path.name}: expected {expected_bytes} bytes, "
            f"received {path.stat().st_size}"
        )
    if looks_like_html(path):
        raise AcquisitionError(f"Received an HTML page instead of a data file: {path}")
    if require_zip and not zipfile.is_zipfile(path):
        raise AcquisitionError(f"Expected a valid ZIP archive: {path}")
    if expected_sha256:
        actual = sha256_file(path)
        if actual.lower() != expected_sha256.lower():
            raise AcquisitionError(f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, received {actual}")
    return path


def safe_extract_zip(archive: Path, destination: Path, *, overwrite: bool = False) -> list[Path]:
    validate_file(archive, minimum_bytes=22, require_zip=True)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if destination_root != target and destination_root not in target.parents:
                    raise AcquisitionError(f"Unsafe ZIP member path: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not overwrite:
                    extracted.append(target)
                    continue
                try:
                    with bundle.open(member) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                except zipfile.BadZipFile:
                    target.unlink(missing_ok=True)
                    raise
                extracted.append(target)
    except zipfile.BadZipFile as exc:
        raise AcquisitionError(f"ZIP archive is internally corrupt: {archive}") from exc
    return extracted


def verify_checksum_manifest(manifest_path: Path) -> dict[str, bool]:
    if not manifest_path.is_file():
        raise AcquisitionError(f"Checksum manifest not found: {manifest_path}")
    results: dict[str, bool] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        expected, filename = raw_line.split(maxsplit=1)
        filename = filename.strip().lstrip("*")
        candidate = manifest_path.parent / filename
        results[filename] = candidate.is_file() and sha256_file(candidate) == expected
    if not results or not all(results.values()):
        failed = [name for name, valid in results.items() if not valid]
        raise AcquisitionError(f"Embedded checksum verification failed: {failed}")
    return results


def csv_columns(path: Path) -> list[str]:
    validate_file(path, minimum_bytes=2)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise AcquisitionError(f"CSV is empty: {path}") from exc


def require_csv_columns(path: Path, required: Iterable[str]) -> Path:
    actual = set(csv_columns(path))
    missing = set(required) - actual
    if missing:
        raise AcquisitionError(f"{path.name} is missing required columns: {sorted(missing)}")
    return path


def find_csv(root: Path, required_columns: Iterable[str]) -> Path:
    candidates = sorted(root.rglob("*.csv")) if root.is_dir() else [root]
    matches: list[Path] = []
    for candidate in candidates:
        try:
            require_csv_columns(candidate, required_columns)
            matches.append(candidate)
        except (AcquisitionError, UnicodeDecodeError):
            continue
    if len(matches) != 1:
        raise AcquisitionError(f"Expected one matching CSV under {root}; found {[str(p) for p in matches]}")
    return matches[0]


def download_google_drive(file_id: str, destination: Path, *, force: bool = False) -> Path:
    if destination.exists() and not force:
        return validate_file(destination, minimum_bytes=100)
    if importlib.util.find_spec("gdown") is None:
        raise AcquisitionError("gdown is not installed. Run: python -m pip install -r requirements.txt")
    import gdown  # type: ignore

    destination.parent.mkdir(parents=True, exist_ok=True)
    if force and destination.exists():
        destination.unlink()
    try:
        result = gdown.download(id=file_id, output=str(destination), quiet=True, resume=True)
    except Exception as exc:
        raise AcquisitionError(f"Google Drive download failed for file ID {file_id}: {exc}") from exc
    if not result:
        raise AcquisitionError(f"Google Drive download failed for file ID {file_id}")
    return validate_file(destination, minimum_bytes=100)


def _copy_cached_source(source: Path, destination: Path, expected_sha256: str | None) -> Path:
    validate_file(source, minimum_bytes=100, expected_sha256=expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return validate_file(destination, minimum_bytes=100, expected_sha256=expected_sha256)


def describe_file(path: Path, source: str, status: str = "READY", hash_limit_bytes: int = 100 * 1024 * 1024) -> dict[str, Any]:
    size = path.stat().st_size
    return {
        "source": source,
        "status": status,
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": sha256_file(path) if size <= hash_limit_bytes else None,
        "hash_status": "COMPUTED" if size <= hash_limit_bytes else "SKIPPED_LARGE_FILE",
    }


def _validate_finreason_root(dataset_root: Path) -> Path:
    data_dir = dataset_root / "data"
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.is_file():
        raise AcquisitionError(f"FinReason metadata.json not found under {dataset_root}")
    verify_checksum_manifest(data_dir / "checksums.sha256")
    require_csv_columns(
        data_dir / "transactions.csv",
        {"transaction_id", "timestamp", "from_account", "to_account", "is_anomaly"},
    )
    return dataset_root


def acquire_finreason(
    *,
    project_root: Path,
    raw_dir: Path,
    file_id: str,
    archive_name: str,
    expected_sha256: str,
    force: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    bundled_root = project_root / "finreason_synthetic_dataset"
    if not force and (bundled_root / "data" / "metadata.json").is_file():
        _validate_finreason_root(bundled_root)
        record = describe_file(
            bundled_root / "data" / "metadata.json",
            "finreason_bundled",
            status="REUSED_BUNDLED_DATASET",
        )
        return bundled_root, [record]

    archive = raw_dir / archive_name
    local_archive = project_root / archive_name
    if not force and archive.exists():
        validate_file(archive, minimum_bytes=100, expected_sha256=expected_sha256, require_zip=True)
    elif local_archive.exists():
        _copy_cached_source(local_archive, archive, expected_sha256)
    else:
        download_google_drive(file_id, archive, force=force)
        validate_file(archive, minimum_bytes=100, expected_sha256=expected_sha256, require_zip=True)

    extraction_root = raw_dir / "finreason"
    metadata_matches = list(extraction_root.rglob("metadata.json")) if extraction_root.exists() else []
    if force or not metadata_matches:
        safe_extract_zip(archive, extraction_root, overwrite=force)
        metadata_matches = list(extraction_root.rglob("metadata.json"))
    if len(metadata_matches) != 1:
        raise AcquisitionError(f"Expected one FinReason metadata.json; found {len(metadata_matches)}")
    dataset_root = metadata_matches[0].parent.parent
    _validate_finreason_root(dataset_root)
    records = [describe_file(archive, "finreason_drive")]
    return dataset_root, records


def acquire_paysim(
    *,
    raw_dir: Path,
    file_id: str,
    download_name: str,
    force: bool = False,
) -> tuple[Path, list[dict[str, Any]]]:
    extraction_root = raw_dir / "paysim"
    required = {"step", "type", "amount", "nameOrig", "nameDest", "isFraud"}
    if extraction_root.exists() and not force:
        try:
            csv_path = find_csv(extraction_root, required)
            return csv_path, [describe_file(csv_path, "paysim_csv", status="REUSED_EXTRACTED_CACHE")]
        except AcquisitionError:
            pass

    payload = raw_dir / download_name
    download_google_drive(file_id, payload, force=force)
    if zipfile.is_zipfile(payload):
        if force or not extraction_root.exists():
            safe_extract_zip(payload, extraction_root, overwrite=force)
        csv_path = find_csv(extraction_root, required)
    else:
        csv_path = require_csv_columns(payload, required)
    records = [describe_file(payload, "paysim_drive"), describe_file(csv_path, "paysim_csv")]
    return csv_path, records


def require_free_space(destination: Path, minimum_free_bytes: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination).free
    if free < minimum_free_bytes:
        raise AcquisitionError(
            f"Insufficient free disk space: {free / (1024 ** 2):.1f} MiB available; "
            f"at least {minimum_free_bytes / (1024 ** 2):.1f} MiB required."
        )


def acquire_ibm_aml(
    *,
    raw_dir: Path,
    file_id: str,
    download_name: str,
    transaction_file: str,
    pattern_file: str,
    transaction_expected_bytes: int | None = None,
    minimum_free_bytes: int = 700 * 1024 * 1024,
    force: bool = False,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    destination = raw_dir / "ibm_aml"
    cached_transactions = sorted(destination.rglob(transaction_file)) if destination.exists() else []
    cached_patterns = sorted(destination.rglob(pattern_file)) if destination.exists() else []
    if not force and len(cached_transactions) == 1 and len(cached_patterns) == 1:
        transaction_path = validate_file(
            cached_transactions[0], minimum_bytes=100, expected_bytes=transaction_expected_bytes
        )
        pattern_path = validate_file(cached_patterns[0], minimum_bytes=100)
        status = "REUSED_EXTRACTED_CACHE"
        payload: Path | None = None
    else:
        require_free_space(destination, minimum_free_bytes)
        payload = raw_dir / download_name
        recovered_corrupt_cache = False
        for attempt in range(2):
            clean_download = force or attempt == 1
            try:
                download_google_drive(file_id, payload, force=clean_download)
                validate_file(payload, minimum_bytes=100, require_zip=True)
                safe_extract_zip(payload, destination, overwrite=clean_download)
                break
            except AcquisitionError as exc:
                if attempt == 1:
                    raise AcquisitionError(
                        "IBM AML archive is invalid after a clean re-download. "
                        "Check the Drive file and available disk space."
                    ) from exc
                payload.unlink(missing_ok=True)
                recovered_corrupt_cache = True
        transaction_matches = sorted(destination.rglob(transaction_file))
        pattern_matches = sorted(destination.rglob(pattern_file))
        if len(transaction_matches) != 1 or len(pattern_matches) != 1:
            raise AcquisitionError(
                "IBM Drive bundle must contain exactly one "
                f"{transaction_file} and one {pattern_file}; found "
                f"{len(transaction_matches)} and {len(pattern_matches)}."
            )
        transaction_path = validate_file(
            transaction_matches[0], minimum_bytes=100, expected_bytes=transaction_expected_bytes
        )
        pattern_path = validate_file(pattern_matches[0], minimum_bytes=100)
        status = "REDOWNLOADED_AFTER_CORRUPT_CACHE" if recovered_corrupt_cache else "READY"

    require_csv_columns(transaction_path, {"Timestamp", "Amount Paid", "Payment Currency", "Is Laundering"})
    paths = {"transactions": transaction_path, "patterns": pattern_path}
    records = []
    if payload is not None:
        records.append(describe_file(payload, "ibm_aml_drive"))
    records.extend([
        describe_file(transaction_path, "ibm_aml_transactions", status=status),
        describe_file(pattern_path, "ibm_aml_patterns", status=status),
    ])
    return paths, records


def write_acquisition_manifest(records: list[dict[str, Any]], destination: Path, source_config: dict[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": source_config,
        "records": records,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination
