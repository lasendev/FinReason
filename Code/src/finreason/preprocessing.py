"""Schema harmonisation, deterministic sampling and quality checks for FinReason."""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd


COMMON_COLUMNS = [
    "transaction_id",
    "timestamp",
    "event_step",
    "from_account",
    "to_account",
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
]

VALID_SPLITS = ("train", "validation", "test")
PAYSIM_TIME_ANCHOR = pd.Timestamp("2017-01-01T00:00:00Z")

IBM_CURRENCY_TO_ISO = {
    "Australian Dollar": "AUD",
    "Bitcoin": "BTC",
    "Brazil Real": "BRL",
    "Canadian Dollar": "CAD",
    "Euro": "EUR",
    "Mexican Peso": "MXN",
    "Ruble": "RUB",
    "Rupee": "INR",
    "Saudi Riyal": "SAR",
    "Shekel": "ILS",
    "Swiss Franc": "CHF",
    "UK Pound": "GBP",
    "US Dollar": "USD",
    "Yen": "JPY",
    "Yuan": "CNY",
}


def count_csv_rows(path: Path) -> int:
    """Count data rows without materialising the CSV."""
    opener = gzip.open if path.suffix == ".gz" else open
    newline_count = 0
    final_byte = b""
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            newline_count += block.count(b"\n")
            final_byte = block[-1:]
    total_lines = newline_count + int(bool(final_byte) and final_byte != b"\n")
    return max(0, total_lines - 1)


def create_exact_paysim_sample(
    source_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    fraction: float,
    seed: int,
    chunksize: int = 250_000,
) -> dict[str, Any]:
    """Create an exact-size uniform sample using stable source-row positions."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    source_stat = source_path.stat()
    signature = {
        "source_path": str(source_path.resolve()),
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "fraction": fraction,
        "seed": seed,
    }
    if output_path.is_file() and metadata_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in signature.items()):
            if output_path.stat().st_size == cached.get("sample_bytes"):
                return {**cached, "status": "REUSED_CACHE"}

    total_rows = count_csv_rows(source_path)
    sample_rows = int(total_rows * fraction)
    selected = np.sort(np.random.default_rng(seed).choice(total_rows, size=sample_rows, replace=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    selected_cursor = 0
    source_offset = 0
    written_rows = 0
    required = {"step", "type", "amount", "nameOrig", "nameDest", "isFraud"}

    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8", newline="") as output_handle:
            first_chunk = True
            for chunk in pd.read_csv(source_path, chunksize=chunksize):
                if first_chunk:
                    missing = required - set(chunk.columns)
                    if missing:
                        raise ValueError(f"PaySim source is missing columns: {sorted(missing)}")
                chunk_end = source_offset + len(chunk)
                next_cursor = int(np.searchsorted(selected, chunk_end, side="left"))
                local_positions = selected[selected_cursor:next_cursor] - source_offset
                if len(local_positions):
                    sampled = chunk.iloc[local_positions].copy()
                    sampled.insert(0, "_source_row", selected[selected_cursor:next_cursor])
                    sampled.to_csv(output_handle, index=False, header=first_chunk)
                    first_chunk = False
                    written_rows += len(sampled)
                selected_cursor = next_cursor
                source_offset = chunk_end
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    if source_offset != total_rows or written_rows != sample_rows:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"PaySim sampling mismatch: source {source_offset}/{total_rows}, sample {written_rows}/{sample_rows}"
        )
    temporary_path.replace(output_path)
    payload = {
        **signature,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "population_rows": total_rows,
        "sample_rows": sample_rows,
        "sample_bytes": output_path.stat().st_size,
        "sampling_method": "uniform_without_replacement_by_source_row",
        "status": "CREATED",
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def assign_temporal_split(
    values: pd.Series,
    minimum: Any,
    maximum: Any,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> pd.Series:
    """Assign whole timestamps/steps to chronological train, validation and test sets."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one")
    span = maximum - minimum
    train_cutoff = minimum + span * train_fraction
    validation_cutoff = minimum + span * (train_fraction + validation_fraction)
    result = np.where(values <= train_cutoff, "train", np.where(values <= validation_cutoff, "validation", "test"))
    return pd.Series(result, index=values.index, dtype="string")


def assign_temporal_split_by_cutoffs(
    values: pd.Series,
    train_cutoff: Any,
    validation_cutoff: Any,
) -> pd.Series:
    """Assign splits using count-balanced cutoffs derived from source data."""
    result = np.where(
        values <= train_cutoff,
        "train",
        np.where(values <= validation_cutoff, "validation", "test"),
    )
    return pd.Series(result, index=values.index, dtype="string")


def derive_temporal_cutoffs(
    path: Path,
    column: str,
    *,
    chunksize: int = 250_000,
    datetime_format: str | None = None,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, Any]:
    """Find chronological cutoffs nearest to target cumulative row shares."""
    counts: Counter[Any] = Counter()
    invalid_rows = 0
    for chunk in pd.read_csv(path, usecols=[column], chunksize=chunksize):
        if datetime_format:
            values = pd.to_datetime(chunk[column], format=datetime_format, utc=True, errors="coerce")
        else:
            values = pd.to_numeric(chunk[column], errors="coerce")
        invalid_rows += int(values.isna().sum())
        counts.update(values.dropna().value_counts().to_dict())
    if not counts:
        raise ValueError(f"No valid temporal values found in {path}:{column}")
    ordered = sorted(counts.items(), key=lambda item: item[0])
    total_rows = int(sum(count for _, count in ordered))
    train_target = total_rows * train_fraction
    validation_target = total_rows * (train_fraction + validation_fraction)
    cumulative = 0
    train_cutoff = None
    validation_cutoff = None
    train_rows = 0
    validation_end_rows = 0
    for value, count in ordered:
        cumulative += int(count)
        if train_cutoff is None and cumulative >= train_target:
            train_cutoff = value
            train_rows = cumulative
        if validation_cutoff is None and cumulative >= validation_target:
            validation_cutoff = value
            validation_end_rows = cumulative
            break
    assert train_cutoff is not None and validation_cutoff is not None
    return {
        "train_cutoff": train_cutoff,
        "validation_cutoff": validation_cutoff,
        "valid_rows": total_rows,
        "invalid_rows": invalid_rows,
        "expected_split_rows": {
            "train": train_rows,
            "validation": validation_end_rows - train_rows,
            "test": total_rows - validation_end_rows,
        },
        "method": "chronological_count_balanced_keep_equal_times_together",
    }


def _normalise_currency(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    return text.map(IBM_CURRENCY_TO_ISO).fillna(text.str.upper())


def _common_frame(**columns: Any) -> pd.DataFrame:
    frame = pd.DataFrame(columns)
    return frame.loc[:, COMMON_COLUMNS]


def harmonize_finreason(frame: pd.DataFrame) -> pd.DataFrame:
    return _common_frame(
        transaction_id=frame["transaction_id"].astype("string"),
        timestamp=pd.to_datetime(frame["timestamp"], utc=True, errors="coerce"),
        event_step=pd.Series(pd.NA, index=frame.index, dtype="Int64"),
        from_account="FR:" + frame["from_account"].astype("string").str.strip(),
        to_account="FR:" + frame["to_account"].astype("string").str.strip(),
        amount=pd.to_numeric(frame["amount"], errors="coerce"),
        payment_currency=frame["payment_currency"].astype("string").str.upper(),
        receiving_currency=frame["receiving_currency"].astype("string").str.upper(),
        payment_format=frame["payment_format"].astype("string").str.upper(),
        transaction_type=frame["transaction_type"].astype("string").str.upper(),
        source="finreason",
        is_anomaly=pd.to_numeric(frame["is_anomaly"], errors="coerce").astype("Int8"),
        is_laundering=pd.to_numeric(frame["is_laundering"], errors="coerce").astype("Int8"),
        scenario_id=frame["scenario_id"].astype("string"),
        split=frame["split"].astype("string").str.lower(),
    )


def harmonize_paysim(
    frame: pd.DataFrame,
    *,
    source_row_start: int = 0,
    minimum_step: int = 1,
    maximum_step: int = 743,
    train_cutoff: int | None = None,
    validation_cutoff: int | None = None,
) -> pd.DataFrame:
    if "_source_row" in frame:
        source_rows = pd.to_numeric(frame["_source_row"], errors="raise").astype("int64")
    else:
        source_rows = pd.Series(np.arange(source_row_start, source_row_start + len(frame)), index=frame.index)
    steps = pd.to_numeric(frame["step"], errors="coerce").astype("Int64")
    fraud = pd.to_numeric(frame["isFraud"], errors="coerce").astype("Int8")
    split = (
        assign_temporal_split_by_cutoffs(steps, train_cutoff, validation_cutoff)
        if train_cutoff is not None and validation_cutoff is not None
        else assign_temporal_split(steps, minimum_step, maximum_step)
    )
    return _common_frame(
        transaction_id="PAYSIM:" + source_rows.astype("string").str.zfill(9),
        timestamp=PAYSIM_TIME_ANCHOR + pd.to_timedelta(steps - minimum_step, unit="h"),
        event_step=steps,
        from_account="PS:" + frame["nameOrig"].astype("string").str.strip(),
        to_account="PS:" + frame["nameDest"].astype("string").str.strip(),
        amount=pd.to_numeric(frame["amount"], errors="coerce"),
        payment_currency="SIM",
        receiving_currency="SIM",
        payment_format=frame["type"].astype("string").str.upper(),
        transaction_type=frame["type"].astype("string").str.upper(),
        source="paysim",
        is_anomaly=fraud,
        is_laundering=pd.Series(pd.NA, index=frame.index, dtype="Int8"),
        scenario_id=pd.Series(pd.NA, index=frame.index, dtype="string"),
        split=split,
    )


def harmonize_ibm(
    frame: pd.DataFrame,
    *,
    source_row_start: int = 0,
    minimum_timestamp: pd.Timestamp = pd.Timestamp("2022-09-01T00:00:00Z"),
    maximum_timestamp: pd.Timestamp = pd.Timestamp("2022-09-18T16:18:00Z"),
    train_cutoff: pd.Timestamp | None = None,
    validation_cutoff: pd.Timestamp | None = None,
) -> pd.DataFrame:
    source_rows = pd.Series(np.arange(source_row_start, source_row_start + len(frame)), index=frame.index)
    timestamp = pd.to_datetime(frame["Timestamp"], format="%Y/%m/%d %H:%M", utc=True, errors="coerce")
    label = pd.to_numeric(frame["Is Laundering"], errors="coerce").astype("Int8")
    from_bank = frame["From Bank"].astype("string").str.strip()
    to_bank = frame["To Bank"].astype("string").str.strip()
    from_account = frame["Account"].astype("string").str.strip()
    to_account = frame["Account.1"].astype("string").str.strip()
    split = (
        assign_temporal_split_by_cutoffs(timestamp, train_cutoff, validation_cutoff)
        if train_cutoff is not None and validation_cutoff is not None
        else assign_temporal_split(timestamp, minimum_timestamp, maximum_timestamp)
    )
    return _common_frame(
        transaction_id="IBM:" + source_rows.astype("string").str.zfill(9),
        timestamp=timestamp,
        event_step=pd.Series(pd.NA, index=frame.index, dtype="Int64"),
        from_account="IBM:" + from_bank + ":" + from_account,
        to_account="IBM:" + to_bank + ":" + to_account,
        amount=pd.to_numeric(frame["Amount Paid"], errors="coerce"),
        payment_currency=_normalise_currency(frame["Payment Currency"]),
        receiving_currency=_normalise_currency(frame["Receiving Currency"]),
        payment_format=frame["Payment Format"].astype("string").str.upper(),
        transaction_type="TRANSFER",
        source="ibm_aml",
        is_anomaly=label,
        is_laundering=label,
        scenario_id=pd.Series(pd.NA, index=frame.index, dtype="string"),
        split=split,
    )


def clean_harmonized(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_endpoint = (
        frame["from_account"].isna()
        | frame["to_account"].isna()
        | frame["from_account"].str.strip().eq("")
        | frame["to_account"].str.strip().eq("")
    )
    invalid_timestamp = frame["timestamp"].isna()
    invalid_amount = frame["amount"].isna() | frame["amount"].le(0)
    invalid_anomaly = frame["is_anomaly"].isna() | ~frame["is_anomaly"].isin([0, 1])
    known_laundering = frame["is_laundering"].notna()
    invalid_laundering = known_laundering & ~frame["is_laundering"].isin([0, 1])
    invalid_split = ~frame["split"].isin(VALID_SPLITS)
    duplicate_id = frame["transaction_id"].duplicated(keep="first")
    drop_mask = (
        missing_endpoint
        | invalid_timestamp
        | invalid_amount
        | invalid_anomaly
        | invalid_laundering
        | invalid_split
        | duplicate_id
    )
    diagnostics = {
        "input_rows": int(len(frame)),
        "missing_endpoint_rows": int(missing_endpoint.sum()),
        "invalid_timestamp_rows": int(invalid_timestamp.sum()),
        "invalid_amount_rows": int(invalid_amount.sum()),
        "invalid_anomaly_label_rows": int(invalid_anomaly.sum()),
        "invalid_laundering_label_rows": int(invalid_laundering.sum()),
        "invalid_split_rows": int(invalid_split.sum()),
        "duplicate_transaction_id_rows": int(duplicate_id.sum()),
        "dropped_rows": int(drop_mask.sum()),
    }
    return frame.loc[~drop_mask].reset_index(drop=True), diagnostics


def iter_finreason(path: Path) -> Iterator[pd.DataFrame]:
    yield harmonize_finreason(pd.read_csv(path))


def iter_paysim(
    path: Path,
    *,
    chunksize: int = 250_000,
    train_cutoff: int | None = None,
    validation_cutoff: int | None = None,
) -> Iterator[pd.DataFrame]:
    source_offset = 0
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield harmonize_paysim(
            chunk,
            source_row_start=source_offset,
            train_cutoff=train_cutoff,
            validation_cutoff=validation_cutoff,
        )
        source_offset += len(chunk)


def iter_ibm(
    path: Path,
    *,
    chunksize: int = 250_000,
    train_cutoff: pd.Timestamp | None = None,
    validation_cutoff: pd.Timestamp | None = None,
) -> Iterator[pd.DataFrame]:
    dtypes = {
        "Timestamp": "string",
        "From Bank": "string",
        "Account": "string",
        "To Bank": "string",
        "Account.1": "string",
        "Payment Currency": "string",
        "Receiving Currency": "string",
        "Payment Format": "string",
    }
    source_offset = 0
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunksize):
        yield harmonize_ibm(
            chunk,
            source_row_start=source_offset,
            train_cutoff=train_cutoff,
            validation_cutoff=validation_cutoff,
        )
        source_offset += len(chunk)


def profile_and_clean(chunks: Iterable[pd.DataFrame], source: str) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_anomaly_positive: Counter[str] = Counter()
    split_anomaly_known: Counter[str] = Counter()
    split_laundering_positive: Counter[str] = Counter()
    split_laundering_known: Counter[str] = Counter()
    split_min: dict[str, pd.Timestamp] = {}
    split_max: dict[str, pd.Timestamp] = {}
    currencies: set[str] = set()
    formats: set[str] = set()
    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None
    amount_min: float | None = None
    amount_max: float | None = None
    amount_sum = 0.0

    for harmonized in chunks:
        clean, diagnostics = clean_harmonized(harmonized)
        totals.update(diagnostics)
        totals["clean_rows"] += len(clean)
        totals["anomaly_known_rows"] += int(clean["is_anomaly"].notna().sum())
        totals["anomaly_positive_rows"] += int(clean["is_anomaly"].fillna(0).sum())
        totals["laundering_known_rows"] += int(clean["is_laundering"].notna().sum())
        totals["laundering_positive_rows"] += int(clean["is_laundering"].fillna(0).sum())
        totals["self_transfer_rows"] += int(clean["from_account"].eq(clean["to_account"]).sum())
        split_counts.update(clean["split"].value_counts().to_dict())
        for split, group in clean.groupby("split", observed=True):
            split_name = str(split)
            split_anomaly_known[split_name] += int(group["is_anomaly"].notna().sum())
            split_anomaly_positive[split_name] += int(group["is_anomaly"].fillna(0).sum())
            split_laundering_known[split_name] += int(group["is_laundering"].notna().sum())
            split_laundering_positive[split_name] += int(group["is_laundering"].fillna(0).sum())
        currencies.update(clean["payment_currency"].dropna().astype(str).unique())
        formats.update(clean["payment_format"].dropna().astype(str).unique())
        if not clean.empty:
            local_min = clean["timestamp"].min()
            local_max = clean["timestamp"].max()
            timestamp_min = local_min if timestamp_min is None else min(timestamp_min, local_min)
            timestamp_max = local_max if timestamp_max is None else max(timestamp_max, local_max)
            local_amount_min = float(clean["amount"].min())
            local_amount_max = float(clean["amount"].max())
            amount_min = local_amount_min if amount_min is None else min(amount_min, local_amount_min)
            amount_max = local_amount_max if amount_max is None else max(amount_max, local_amount_max)
            amount_sum += float(clean["amount"].sum())
            for split, group in clean.groupby("split", observed=True):
                group_min, group_max = group["timestamp"].min(), group["timestamp"].max()
                split_min[str(split)] = min(split_min.get(str(split), group_min), group_min)
                split_max[str(split)] = max(split_max.get(str(split), group_max), group_max)

    temporal_order_valid = all(
        left not in split_max or right not in split_min or split_max[left] <= split_min[right]
        for left, right in (("train", "validation"), ("validation", "test"))
    )
    clean_rows = totals["clean_rows"]
    return {
        "source": source,
        **dict(totals),
        "dropped_rate": totals["dropped_rows"] / totals["input_rows"] if totals["input_rows"] else 0.0,
        "anomaly_rate": totals["anomaly_positive_rows"] / totals["anomaly_known_rows"] if totals["anomaly_known_rows"] else None,
        "laundering_rate": totals["laundering_positive_rows"] / totals["laundering_known_rows"] if totals["laundering_known_rows"] else None,
        "split_counts": dict(split_counts),
        "split_label_counts": {
            split: {
                "anomaly_known": int(split_anomaly_known[split]),
                "anomaly_positive": int(split_anomaly_positive[split]),
                "laundering_known": int(split_laundering_known[split]),
                "laundering_positive": int(split_laundering_positive[split]),
            }
            for split in VALID_SPLITS
        },
        "timestamp_min": timestamp_min.isoformat() if timestamp_min is not None else None,
        "timestamp_max": timestamp_max.isoformat() if timestamp_max is not None else None,
        "amount_min": amount_min,
        "amount_max": amount_max,
        "amount_mean": amount_sum / clean_rows if clean_rows else None,
        "currency_values": sorted(currencies),
        "payment_format_values": sorted(formats),
        "split_time_ranges": {
            split: {
                "min": split_min[split].isoformat() if split in split_min else None,
                "max": split_max[split].isoformat() if split in split_max else None,
            }
            for split in VALID_SPLITS
        },
        "temporal_order_valid": temporal_order_valid,
        "quality_status": "PASS" if totals["dropped_rows"] == 0 and temporal_order_valid else "PASS_WITH_CLEANING",
    }


def parse_ibm_patterns(path: Path) -> dict[str, Any]:
    begin_pattern = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([^:]+)")
    attempts: Counter[str] = Counter()
    transaction_rows = 0
    open_attempt: str | None = None
    malformed_rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        match = begin_pattern.match(line)
        if match:
            open_attempt = match.group(1).strip()
            attempts[open_attempt] += 1
        elif line.startswith("END LAUNDERING ATTEMPT"):
            open_attempt = None
        elif line.startswith("20"):
            try:
                if len(next(csv.reader([line]))) != 11:
                    malformed_rows += 1
                else:
                    transaction_rows += 1
            except csv.Error:
                malformed_rows += 1
    return {
        "attempt_count": int(sum(attempts.values())),
        "transaction_rows": transaction_rows,
        "attempts_by_typology": dict(sorted(attempts.items())),
        "malformed_transaction_rows": malformed_rows,
        "unclosed_attempt": open_attempt,
    }


def schema_mapping_records() -> list[dict[str, str]]:
    mappings = {
        "transaction_id": ("transaction_id", "derived from source row", "derived from source row"),
        "timestamp": ("timestamp", "step → relative UTC hour", "Timestamp"),
        "from_account": ("from_account", "nameOrig", "From Bank + Account"),
        "to_account": ("to_account", "nameDest", "To Bank + Account.1"),
        "amount": ("amount", "amount", "Amount Paid"),
        "payment_currency": ("payment_currency", "SIM (unit unspecified)", "Payment Currency → ISO"),
        "payment_format": ("payment_format", "type", "Payment Format"),
        "is_anomaly": ("is_anomaly", "isFraud", "Is Laundering"),
        "is_laundering": ("is_laundering", "unknown (not inferred)", "Is Laundering"),
        "split": ("provided chronological split", "derived from step", "derived from timestamp"),
    }
    return [
        {"common_field": field, "finreason": values[0], "paysim": values[1], "ibm_aml": values[2]}
        for field, values in mappings.items()
    ]
