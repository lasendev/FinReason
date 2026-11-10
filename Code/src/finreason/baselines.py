"""Leakage-safe statistical baselines with transaction-level explanations."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


REQUIRED_SCORE_COLUMNS = {
    "transaction_id", "timestamp", "from_account", "amount", "payment_currency", "split"
}


def _utc_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _validate_transactions(frame: pd.DataFrame) -> None:
    missing = REQUIRED_SCORE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing baseline columns: {sorted(missing)}")
    if frame["transaction_id"].duplicated().any():
        raise ValueError("transaction_id must be unique")


@dataclass(frozen=True)
class StatisticalAlert:
    alert_id: str
    detector: str
    pattern_type: str
    subject_id: str
    split: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    observed_value: float
    expected_center: float
    threshold: float
    score: float
    statistics_level: str
    transaction_ids: tuple[str, ...]
    fit_split: str
    explanation: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["window_start"] = self.window_start.isoformat()
        record["window_end"] = self.window_end.isoformat()
        record["transaction_ids"] = "|".join(self.transaction_ids)
        return record

    def to_evidence_record(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "detector_family": "STATISTICAL",
            "alert_type": self.pattern_type,
            "subject_id": self.subject_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "transaction_ids": "|".join(self.transaction_ids),
            "explanation": self.explanation,
        }


class AccountCurrencyZScoreBaseline:
    """Absolute amount Z-score using only fitted account/currency history."""

    def __init__(
        self,
        *,
        z_threshold: float = 3.0,
        minimum_history: int = 5,
        minimum_standard_deviation: float = 1e-9,
    ) -> None:
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if minimum_history < 2:
            raise ValueError("minimum_history must be at least two")
        self.z_threshold = float(z_threshold)
        self.minimum_history = int(minimum_history)
        self.minimum_standard_deviation = float(minimum_standard_deviation)
        self.account_currency_statistics: Optional[pd.DataFrame] = None
        self.currency_statistics: Optional[pd.DataFrame] = None
        self.global_mean: Optional[float] = None
        self.global_standard_deviation: Optional[float] = None
        self.fit_metadata: dict[str, Any] = {}

    def fit(self, transactions: pd.DataFrame, *, fit_split: str = "train") -> "AccountCurrencyZScoreBaseline":
        _validate_transactions(transactions)
        fit_rows = transactions.loc[transactions["split"].eq(fit_split), list(REQUIRED_SCORE_COLUMNS)].copy()
        fit_rows["timestamp"] = _utc_series(fit_rows["timestamp"])
        fit_rows["amount"] = pd.to_numeric(fit_rows["amount"], errors="coerce")
        fit_rows = fit_rows.loc[fit_rows["timestamp"].notna() & fit_rows["amount"].gt(0)]
        if len(fit_rows) < 2:
            raise ValueError("at least two valid fit rows are required")

        global_std = float(fit_rows["amount"].std(ddof=0))
        if not np.isfinite(global_std) or global_std <= self.minimum_standard_deviation:
            raise ValueError("global training amount variance must be positive")
        self.global_mean = float(fit_rows["amount"].mean())
        self.global_standard_deviation = global_std

        account = fit_rows.groupby(["from_account", "payment_currency"], dropna=False)["amount"].agg(
            history_count="count", mean="mean", standard_deviation=lambda values: values.std(ddof=0)
        ).reset_index()
        currency = fit_rows.groupby("payment_currency", dropna=False)["amount"].agg(
            history_count="count", mean="mean", standard_deviation=lambda values: values.std(ddof=0)
        ).reset_index()
        self.account_currency_statistics = account
        self.currency_statistics = currency
        self.fit_metadata = {
            "fit_split": fit_split,
            "fit_rows": int(len(fit_rows)),
            "fit_timestamp_min": fit_rows["timestamp"].min().isoformat(),
            "fit_timestamp_max": fit_rows["timestamp"].max().isoformat(),
            "z_threshold": self.z_threshold,
            "minimum_history": self.minimum_history,
            "amount_transform": "none",
            "grouping": "from_account + payment_currency",
            "label_columns_used": [],
        }
        return self

    def score_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if self.account_currency_statistics is None or self.currency_statistics is None:
            raise RuntimeError("fit must be called before scoring")
        _validate_transactions(transactions)
        scored = transactions[list(REQUIRED_SCORE_COLUMNS)].copy()
        scored["timestamp"] = _utc_series(scored["timestamp"])
        scored["amount"] = pd.to_numeric(scored["amount"], errors="coerce")
        scored = scored.merge(
            self.account_currency_statistics.rename(columns={
                "history_count": "account_history_count",
                "mean": "account_mean",
                "standard_deviation": "account_standard_deviation",
            }),
            on=["from_account", "payment_currency"], how="left", validate="many_to_one",
        ).merge(
            self.currency_statistics.rename(columns={
                "history_count": "currency_history_count",
                "mean": "currency_mean",
                "standard_deviation": "currency_standard_deviation",
            }),
            on="payment_currency", how="left", validate="many_to_one",
        )
        account_valid = (
            scored["account_history_count"].ge(self.minimum_history)
            & scored["account_standard_deviation"].gt(self.minimum_standard_deviation)
        )
        currency_valid = scored["currency_standard_deviation"].gt(self.minimum_standard_deviation)
        scored["expected_mean"] = np.where(
            account_valid, scored["account_mean"],
            np.where(currency_valid, scored["currency_mean"], self.global_mean),
        )
        scored["expected_standard_deviation"] = np.where(
            account_valid, scored["account_standard_deviation"],
            np.where(currency_valid, scored["currency_standard_deviation"], self.global_standard_deviation),
        )
        scored["statistics_level"] = np.select(
            [account_valid, currency_valid], ["ACCOUNT_CURRENCY", "CURRENCY"], default="GLOBAL"
        )
        scored["z_score"] = (
            (scored["amount"] - scored["expected_mean"])
            / scored["expected_standard_deviation"]
        )
        scored["absolute_z_score"] = scored["z_score"].abs()
        scored["is_alert"] = (
            scored["timestamp"].notna()
            & scored["amount"].gt(0)
            & scored["absolute_z_score"].ge(self.z_threshold)
        )
        return scored

    def predict(self, transactions: pd.DataFrame) -> list[StatisticalAlert]:
        scored = self.score_transactions(transactions)
        alerts: list[StatisticalAlert] = []
        for row in scored.loc[scored["is_alert"]].sort_values(["timestamp", "transaction_id"]).itertuples(index=False):
            alerts.append(StatisticalAlert(
                alert_id="",
                detector="ZSCORE_AMOUNT",
                pattern_type="UNUSUAL_AMOUNT",
                subject_id=str(row.from_account),
                split=str(row.split),
                window_start=row.timestamp,
                window_end=row.timestamp,
                observed_value=float(row.amount),
                expected_center=float(row.expected_mean),
                threshold=self.z_threshold,
                score=float(row.absolute_z_score),
                statistics_level=str(row.statistics_level),
                transaction_ids=(str(row.transaction_id),),
                fit_split=str(self.fit_metadata["fit_split"]),
                explanation=(
                    f"Amount {float(row.amount):,.2f} {row.payment_currency} is "
                    f"{float(row.absolute_z_score):.2f} standard deviations from the "
                    f"training-only {str(row.statistics_level).lower().replace('_', '/')} "
                    f"mean {float(row.expected_mean):,.2f}; threshold={self.z_threshold:.2f}."
                ),
            ))
        return [
            StatisticalAlert(**{**asdict(alert), "alert_id": f"ZAMT-{index:06d}"})
            for index, alert in enumerate(alerts, start=1)
        ]


class AccountDailyFrequencyIQRBaseline:
    """Upper Tukey-fence detector for outgoing UTC account-day counts."""

    def __init__(self, *, iqr_multiplier: float = 1.5, minimum_active_days: int = 5) -> None:
        if iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier must be positive")
        if minimum_active_days < 2:
            raise ValueError("minimum_active_days must be at least two")
        self.iqr_multiplier = float(iqr_multiplier)
        self.minimum_active_days = int(minimum_active_days)
        self.account_statistics: Optional[pd.DataFrame] = None
        self.global_q1: Optional[float] = None
        self.global_q3: Optional[float] = None
        self.global_iqr: Optional[float] = None
        self.global_upper_fence: Optional[float] = None
        self.fit_metadata: dict[str, Any] = {}

    @staticmethod
    def account_day_counts(transactions: pd.DataFrame) -> pd.DataFrame:
        _validate_transactions(transactions)
        frame = transactions[list(REQUIRED_SCORE_COLUMNS)].copy()
        frame["timestamp"] = _utc_series(frame["timestamp"])
        frame = frame.loc[frame["timestamp"].notna()]
        frame["utc_date"] = frame["timestamp"].dt.floor("D")
        counts = frame.groupby(["from_account", "utc_date", "split"], dropna=False).agg(
            transaction_count=("transaction_id", "size"),
            transaction_ids=("transaction_id", lambda values: tuple(sorted(map(str, values)))),
        ).reset_index()
        return counts

    def fit(self, transactions: pd.DataFrame, *, fit_split: str = "train") -> "AccountDailyFrequencyIQRBaseline":
        daily = self.account_day_counts(transactions)
        fit_daily = daily.loc[daily["split"].eq(fit_split)].copy()
        if len(fit_daily) < 2:
            raise ValueError("at least two active account-days are required")
        self.global_q1 = float(fit_daily["transaction_count"].quantile(0.25))
        self.global_q3 = float(fit_daily["transaction_count"].quantile(0.75))
        self.global_iqr = self.global_q3 - self.global_q1
        self.global_upper_fence = self.global_q3 + self.iqr_multiplier * self.global_iqr
        account = fit_daily.groupby("from_account")["transaction_count"].agg(
            active_days="count",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        ).reset_index()
        account["iqr"] = account["q3"] - account["q1"]
        account["local_upper_fence"] = account["q3"] + self.iqr_multiplier * account["iqr"]
        account["upper_fence"] = np.maximum(account["local_upper_fence"], self.global_upper_fence)
        self.account_statistics = account
        self.fit_metadata = {
            "fit_split": fit_split,
            "fit_account_days": int(len(fit_daily)),
            "fit_timestamp_min": fit_daily["utc_date"].min().isoformat(),
            "fit_timestamp_max": fit_daily["utc_date"].max().isoformat(),
            "iqr_multiplier": self.iqr_multiplier,
            "minimum_active_days": self.minimum_active_days,
            "window_definition": "UTC calendar day [00:00, next 00:00)",
            "zero_activity_days_included": False,
            "label_columns_used": [],
        }
        return self

    def score_account_days(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if self.account_statistics is None:
            raise RuntimeError("fit must be called before scoring")
        daily = self.account_day_counts(transactions).merge(
            self.account_statistics, on="from_account", how="left", validate="many_to_one"
        )
        account_valid = daily["active_days"].ge(self.minimum_active_days)
        daily["expected_q3"] = np.where(account_valid, daily["q3"], self.global_q3)
        daily["expected_iqr"] = np.where(account_valid, daily["iqr"], self.global_iqr)
        daily["effective_upper_fence"] = np.where(
            account_valid, daily["upper_fence"], self.global_upper_fence
        )
        daily["statistics_level"] = np.where(account_valid, "ACCOUNT", "GLOBAL")
        daily["iqr_excess_score"] = np.where(
            daily["expected_iqr"].gt(0),
            (daily["transaction_count"] - daily["expected_q3"]) / daily["expected_iqr"],
            np.where(daily["transaction_count"].gt(daily["expected_q3"]), np.inf, 0.0),
        )
        daily["is_alert"] = daily["transaction_count"].gt(daily["effective_upper_fence"])
        return daily

    def predict(self, transactions: pd.DataFrame) -> list[StatisticalAlert]:
        scored = self.score_account_days(transactions)
        alerts: list[StatisticalAlert] = []
        for row in scored.loc[scored["is_alert"]].sort_values(["utc_date", "from_account"]).itertuples(index=False):
            window_start = pd.Timestamp(row.utc_date)
            alerts.append(StatisticalAlert(
                alert_id="",
                detector="IQR_FREQUENCY",
                pattern_type="UNUSUAL_DAILY_FREQUENCY",
                subject_id=str(row.from_account),
                split=str(row.split),
                window_start=window_start,
                window_end=window_start + pd.Timedelta(days=1),
                observed_value=float(row.transaction_count),
                expected_center=float(row.expected_q3),
                threshold=float(row.effective_upper_fence),
                score=float(row.iqr_excess_score),
                statistics_level=str(row.statistics_level),
                transaction_ids=tuple(row.transaction_ids),
                fit_split=str(self.fit_metadata["fit_split"]),
                explanation=(
                    f"Outgoing UTC-day count {int(row.transaction_count)} exceeds the "
                    f"training-only {str(row.statistics_level).lower()} Tukey upper fence "
                    f"{float(row.effective_upper_fence):.2f} "
                    f"(Q3={float(row.expected_q3):.2f}, multiplier={self.iqr_multiplier:.2f})."
                ),
            ))
        return [
            StatisticalAlert(**{**asdict(alert), "alert_id": f"IFRQ-{index:06d}"})
            for index, alert in enumerate(alerts, start=1)
        ]


class CurrencyLogAmountZScoreBaseline:
    """One-sided log-amount Z-score with stable currency-level backoff.

    Raw transaction amounts are typically right-skewed and account/currency
    histories are often too sparse to estimate a reliable standard deviation.
    This baseline therefore fits ``log1p(amount)`` at currency level and only
    flags the upper tail.  Labels are never read while fitting or scoring.
    """

    def __init__(
        self,
        *,
        z_threshold: float = 2.0,
        minimum_currency_history: int = 30,
        minimum_standard_deviation: float = 1e-9,
    ) -> None:
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        if minimum_currency_history < 2:
            raise ValueError("minimum_currency_history must be at least two")
        self.z_threshold = float(z_threshold)
        self.minimum_currency_history = int(minimum_currency_history)
        self.minimum_standard_deviation = float(minimum_standard_deviation)
        self.currency_statistics: Optional[pd.DataFrame] = None
        self.global_log_mean: Optional[float] = None
        self.global_log_standard_deviation: Optional[float] = None
        self.fit_metadata: dict[str, Any] = {}

    def fit(
        self,
        transactions: pd.DataFrame,
        *,
        fit_split: str = "train",
    ) -> "CurrencyLogAmountZScoreBaseline":
        _validate_transactions(transactions)
        fit_rows = transactions.loc[
            transactions["split"].eq(fit_split), list(REQUIRED_SCORE_COLUMNS)
        ].copy()
        fit_rows["timestamp"] = _utc_series(fit_rows["timestamp"])
        fit_rows["amount"] = pd.to_numeric(fit_rows["amount"], errors="coerce")
        fit_rows = fit_rows.loc[fit_rows["timestamp"].notna() & fit_rows["amount"].gt(0)]
        if len(fit_rows) < 2:
            raise ValueError("at least two valid fit rows are required")
        fit_rows["log_amount"] = np.log1p(fit_rows["amount"])
        global_std = float(fit_rows["log_amount"].std(ddof=0))
        if not np.isfinite(global_std) or global_std <= self.minimum_standard_deviation:
            raise ValueError("global training log-amount variance must be positive")
        self.global_log_mean = float(fit_rows["log_amount"].mean())
        self.global_log_standard_deviation = global_std
        self.currency_statistics = fit_rows.groupby(
            "payment_currency", dropna=False
        )["log_amount"].agg(
            history_count="count",
            log_mean="mean",
            log_standard_deviation=lambda values: values.std(ddof=0),
        ).reset_index()
        self.fit_metadata = {
            "fit_split": fit_split,
            "fit_rows": int(len(fit_rows)),
            "fit_timestamp_min": fit_rows["timestamp"].min().isoformat(),
            "fit_timestamp_max": fit_rows["timestamp"].max().isoformat(),
            "z_threshold": self.z_threshold,
            "minimum_currency_history": self.minimum_currency_history,
            "amount_transform": "log1p",
            "tail": "upper_only",
            "grouping": "payment_currency",
            "label_columns_used": [],
        }
        return self

    def score_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if self.currency_statistics is None:
            raise RuntimeError("fit must be called before scoring")
        _validate_transactions(transactions)
        scored = transactions[list(REQUIRED_SCORE_COLUMNS)].copy()
        scored["timestamp"] = _utc_series(scored["timestamp"])
        scored["amount"] = pd.to_numeric(scored["amount"], errors="coerce")
        scored["log_amount"] = np.log1p(scored["amount"].clip(lower=0))
        scored = scored.merge(
            self.currency_statistics,
            on="payment_currency",
            how="left",
            validate="many_to_one",
        )
        currency_valid = (
            scored["history_count"].ge(self.minimum_currency_history)
            & scored["log_standard_deviation"].gt(self.minimum_standard_deviation)
        )
        scored["expected_log_mean"] = np.where(
            currency_valid, scored["log_mean"], self.global_log_mean
        )
        scored["expected_log_standard_deviation"] = np.where(
            currency_valid,
            scored["log_standard_deviation"],
            self.global_log_standard_deviation,
        )
        scored["statistics_level"] = np.where(currency_valid, "CURRENCY", "GLOBAL")
        scored["upper_z_score"] = (
            (scored["log_amount"] - scored["expected_log_mean"])
            / scored["expected_log_standard_deviation"]
        )
        scored["is_alert"] = (
            scored["timestamp"].notna()
            & scored["amount"].gt(0)
            & scored["upper_z_score"].ge(self.z_threshold)
        )
        return scored

    def predict(self, transactions: pd.DataFrame) -> list[StatisticalAlert]:
        scored = self.score_transactions(transactions)
        alerts: list[StatisticalAlert] = []
        selected = scored.loc[scored["is_alert"]].sort_values(
            ["timestamp", "transaction_id"]
        )
        for row in selected.itertuples(index=False):
            expected_amount = float(np.expm1(row.expected_log_mean))
            alerts.append(StatisticalAlert(
                alert_id="",
                detector="LOG_ZSCORE_AMOUNT",
                pattern_type="UNUSUAL_UPPER_TAIL_AMOUNT",
                subject_id=str(row.from_account),
                split=str(row.split),
                window_start=row.timestamp,
                window_end=row.timestamp,
                observed_value=float(row.amount),
                expected_center=expected_amount,
                threshold=self.z_threshold,
                score=float(row.upper_z_score),
                statistics_level=str(row.statistics_level),
                transaction_ids=(str(row.transaction_id),),
                fit_split=str(self.fit_metadata["fit_split"]),
                explanation=(
                    f"Amount {float(row.amount):,.2f} {row.payment_currency} has a "
                    f"training-only upper-tail log-amount Z-score of "
                    f"{float(row.upper_z_score):.2f}; threshold={self.z_threshold:.3f}."
                ),
            ))
        return [
            StatisticalAlert(**{**asdict(alert), "alert_id": f"LZAMT-{index:06d}"})
            for index, alert in enumerate(alerts, start=1)
        ]


class AccountRollingFrequencyBaseline:
    """Causal rolling account-frequency detector with bounded event state.

    Unlike UTC calendar-day bins, the rolling horizon crosses midnight.  The
    implementation keeps only events inside the active horizon, so memory is
    bounded by recent activity rather than the full transaction history.
    Overlapping trigger windows for an account are merged into one episode.
    """

    REQUIRED_COLUMNS = REQUIRED_SCORE_COLUMNS | {"transaction_type"}

    def __init__(
        self,
        *,
        window_hours: float = 48.0,
        minimum_count: int = 4,
        included_transaction_types: Iterable[str] = ("TRANSFER",),
    ) -> None:
        if window_hours <= 0:
            raise ValueError("window_hours must be positive")
        if minimum_count < 2:
            raise ValueError("minimum_count must be at least two")
        included = tuple(sorted({str(value).upper() for value in included_transaction_types}))
        if not included:
            raise ValueError("included_transaction_types cannot be empty")
        self.window_hours = float(window_hours)
        self.minimum_count = int(minimum_count)
        self.included_transaction_types = included
        self.fit_metadata: dict[str, Any] = {}
        self.last_score_metadata: dict[str, Any] = {}

    @classmethod
    def _validate_rolling_transactions(cls, frame: pd.DataFrame) -> None:
        missing = cls.REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"missing rolling-frequency columns: {sorted(missing)}")
        _validate_transactions(frame)

    def fit(
        self,
        transactions: pd.DataFrame,
        *,
        fit_split: str = "train",
    ) -> "AccountRollingFrequencyBaseline":
        self._validate_rolling_transactions(transactions)
        fit_rows = transactions.loc[transactions["split"].eq(fit_split)].copy()
        timestamps = _utc_series(fit_rows["timestamp"])
        eligible = fit_rows["transaction_type"].astype("string").str.upper().isin(
            self.included_transaction_types
        )
        valid = timestamps.notna() & eligible
        if int(valid.sum()) < self.minimum_count:
            raise ValueError("insufficient eligible fit rows for the rolling threshold")
        self.fit_metadata = {
            "fit_split": fit_split,
            "fit_rows": int(valid.sum()),
            "fit_timestamp_min": timestamps.loc[valid].min().isoformat(),
            "fit_timestamp_max": timestamps.loc[valid].max().isoformat(),
            "window_hours": self.window_hours,
            "minimum_count": self.minimum_count,
            "included_transaction_types": list(self.included_transaction_types),
            "window_definition": "causal rolling interval inclusive of both endpoints",
            "state_reset": "split boundary",
            "label_columns_used": [],
        }
        return self

    def score_windows(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if not self.fit_metadata:
            raise RuntimeError("fit must be called before scoring")
        self._validate_rolling_transactions(transactions)
        columns = [
            "transaction_id", "timestamp", "from_account", "amount",
            "payment_currency", "transaction_type", "split",
        ]
        frame = transactions[columns].copy()
        frame["timestamp"] = _utc_series(frame["timestamp"])
        frame["transaction_type"] = frame["transaction_type"].astype("string").str.upper()
        frame = frame.loc[
            frame["timestamp"].notna()
            & frame["transaction_type"].isin(self.included_transaction_types)
        ].sort_values(["split", "timestamp", "transaction_id"])

        duration = pd.Timedelta(hours=self.window_hours)
        episodes: list[dict[str, Any]] = []
        peak_active_events = 0
        peak_account_queue = 0
        scored_rows = 0

        for split_name, split_frame in frame.groupby("split", sort=False):
            account_events: defaultdict[str, deque[tuple[pd.Timestamp, str, float, str]]] = defaultdict(deque)
            global_events: deque[tuple[pd.Timestamp, str, str]] = deque()
            active_count = 0
            latest_episode_by_account: dict[str, dict[str, Any]] = {}

            for row in split_frame.itertuples(index=False):
                timestamp = pd.Timestamp(row.timestamp)
                cutoff = timestamp - duration
                while global_events and global_events[0][0] < cutoff:
                    old_timestamp, old_account, old_transaction_id = global_events.popleft()
                    queue = account_events[old_account]
                    if queue and queue[0][0] == old_timestamp and queue[0][1] == old_transaction_id:
                        queue.popleft()
                        active_count -= 1
                    if not queue:
                        account_events.pop(old_account, None)

                account = str(row.from_account)
                transaction_id = str(row.transaction_id)
                event = (timestamp, transaction_id, float(row.amount), str(row.payment_currency))
                account_events[account].append(event)
                global_events.append((timestamp, account, transaction_id))
                active_count += 1
                scored_rows += 1
                queue = account_events[account]
                peak_active_events = max(peak_active_events, active_count)
                peak_account_queue = max(peak_account_queue, len(queue))
                if len(queue) < self.minimum_count:
                    continue

                context_ids = [item[1] for item in queue]
                context_amounts = [item[2] for item in queue]
                window_start = queue[0][0]
                previous = latest_episode_by_account.get(account)
                if previous is not None and window_start <= previous["window_end"]:
                    previous["window_end"] = timestamp
                    previous["observed_value"] = max(previous["observed_value"], len(queue))
                    previous["total_amount"] += float(row.amount)
                    previous["transaction_ids"].update(context_ids)
                else:
                    episode = {
                        "subject_id": account,
                        "split": str(split_name),
                        "window_start": window_start,
                        "window_end": timestamp,
                        "observed_value": len(queue),
                        "total_amount": float(sum(context_amounts)),
                        "payment_currency": str(row.payment_currency),
                        "transaction_ids": set(context_ids),
                    }
                    episodes.append(episode)
                    latest_episode_by_account[account] = episode

        self.last_score_metadata = {
            "scored_eligible_rows": scored_rows,
            "episodes": len(episodes),
            "peak_active_events": peak_active_events,
            "peak_account_queue": peak_account_queue,
            "window_hours": self.window_hours,
            "complexity": "O(N) time; O(events in active horizon) state",
        }
        records = []
        for episode in episodes:
            records.append({
                **episode,
                "transaction_ids": tuple(sorted(episode["transaction_ids"])),
            })
        return pd.DataFrame(records)

    def predict(self, transactions: pd.DataFrame) -> list[StatisticalAlert]:
        windows = self.score_windows(transactions)
        if windows.empty:
            return []
        alerts = []
        ordered = windows.sort_values(["window_start", "subject_id", "window_end"])
        for row in ordered.itertuples(index=False):
            alerts.append(StatisticalAlert(
                alert_id="",
                detector="ROLLING_FREQUENCY",
                pattern_type="UNUSUAL_ROLLING_FREQUENCY",
                subject_id=str(row.subject_id),
                split=str(row.split),
                window_start=pd.Timestamp(row.window_start),
                window_end=pd.Timestamp(row.window_end),
                observed_value=float(row.observed_value),
                expected_center=float(self.minimum_count - 1),
                threshold=float(self.minimum_count),
                score=float(row.observed_value / self.minimum_count),
                statistics_level="ACCOUNT_ROLLING",
                transaction_ids=tuple(row.transaction_ids),
                fit_split=str(self.fit_metadata["fit_split"]),
                explanation=(
                    f"Account {row.subject_id} has at least {int(row.observed_value)} "
                    f"eligible transfers in a causal {self.window_hours:g}-hour rolling "
                    f"window; training-selected minimum={self.minimum_count}."
                ),
            ))
        return [
            StatisticalAlert(**{**asdict(alert), "alert_id": f"RFRQ-{index:06d}"})
            for index, alert in enumerate(alerts, start=1)
        ]


def evaluate_transaction_evidence(
    alerts: Iterable[StatisticalAlert],
    transactions: pd.DataFrame,
    *,
    split: str,
    label_column: str,
) -> dict[str, Any]:
    """Evaluate alert evidence at transaction level with a shared metric contract."""

    if label_column not in transactions.columns:
        raise ValueError(f"missing label column: {label_column}")
    target = transactions.loc[transactions["split"].eq(split), ["transaction_id", label_column]].copy()
    labels = pd.to_numeric(target[label_column], errors="coerce")
    known = target.loc[labels.isin([0, 1])].copy()
    known[label_column] = labels.loc[known.index].astype(int)
    known_ids = set(known["transaction_id"].astype(str))
    positive_ids = set(known.loc[known[label_column].eq(1), "transaction_id"].astype(str))
    predicted_ids = {
        transaction_id
        for alert in alerts if alert.split == split
        for transaction_id in alert.transaction_ids
        if transaction_id in known_ids
    }
    true_positive = len(predicted_ids & positive_ids)
    false_positive = len(predicted_ids - positive_ids)
    false_negative = len(positive_ids - predicted_ids)
    negative_count = len(known_ids) - len(positive_ids)
    precision = true_positive / len(predicted_ids) if predicted_ids else 0.0
    recall = true_positive / len(positive_ids) if positive_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "detector": "STATISTICAL_BASELINE",
        "split": split,
        "label_column": label_column,
        "known_transactions": len(known_ids),
        "expected_positive_transactions": len(positive_ids),
        "predicted_positive_transactions": len(predicted_ids),
        "true_positive_transactions": true_positive,
        "false_positive_transactions": false_positive,
        "false_negative_transactions": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive / negative_count if negative_count else 0.0,
    }


def evaluate_scenario_evidence(
    alerts: Iterable[StatisticalAlert],
    scenario_transactions: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    split: str,
    label_column: str = "is_anomaly",
) -> dict[str, Any]:
    """Evaluate whether alert evidence touches labelled scenario transactions."""

    required_membership = {"scenario_id", "transaction_id"}
    if required_membership - set(scenario_transactions.columns):
        raise ValueError(f"scenario transaction table needs {sorted(required_membership)}")
    if not {"scenario_id", "split", label_column} <= set(scenarios.columns):
        raise ValueError("scenario table is missing identity, split or label columns")
    target = scenarios.loc[scenarios["split"].eq(split), ["scenario_id", label_column]].copy()
    labels = pd.to_numeric(target[label_column], errors="coerce")
    known = target.loc[labels.isin([0, 1])].copy()
    known[label_column] = labels.loc[known.index].astype(int)
    known_ids = set(known["scenario_id"].astype(str))
    positive_ids = set(known.loc[known[label_column].eq(1), "scenario_id"].astype(str))
    evidence_ids = {
        transaction_id
        for alert in alerts if alert.split == split
        for transaction_id in alert.transaction_ids
    }
    predicted_ids = set(
        scenario_transactions.loc[
            scenario_transactions["transaction_id"].astype(str).isin(evidence_ids), "scenario_id"
        ].astype(str)
    ) & known_ids
    true_positive = len(predicted_ids & positive_ids)
    false_positive = len(predicted_ids - positive_ids)
    false_negative = len(positive_ids - predicted_ids)
    negative_count = len(known_ids) - len(positive_ids)
    precision = true_positive / len(predicted_ids) if predicted_ids else 0.0
    recall = true_positive / len(positive_ids) if positive_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "detector": "STATISTICAL_BASELINE",
        "split": split,
        "label_column": label_column,
        "evaluation_unit": "scenario",
        "known_scenarios": len(known_ids),
        "expected_positive_scenarios": len(positive_ids),
        "predicted_positive_scenarios": len(predicted_ids),
        "true_positive_scenarios": true_positive,
        "false_positive_scenarios": false_positive,
        "false_negative_scenarios": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive / negative_count if negative_count else 0.0,
    }
