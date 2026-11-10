#!/usr/bin/env python3
"""Generate the deterministic FinReason Synthetic Financial Crime Dataset.

The generator uses only the Python standard library. All monetary values are
generated as integer cents and written with two decimal places.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 20250713
VERSION = "1.0.0"
RELEASE_TIMESTAMP = "2026-07-29T00:00:00Z"
AS_OF = datetime(2025, 12, 31, 23, 59, tzinfo=timezone.utc)
SPLIT_RANGES = {
    "train": (datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 6, 30, 23, 59, tzinfo=timezone.utc)),
    "validation": (datetime(2025, 7, 1, tzinfo=timezone.utc), datetime(2025, 8, 31, 23, 59, tzinfo=timezone.utc)),
    "test": (datetime(2025, 9, 1, tzinfo=timezone.utc), AS_OF),
}
SPLIT_WEIGHTS = ("train", "train", "train", "validation", "test", "test")
CURRENCIES = ("GBP", "USD", "EUR", "LKR", "SGD", "AED")
COUNTRIES = ("GB", "GB", "GB", "US", "DE", "FR", "NL", "LK", "SG", "AE", "XZ", "XY")
ELEVATED_TEST_COUNTRIES = {"XZ", "XY"}  # Fictional codes, not real jurisdiction ratings.
PAYMENT_FORMATS = ("BANK_TRANSFER", "CARD", "DIRECT_DEBIT", "MOBILE", "CASH", "CHEQUE")


def money(cents: int) -> str:
    return f"{cents / 100:.2f}"


def iso(value: datetime | None) -> str:
    return "" if value is None else value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_for(value: datetime) -> str:
    for name, (start, end) in SPLIT_RANGES.items():
        if start <= value <= end:
            return name
    raise ValueError(f"Timestamp outside split ranges: {value}")


class Builder:
    def __init__(self, seed: int, account_count: int, normal_transactions: int, normal_invoices: int):
        self.rng = random.Random(seed)
        self.seed = seed
        self.account_count = account_count
        self.normal_transactions = normal_transactions
        self.normal_invoices = normal_invoices
        self.accounts: list[dict] = []
        self.account_by_id: dict[str, dict] = {}
        self.transactions: list[dict] = []
        self.invoices: list[dict] = []
        self.scenarios: list[dict] = []
        self.scenario_transactions: list[dict] = []
        self.scenario_invoices: list[dict] = []
        self.expected_alerts: list[dict] = []
        self.expected_rule_firings: list[dict] = []
        self.tx_counter = 0
        self.invoice_counter = 0
        self.scenario_counter = 0
        self.alert_counter = 0

    def random_time(self, split: str | None = None, margin_days: int = 0) -> datetime:
        split = split or self.rng.choice(SPLIT_WEIGHTS)
        start, end = SPLIT_RANGES[split]
        start += timedelta(days=margin_days)
        end -= timedelta(days=margin_days)
        seconds = max(1, int((end - start).total_seconds()))
        return start + timedelta(seconds=self.rng.randrange(seconds))

    def pick_accounts(self, count: int, *, pep: bool | None = None, new: bool | None = None) -> list[dict]:
        pool = self.accounts
        if pep is not None:
            pool = [a for a in pool if bool(a["is_pep"]) is pep]
        if new is not None:
            cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
            pool = [a for a in pool if (datetime.fromisoformat(a["opened_at"].replace("Z", "+00:00")) >= cutoff) is new]
        return self.rng.sample(pool, count)

    def create_accounts(self) -> None:
        account_types = ("INDIVIDUAL", "INDIVIDUAL", "INDIVIDUAL", "BUSINESS", "MERCHANT", "PAYROLL", "CHARITY")
        industries = ("PERSONAL", "RETAIL", "TECHNOLOGY", "HOSPITALITY", "LOGISTICS", "CONSTRUCTION", "FINANCIAL_SERVICES", "NON_PROFIT")
        for index in range(1, self.account_count + 1):
            account_type = self.rng.choice(account_types)
            country = self.rng.choice(COUNTRIES)
            currency = {"GB": "GBP", "US": "USD", "DE": "EUR", "FR": "EUR", "NL": "EUR", "LK": "LKR", "SG": "SGD", "AE": "AED"}.get(country, "USD")
            if 21 <= index <= 70:
                opened = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=self.rng.randrange(31))
            else:
                opened = datetime(2015, 1, 1, tzinfo=timezone.utc) + timedelta(days=self.rng.randrange(3650))
            is_pep = 1 if index <= max(20, self.account_count // 100) else 0
            initial = self.rng.randint(2_000_000_00, 8_000_000_00) if account_type == "INDIVIDUAL" else self.rng.randint(20_000_000_00, 80_000_000_00)
            row = {
                "account_id": f"ACC-{index:06d}",
                "account_type": account_type,
                "country": country,
                "base_currency": currency,
                "opened_at": iso(opened),
                "risk_band": "ELEVATED_TEST" if country in ELEVATED_TEST_COUNTRIES else self.rng.choice(("LOW", "LOW", "MEDIUM")),
                "is_pep": is_pep,
                "industry": "PERSONAL" if account_type == "INDIVIDUAL" else self.rng.choice(industries[1:]),
                "expected_monthly_volume": money(self.rng.randint(50_000_00, 300_000_00) if account_type == "INDIVIDUAL" else self.rng.randint(500_000_00, 5_000_000_00)),
                "opening_balance": money(initial),
                "is_synthetic": 1,
            }
            self.accounts.append(row)
            self.account_by_id[row["account_id"]] = row

    def add_scenario(self, scenario_type: str, anomaly: int, laundering: int, anchor: datetime,
                     difficulty: str, description: str, expected_detectors: str) -> str:
        self.scenario_counter += 1
        scenario_id = f"SCN-{self.scenario_counter:05d}"
        self.scenarios.append({
            "scenario_id": scenario_id, "scenario_type": scenario_type,
            "is_anomaly": anomaly, "is_laundering": laundering,
            "start_timestamp": iso(anchor), "end_timestamp": iso(anchor),
            "split": split_for(anchor), "difficulty": difficulty,
            "expected_detectors": expected_detectors, "description": description,
            "transaction_count": 0, "account_count": 0,
            "invoice_count": 0,
        })
        return scenario_id

    def update_scenario_bounds(self, scenario_id: str) -> None:
        members = [x for x in self.scenario_transactions if x["scenario_id"] == scenario_id]
        row = next(s for s in self.scenarios if s["scenario_id"] == scenario_id)
        invoice_members = [x for x in self.scenario_invoices if x["scenario_id"] == scenario_id]
        if members or invoice_members:
            tx_by_id = {t["transaction_id"]: t for t in self.transactions}
            txs = [tx_by_id[m["transaction_id"]] for m in members]
            invoice_by_id = {x["invoice_id"]: x for x in self.invoices}
            invoices = [invoice_by_id[m["invoice_id"]] for m in invoice_members]
            stamps = [datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00")) for t in txs]
            stamps += [datetime.fromisoformat(x["issue_date"] + "T00:00:00+00:00") for x in invoices]
            accounts = ({t["from_account"] for t in txs} | {t["to_account"] for t in txs} |
                        {x["supplier_account"] for x in invoices} | {x["customer_account"] for x in invoices})
            row.update(start_timestamp=iso(min(stamps)), end_timestamp=iso(max(stamps)),
                       split=split_for(min(stamps)), transaction_count=len(txs), account_count=len(accounts),
                       invoice_count=len(invoices))

    def add_transaction(self, timestamp: datetime, sender: dict, receiver: dict, amount_cents: int,
                        *, scenario_id: str = "", typology: str = "NORMAL", anomaly: int = 0,
                        laundering: int = 0, severity: str = "NONE", role: str = "BACKGROUND",
                        sequence: int = 0, payment_format: str | None = None,
                        receiving_currency: str | None = None, invoice_id: str = "",
                        transaction_type: str = "TRANSFER", description: str = "Synthetic transaction") -> str:
        self.tx_counter += 1
        sender_opened = datetime.fromisoformat(sender["opened_at"].replace("Z", "+00:00"))
        receiver_opened = datetime.fromisoformat(receiver["opened_at"].replace("Z", "+00:00"))
        timestamp = max(timestamp, sender_opened, receiver_opened)
        tx_id = f"TX-{self.tx_counter:08d}"
        payment_format = payment_format or self.rng.choice(PAYMENT_FORMATS[:-1])
        receiving_currency = receiving_currency or receiver["base_currency"]
        row = {
            "transaction_id": tx_id, "timestamp": iso(timestamp),
            "from_account": sender["account_id"], "to_account": receiver["account_id"],
            "amount": money(amount_cents), "payment_currency": sender["base_currency"],
            "receiving_currency": receiving_currency, "payment_format": payment_format,
            "transaction_type": transaction_type,
            "channel": self.rng.choice(("ONLINE", "MOBILE", "BRANCH", "API")),
            "from_country": sender["country"], "to_country": receiver["country"],
            "is_cross_border": int(sender["country"] != receiver["country"]),
            "from_balance_before": "", "from_balance_after": "",
            "to_balance_before": "", "to_balance_after": "",
            "invoice_id": invoice_id, "description": description,
            "split": split_for(timestamp), "scenario_id": scenario_id,
            "is_anomaly": anomaly, "is_laundering": laundering,
            "primary_typology": typology, "expected_severity": severity,
            "is_synthetic": 1,
        }
        self.transactions.append(row)
        if scenario_id:
            self.scenario_transactions.append({
                "scenario_id": scenario_id, "transaction_id": tx_id,
                "sequence_number": sequence, "transaction_role": role,
                "is_core_pattern_transaction": 1,
            })
        return tx_id

    def add_alert(self, scenario_id: str, detector: str, alert_type: str, severity: str,
                  root: str = "", target: str = "", min_depth: int | str = "",
                  max_depth: int | str = "", rule_ids: str = "", explanation: str = "") -> None:
        self.alert_counter += 1
        self.expected_alerts.append({
            "expected_alert_id": f"ALT-{self.alert_counter:06d}", "scenario_id": scenario_id,
            "expected_detector": detector, "alert_type": alert_type, "expected_severity": severity,
            "root_account": root, "target_account": target, "minimum_depth": min_depth,
            "maximum_depth": max_depth, "expected_rule_ids": rule_ids,
            "expected_explanation_contains": explanation,
        })

    def add_firings(self, scenario_id: str, rule_ids: list[str]) -> None:
        for order, rule_id in enumerate(rule_ids, 1):
            self.expected_rule_firings.append({
                "scenario_id": scenario_id, "rule_id": rule_id,
                "expected_fire": 1, "expected_order": order,
            })

    def normal_background(self) -> None:
        for _ in range(self.normal_transactions):
            sender, receiver = self.pick_accounts(2)
            amount = min(250_000_00, max(500, int(self.rng.lognormvariate(8.1, 1.15) * 100)))
            self.add_transaction(self.random_time(), sender, receiver, amount)

    def graph_and_transaction_scenarios(self) -> None:
        plans = [
            ("STRUCTURING", 110), ("FAN_OUT", 85), ("FAN_IN", 85), ("LAYERING_CHAIN", 85),
            ("CIRCULAR_FLOW", 70), ("RAPID_PASS_THROUGH", 85), ("COMBINED_FANOUT_FANIN", 55),
            ("LARGE_FOREIGN_CURRENCY", 50), ("HIGH_FREQUENCY", 50),
            ("ELEVATED_RISK_ROUTE", 25), ("HIGH_VALUE_CASH", 25), ("UNUSUAL_HOUR", 25),
            ("NEW_ACCOUNT_HIGH_VALUE", 25), ("PEP_HIGH_VALUE", 25), ("ROUND_AMOUNT_BURST", 25),
            ("RAPID_BALANCE_DRAIN", 25), ("CROSS_BORDER_LAYERING", 25),
            ("PEP_GRAPH_PATTERN", 20), ("STRUCTURING_FANOUT", 20),
            ("MULTI_INDICATOR", 20), ("CRITICAL_CASE", 15),
        ]
        for kind, count in plans:
            for _ in range(count):
                self.build_transaction_scenario(kind)
        for kind, count in (("LEGIT_PAYROLL_FANOUT", 55), ("LEGIT_MERCHANT_FANIN", 55),
                            ("LEGIT_HIGH_VALUE", 55), ("LEGIT_NEAR_THRESHOLD", 55),
                            ("LEGIT_SUPPLY_CHAIN", 45)):
            for _ in range(count):
                self.build_hard_negative(kind)

    def build_transaction_scenario(self, kind: str) -> None:
        anchor = self.random_time(margin_days=8)
        difficulty = self.rng.choice(("EASY", "MEDIUM", "HARD"))
        sid = self.add_scenario(kind, 1, int(kind not in {"UNUSUAL_HOUR", "NEW_ACCOUNT_HIGH_VALUE", "ROUND_AMOUNT_BURST"}),
                                anchor, difficulty, f"Synthetic {kind.lower().replace('_', ' ')} case.", "RULE_ENGINE")
        accounts = self.pick_accounts(12)
        root, target = accounts[0], accounts[1]

        if kind == "STRUCTURING":
            for i in range(self.rng.randint(4, 7)):
                self.add_transaction(anchor + timedelta(hours=i * 16), root, target, self.rng.randint(900_000, 995_000), scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="HIGH", role="STRUCTURED_PAYMENT", sequence=i + 1)
            self.add_alert(sid, "RULE_ENGINE", kind, "HIGH", root["account_id"], target["account_id"], rule_ids="R005|R006", explanation="structuring")
            self.add_firings(sid, ["R005", "R006"])
        elif kind in {"FAN_OUT", "PEP_GRAPH_PATTERN"}:
            if kind == "PEP_GRAPH_PATTERN":
                root = self.pick_accounts(1, pep=True)[0]
            recipients = accounts[2:2 + self.rng.randint(4, 7)]
            for i, receiver in enumerate(recipients):
                self.add_transaction(anchor + timedelta(minutes=15 * i), root, receiver, self.rng.randint(120_000, 750_000), scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="HIGH", role="FAN_OUT_EDGE", sequence=i + 1)
            rules = ["R014"] if kind == "FAN_OUT" else ["R014", "R021", "R023"]
            self.add_alert(sid, "BFS", "FAN_OUT", "HIGH", root["account_id"], min_depth=1, max_depth=1, rule_ids="|".join(rules), explanation="fan-out")
            self.add_firings(sid, rules)
        elif kind == "FAN_IN":
            senders = accounts[2:2 + self.rng.randint(4, 7)]
            for i, sender in enumerate(senders):
                self.add_transaction(anchor + timedelta(minutes=18 * i), sender, target, self.rng.randint(110_000, 800_000), scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="HIGH", role="FAN_IN_EDGE", sequence=i + 1)
            self.add_alert(sid, "BFS_REVERSE", kind, "HIGH", target=target["account_id"], min_depth=1, max_depth=1, rule_ids="R015", explanation="fan-in")
            self.add_firings(sid, ["R015"])
        elif kind in {"LAYERING_CHAIN", "CROSS_BORDER_LAYERING"}:
            path = accounts[:self.rng.randint(5, 7)]
            if kind == "CROSS_BORDER_LAYERING":
                path[0] = next(a for a in self.accounts if a["country"] == "GB")
                path[1] = next(a for a in self.accounts if a["country"] == "US" and a not in path)
            amount = self.rng.randint(2_000_000, 12_000_000)
            for i in range(len(path) - 1):
                self.add_transaction(anchor + timedelta(minutes=40 * i), path[i], path[i + 1], amount, scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="HIGH", role="LAYERING_EDGE", sequence=i + 1)
                amount = int(amount * self.rng.uniform(0.97, 0.995))
            rules = ["R016"] if kind == "LAYERING_CHAIN" else ["R002", "R016", "R020", "R023"]
            self.add_alert(sid, "DFS", "LAYERING_CHAIN", "HIGH", path[0]["account_id"], path[-1]["account_id"], len(path) - 1, len(path) - 1, "|".join(rules), "layering")
            self.add_firings(sid, rules)
        elif kind == "CIRCULAR_FLOW":
            path = accounts[:self.rng.randint(3, 6)] + [accounts[0]]
            amount = self.rng.randint(1_500_000, 10_000_000)
            for i in range(len(path) - 1):
                self.add_transaction(anchor + timedelta(minutes=35 * i), path[i], path[i + 1], amount, scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="CRITICAL", role="CYCLE_EDGE", sequence=i + 1)
                amount = int(amount * 0.99)
            self.add_alert(sid, "DFS", kind, "CRITICAL", root["account_id"], root["account_id"], len(path) - 1, len(path) - 1, "R017", "circular")
            self.add_firings(sid, ["R017"])
        elif kind == "RAPID_PASS_THROUGH":
            amount = self.rng.randint(2_000_000, 8_000_000)
            self.add_transaction(anchor, accounts[0], accounts[1], amount, scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="HIGH", role="INBOUND", sequence=1)
            self.add_transaction(anchor + timedelta(minutes=self.rng.randint(5, 25)), accounts[1], accounts[2], int(amount * 0.985), scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="HIGH", role="OUTBOUND", sequence=2)
            self.add_alert(sid, "GRAPH_TEMPORAL", kind, "HIGH", accounts[1]["account_id"], accounts[2]["account_id"], 2, 2, "R018", "rapid pass-through")
            self.add_firings(sid, ["R018"])
        elif kind in {"COMBINED_FANOUT_FANIN", "STRUCTURING_FANOUT", "MULTI_INDICATOR", "CRITICAL_CASE"}:
            branches = accounts[2:6]
            hub, exit_account, final_exit = accounts[6], accounts[7], accounts[8]
            amount = 950_000 if kind == "STRUCTURING_FANOUT" else self.rng.randint(1_500_000, 4_000_000)
            seq = 0
            for branch in branches:
                seq += 1
                self.add_transaction(anchor + timedelta(minutes=seq * 12), root, branch, amount, scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="CRITICAL", role="FAN_OUT_EDGE", sequence=seq)
            for branch in branches:
                seq += 1
                self.add_transaction(anchor + timedelta(minutes=seq * 12), branch, hub, int(amount * 0.98), scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="CRITICAL", role="FAN_IN_EDGE", sequence=seq)
            seq += 1
            self.add_transaction(anchor + timedelta(minutes=seq * 12), hub, exit_account, int(amount * 2.85), scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="CRITICAL", role="EXIT_EDGE", sequence=seq)
            seq += 1
            self.add_transaction(anchor + timedelta(minutes=seq * 12), exit_account, final_exit, int(amount * 2.80), scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="CRITICAL", role="FINAL_EXIT_EDGE", sequence=seq)
            rules = ["R014", "R015", "R016", "R019"]
            if kind == "STRUCTURING_FANOUT": rules += ["R005", "R022", "R023"]
            if kind in {"MULTI_INDICATOR", "CRITICAL_CASE"}: rules += ["R023"]
            if kind == "CRITICAL_CASE": rules += ["R033"]
            self.add_alert(sid, "HYBRID", kind, "CRITICAL", root["account_id"], final_exit["account_id"], 1, 4, "|".join(rules), "combined")
            self.add_firings(sid, rules)
        elif kind == "HIGH_FREQUENCY":
            for i in range(self.rng.randint(12, 18)):
                self.add_transaction(anchor + timedelta(minutes=i * 30), root, accounts[1 + (i % 8)], self.rng.randint(10_000, 90_000), scenario_id=sid, typology=kind, anomaly=1, laundering=1, severity="MEDIUM", role="BURST_PAYMENT", sequence=i + 1)
            self.add_alert(sid, "RULE_ENGINE", kind, "MEDIUM", root["account_id"], rule_ids="R007", explanation="frequency")
            self.add_firings(sid, ["R007"])
        else:
            self.build_single_transaction_case(kind, sid, anchor, accounts)
        self.update_scenario_bounds(sid)

    def build_single_transaction_case(self, kind: str, sid: str, anchor: datetime, accounts: list[dict]) -> None:
        sender, receiver = accounts[0], accounts[1]
        amount, fmt, rules, severity = 12_500_000, "BANK_TRANSFER", [], "HIGH"
        if kind == "LARGE_FOREIGN_CURRENCY":
            receiver = next(a for a in self.accounts if a["country"] != sender["country"] and a["base_currency"] != sender["base_currency"])
            rules = ["R001", "R002", "R011"]
        elif kind == "ELEVATED_RISK_ROUTE":
            receiver = next(a for a in self.accounts if a["country"] in ELEVATED_TEST_COUNTRIES)
            rules = ["R002", "R003", "R023"]
        elif kind == "HIGH_VALUE_CASH":
            fmt, rules = "CASH", ["R001", "R004", "R023"]
        elif kind == "UNUSUAL_HOUR":
            anchor = anchor.replace(hour=2, minute=17); amount, rules, severity = 600_000, ["R008"], "MEDIUM"
        elif kind == "NEW_ACCOUNT_HIGH_VALUE":
            sender = self.pick_accounts(1, new=True)[0]
            anchor = datetime(2025, 3, 1, tzinfo=timezone.utc) + timedelta(days=self.rng.randrange(75), minutes=self.rng.randrange(1440))
            rules = ["R001", "R009", "R023"]
        elif kind == "PEP_HIGH_VALUE":
            sender = self.pick_accounts(1, pep=True)[0]; rules = ["R001", "R010", "R023"]
        elif kind == "ROUND_AMOUNT_BURST":
            amount, rules, severity = 5_000_000, ["R012"], "MEDIUM"
        elif kind == "RAPID_BALANCE_DRAIN":
            amount, rules = 10_000_000_000, ["R001", "R012", "R013", "R023"]
        else:
            raise ValueError(kind)
        self.add_transaction(anchor, sender, receiver, amount, scenario_id=sid, typology=kind, anomaly=1,
                             laundering=int(kind not in {"UNUSUAL_HOUR", "NEW_ACCOUNT_HIGH_VALUE", "ROUND_AMOUNT_BURST"}),
                             severity=severity, role="TRIGGER_TRANSACTION", sequence=1, payment_format=fmt,
                             receiving_currency=receiver["base_currency"], description=f"Synthetic {kind} trigger")
        self.add_alert(sid, "RULE_ENGINE", kind, severity, sender["account_id"], receiver["account_id"], rule_ids="|".join(rules), explanation=kind.lower().replace("_", " "))
        self.add_firings(sid, rules)

    def build_hard_negative(self, kind: str) -> None:
        anchor = self.random_time(margin_days=8)
        sid = self.add_scenario(kind, 0, 0, anchor, "HARD", f"Legitimate look-alike: {kind.lower().replace('_', ' ')}.", "NONE")
        accounts = self.pick_accounts(15)
        if kind == "LEGIT_PAYROLL_FANOUT":
            root = next((a for a in self.accounts if a["account_type"] == "PAYROLL"), accounts[0])
            for i, receiver in enumerate(accounts[1:10], 1):
                self.add_transaction(anchor + timedelta(minutes=5 * i), root, receiver, self.rng.randint(180_000, 600_000), scenario_id=sid, typology=kind, role="LEGIT_PAYROLL", sequence=i, transaction_type="PAYROLL", description="Scheduled payroll payment")
        elif kind == "LEGIT_MERCHANT_FANIN":
            target = next((a for a in self.accounts if a["account_type"] == "MERCHANT"), accounts[0])
            for i, sender in enumerate(accounts[1:10], 1):
                self.add_transaction(anchor + timedelta(minutes=8 * i), sender, target, self.rng.randint(2_000, 80_000), scenario_id=sid, typology=kind, role="LEGIT_SALE", sequence=i, transaction_type="PURCHASE", description="Retail customer payment")
        elif kind == "LEGIT_HIGH_VALUE":
            self.add_transaction(anchor, accounts[0], accounts[1], self.rng.randint(15_000_000, 80_000_000), scenario_id=sid, typology=kind, role="LEGIT_HIGH_VALUE", sequence=1, description="Documented business equipment purchase")
        elif kind == "LEGIT_NEAR_THRESHOLD":
            self.add_transaction(anchor, accounts[0], accounts[1], self.rng.randint(900_000, 995_000), scenario_id=sid, typology=kind, role="LEGIT_SINGLE_PAYMENT", sequence=1, description="Single documented invoice payment")
        elif kind == "LEGIT_SUPPLY_CHAIN":
            path = accounts[:6]
            for i in range(5):
                self.add_transaction(anchor + timedelta(days=i), path[i], path[i + 1], self.rng.randint(2_000_000, 7_000_000), scenario_id=sid, typology=kind, role="LEGIT_SUPPLY_PAYMENT", sequence=i + 1, transaction_type="INVOICE_PAYMENT", description="Documented supply-chain invoice")
        self.update_scenario_bounds(sid)

    def add_invoice(self, issue: datetime, supplier: dict, customer: dict, net_cents: int, vat_rate: int,
                    *, scenario_id: str = "", external_number: str | None = None, due_days: int = 30,
                    paid_at: datetime | None = None, vat_override: int | None = None,
                    payment_override: int | None = None, anomaly: int = 0, anomaly_type: str = "NORMAL") -> str:
        self.invoice_counter += 1
        iid = f"INV-{self.invoice_counter:07d}"
        expected_vat = int(round(net_cents * vat_rate / 100))
        vat = expected_vat if vat_override is None else vat_override
        gross = net_cents + vat
        due = issue + timedelta(days=due_days)
        days_overdue = max(0, (AS_OF.date() - due.date()).days) if paid_at is None else max(0, (paid_at.date() - due.date()).days)
        status = "UNPAID" if paid_at is None else ("PAID_LATE" if paid_at > due else "PAID_ON_TIME")
        row = {
            "invoice_id": iid, "external_invoice_number": external_number or f"EXT-{self.invoice_counter:07d}",
            "issue_date": issue.date().isoformat(), "due_date": due.date().isoformat(),
            "paid_at": iso(paid_at), "supplier_account": supplier["account_id"],
            "customer_account": customer["account_id"], "net_amount": money(net_cents),
            "vat_rate": vat_rate, "expected_vat_amount": money(expected_vat),
            "reported_vat_amount": money(vat), "gross_amount": money(gross),
            "invoice_currency": supplier["base_currency"], "status": status,
            "is_overdue_as_of_2025_12_31": int(paid_at is None and due < AS_OF),
            "days_overdue": days_overdue, "is_vat_anomaly": int(vat != expected_vat),
            "is_anomaly": anomaly, "anomaly_type": anomaly_type,
            "scenario_id": scenario_id, "split": split_for(issue), "is_synthetic": 1,
        }
        self.invoices.append(row)
        if scenario_id:
            self.scenario_invoices.append({
                "scenario_id": scenario_id, "invoice_id": iid,
                "invoice_role": "CORE_INVOICE", "is_core_pattern_invoice": 1,
            })
        if paid_at is not None:
            payment = gross if payment_override is None else payment_override
            self.add_transaction(paid_at, customer, supplier, payment, scenario_id=scenario_id,
                                 typology=anomaly_type, anomaly=anomaly, laundering=0,
                                 severity="HIGH" if anomaly else "NONE", role="INVOICE_PAYMENT",
                                 sequence=1, invoice_id=iid, transaction_type="INVOICE_PAYMENT",
                                 description="Payment linked to synthetic invoice")
        return iid

    def invoice_scenarios(self) -> None:
        for _ in range(self.normal_invoices):
            supplier, customer = self.pick_accounts(2)
            issue = self.random_time()
            due_days = self.rng.choice((14, 30, 45, 60))
            paid = issue + timedelta(days=self.rng.randint(1, max(2, due_days - 1))) if self.rng.random() < 0.85 else None
            if paid is not None and paid > AS_OF:
                paid = None
            if paid is None and issue + timedelta(days=due_days) < AS_OF:
                paid = issue + timedelta(days=max(1, due_days - 1))
            self.add_invoice(issue, supplier, customer, self.rng.randint(20_000, 20_000_000), self.rng.choice((0, 5, 10, 20)), due_days=due_days, paid_at=paid)

        invoice_plans = (("OVERDUE_30", 50), ("OVERDUE_60", 45), ("OVERDUE_90", 45),
                         ("VAT_AMOUNT_MISMATCH", 45), ("INVALID_VAT_RATE", 35),
                         ("DUPLICATE_INVOICE", 30), ("PAYMENT_EXCEEDS_GROSS", 35),
                         ("FOREIGN_CURRENCY_INVOICE", 35), ("HIGH_VALUE_OVERDUE", 35))
        for kind, count in invoice_plans:
            for _ in range(count):
                self.build_invoice_scenario(kind)

    def build_invoice_scenario(self, kind: str) -> None:
        split = self.rng.choice(SPLIT_WEIGHTS)
        anchor = self.random_time(split, margin_days=5)
        sid = self.add_scenario(kind, 1, 0, anchor, "MEDIUM", f"Synthetic invoice rule case: {kind}.", "RULE_ENGINE")
        supplier, customer = self.pick_accounts(2)
        rules: list[str]
        if kind.startswith("OVERDUE") or kind == "HIGH_VALUE_OVERDUE":
            days = {"OVERDUE_30": 40, "OVERDUE_60": 70, "OVERDUE_90": 110, "HIGH_VALUE_OVERDUE": 120}[kind]
            issue = AS_OF - timedelta(days=days + 30)
            net = 80_000_000 if kind == "HIGH_VALUE_OVERDUE" else self.rng.randint(100_000, 8_000_000)
            self.add_invoice(issue, supplier, customer, net, 20, scenario_id=sid, paid_at=None, anomaly=1, anomaly_type=kind)
            rules = ["R024"] + (["R025"] if days >= 60 else []) + (["R026"] if days >= 90 else []) + (["R032"] if kind == "HIGH_VALUE_OVERDUE" else [])
        elif kind == "VAT_AMOUNT_MISMATCH":
            net = self.rng.randint(100_000, 5_000_000)
            self.add_invoice(anchor, supplier, customer, net, 20, scenario_id=sid, paid_at=anchor + timedelta(days=5), vat_override=int(net * 0.13), anomaly=1, anomaly_type=kind)
            rules = ["R027"]
        elif kind == "INVALID_VAT_RATE":
            self.add_invoice(anchor, supplier, customer, self.rng.randint(100_000, 5_000_000), 17, scenario_id=sid, paid_at=anchor + timedelta(days=5), anomaly=1, anomaly_type=kind)
            rules = ["R028"]
        elif kind == "DUPLICATE_INVOICE":
            ext = f"DUP-{self.scenario_counter:05d}"
            for offset in (0, 1):
                self.add_invoice(anchor + timedelta(minutes=offset), supplier, customer, 2_000_000, 20, scenario_id=sid, external_number=ext, paid_at=anchor + timedelta(days=3 + offset), anomaly=1, anomaly_type=kind)
            rules = ["R029"]
        elif kind == "PAYMENT_EXCEEDS_GROSS":
            self.add_invoice(anchor, supplier, customer, 2_000_000, 20, scenario_id=sid, paid_at=anchor + timedelta(days=3), payment_override=3_500_000, anomaly=1, anomaly_type=kind)
            rules = ["R030"]
        elif kind == "FOREIGN_CURRENCY_INVOICE":
            customer = next(a for a in self.accounts if a["base_currency"] != supplier["base_currency"])
            self.add_invoice(anchor, supplier, customer, 5_000_000, 20, scenario_id=sid, paid_at=anchor + timedelta(days=3), anomaly=1, anomaly_type=kind)
            rules = ["R031"]
        else:
            raise ValueError(kind)
        self.add_alert(sid, "RULE_ENGINE", kind, "HIGH", supplier["account_id"], customer["account_id"], rule_ids="|".join(rules), explanation=kind.lower().replace("_", " "))
        self.add_firings(sid, rules)
        self.update_scenario_bounds(sid)

    def calculate_balances(self) -> None:
        ordered = sorted(self.transactions, key=lambda x: (x["timestamp"], x["transaction_id"]))
        deltas: dict[str, int] = defaultdict(int)
        minima: dict[str, int] = defaultdict(int)
        for tx in ordered:
            cents = int(round(float(tx["amount"]) * 100))
            deltas[tx["from_account"]] -= cents
            minima[tx["from_account"]] = min(minima[tx["from_account"]], deltas[tx["from_account"]])
            deltas[tx["to_account"]] += cents
        balances = {}
        for account in self.accounts:
            aid = account["account_id"]
            configured = int(round(float(account["opening_balance"]) * 100))
            opening = max(configured, -minima[aid] + 1_000_000)
            account["opening_balance"] = money(opening)
            balances[aid] = opening
        for tx in ordered:
            cents = int(round(float(tx["amount"]) * 100))
            sender, receiver = tx["from_account"], tx["to_account"]
            tx["from_balance_before"] = money(balances[sender]); balances[sender] -= cents
            tx["from_balance_after"] = money(balances[sender])
            tx["to_balance_before"] = money(balances[receiver]); balances[receiver] += cents
            tx["to_balance_after"] = money(balances[receiver])
        self.transactions = ordered

    def build(self) -> None:
        self.create_accounts()
        self.normal_background()
        self.graph_and_transaction_scenarios()
        self.invoice_scenarios()
        self.calculate_balances()


RULES = [
    ("R001", "High-value transaction", "TRANSACTION", 60, "amount >= 100000 GBP-equivalent", "HIGH_VALUE_TX", "MEDIUM"),
    ("R002", "Cross-border transfer", "TRANSACTION", 40, "from_country != to_country", "CROSS_BORDER_TX", "LOW"),
    ("R003", "Elevated-risk test route", "TRANSACTION", 80, "country in fictional XZ/XY test list", "ELEVATED_ROUTE", "HIGH"),
    ("R004", "High-value cash", "TRANSACTION", 90, "amount >= 100000 and payment_format = CASH", "HIGH_VALUE_CASH", "HIGH"),
    ("R005", "Near-threshold payment", "TRANSACTION", 55, "9000 <= amount < 10000", "NEAR_THRESHOLD_TX", "MEDIUM"),
    ("R006", "Seven-day structuring", "AGGREGATE", 95, ">=4 near-threshold payments in 7 days", "STRUCTURING_PATTERN", "HIGH"),
    ("R007", "High transaction frequency", "AGGREGATE", 65, ">=12 outgoing payments in 24 hours", "HIGH_FREQUENCY", "MEDIUM"),
    ("R008", "Unusual-hour transaction", "TRANSACTION", 25, "timestamp hour between 00:00 and 04:00 UTC", "UNUSUAL_HOUR", "LOW"),
    ("R009", "New-account high value", "TRANSACTION", 75, "account age < 180 days and high value", "NEW_ACCOUNT_HIGH_VALUE", "HIGH"),
    ("R010", "PEP high value", "TRANSACTION", 90, "is_pep and high value", "PEP_HIGH_VALUE", "HIGH"),
    ("R011", "Currency conversion", "TRANSACTION", 35, "payment_currency != receiving_currency", "CURRENCY_CONVERSION", "LOW"),
    ("R012", "Large round amount", "TRANSACTION", 30, "amount >= 50000 and divisible by 10000", "ROUND_AMOUNT", "LOW"),
    ("R013", "Rapid balance drain", "TRANSACTION", 70, "outgoing amount >= 80% of pre-transaction balance", "BALANCE_DRAIN", "HIGH"),
    ("R014", "Graph fan-out", "GRAPH", 80, ">=4 distinct successors in 24 hours", "FAN_OUT_PATTERN", "HIGH"),
    ("R015", "Graph fan-in", "GRAPH", 80, ">=4 distinct predecessors in 24 hours", "FAN_IN_PATTERN", "HIGH"),
    ("R016", "Temporal layering chain", "GRAPH", 90, ">=4 edges with increasing timestamps and amount retention", "LAYERING_PATTERN", "HIGH"),
    ("R017", "Temporal circular flow", "GRAPH", 100, "time-respecting path returns to origin", "CIRCULAR_FLOW", "CRITICAL"),
    ("R018", "Rapid pass-through", "GRAPH", 85, "outflow within 30 minutes retaining >=95% of inflow", "RAPID_PASS_THROUGH", "HIGH"),
    ("R019", "Combined fan-out and fan-in", "DERIVED", 95, "FAN_OUT_PATTERN and FAN_IN_PATTERN", "COMBINED_GRAPH_PATTERN", "CRITICAL"),
    ("R020", "Cross-border layering", "DERIVED", 95, "LAYERING_PATTERN and CROSS_BORDER_TX", "CROSS_BORDER_LAYERING", "CRITICAL"),
    ("R021", "PEP graph pattern", "DERIVED", 100, "is_pep and graph pattern", "PEP_GRAPH_PATTERN", "CRITICAL"),
    ("R022", "Structuring plus fan-out", "DERIVED", 95, "STRUCTURING_PATTERN and FAN_OUT_PATTERN", "STRUCTURING_FANOUT", "CRITICAL"),
    ("R023", "Multiple-indicator escalation", "DERIVED", 98, ">=2 independent risk facts", "MULTI_INDICATOR_CASE", "CRITICAL"),
    ("R024", "Receivable overdue 30 days", "INVOICE", 50, "unpaid and days_overdue >= 30", "OVERDUE_30", "MEDIUM"),
    ("R025", "Receivable overdue 60 days", "INVOICE", 65, "unpaid and days_overdue >= 60", "OVERDUE_60", "HIGH"),
    ("R026", "Receivable overdue 90 days", "INVOICE", 80, "unpaid and days_overdue >= 90", "OVERDUE_90", "CRITICAL"),
    ("R027", "VAT amount mismatch", "INVOICE", 80, "reported VAT differs from expected VAT", "VAT_MISMATCH", "HIGH"),
    ("R028", "Invalid synthetic VAT rate", "INVOICE", 70, "VAT rate not in configured educational list", "INVALID_VAT_RATE", "HIGH"),
    ("R029", "Duplicate invoice", "INVOICE", 85, "duplicate external invoice number, supplier and customer", "DUPLICATE_INVOICE", "HIGH"),
    ("R030", "Payment exceeds gross", "INVOICE", 75, "linked payment > invoice gross amount", "OVERPAYMENT", "HIGH"),
    ("R031", "Foreign-currency invoice", "INVOICE", 35, "invoice currency differs from customer base currency", "FOREIGN_CURRENCY_INVOICE", "LOW"),
    ("R032", "High-value overdue receivable", "DERIVED", 90, "OVERDUE_90 and gross amount >= 500000", "HIGH_VALUE_OVERDUE", "CRITICAL"),
    ("R033", "Critical case escalation", "DERIVED", 100, ">=3 high/critical derived facts", "CRITICAL_CASE", "CRITICAL"),
]


def rule_catalog() -> list[dict]:
    return [{"rule_id": r[0], "rule_name": r[1], "fact_category": r[2], "priority": r[3],
             "condition_summary": r[4], "derived_fact": r[5], "severity": r[6],
             "recommended_action": "Queue for human review; educational synthetic rule only."} for r in RULES]


def rule_test_cases() -> list[dict]:
    rows = []
    for rule_id, name, category, _, condition, fact, _ in RULES:
        for expected in (1, 0):
            rows.append({
                "test_case_id": f"RTC-{rule_id}-{expected}", "rule_id": rule_id,
                "expected_fire": expected,
                "input_facts_json": json.dumps({"synthetic_case": True, "condition": condition, "condition_satisfied": bool(expected)}, sort_keys=True),
                "expected_derived_fact": fact if expected else "",
                "notes": f"{'Positive' if expected else 'Negative'} unit case for {name}.",
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--accounts", type=int, default=5000)
    parser.add_argument("--normal-transactions", type=int, default=60000)
    parser.add_argument("--normal-invoices", type=int, default=1500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    builder = Builder(args.seed, args.accounts, args.normal_transactions, args.normal_invoices)
    builder.build()
    tables = {
        "accounts.csv": builder.accounts,
        "transactions.csv": builder.transactions,
        "invoices.csv": builder.invoices,
        "scenarios.csv": builder.scenarios,
        "scenario_transactions.csv": builder.scenario_transactions,
        "scenario_invoices.csv": builder.scenario_invoices,
        "expected_alerts.csv": builder.expected_alerts,
        "expected_rule_firings.csv": builder.expected_rule_firings,
        "rule_catalog.csv": rule_catalog(),
        "rule_test_cases.csv": rule_test_cases(),
    }
    for filename, rows in tables.items():
        write_csv(args.output / filename, rows)

    metadata = {
        "dataset_name": "FinReason Synthetic Financial Crime Dataset",
        "version": VERSION, "seed": args.seed, "release_timestamp_utc": RELEASE_TIMESTAMP,
        "observation_period": ["2025-01-01T00:00:00Z", "2025-12-31T23:59:00Z"],
        "as_of_date_for_receivables": "2025-12-31",
        "license": "CC BY 4.0", "synthetic": True,
        "educational_use_notice": "All persons, accounts, jurisdictions and transactions are synthetic. Thresholds are coursework assumptions, not legal advice.",
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "class_counts": {
            "transaction_anomaly_positive": sum(int(t["is_anomaly"]) for t in builder.transactions),
            "transaction_laundering_positive": sum(int(t["is_laundering"]) for t in builder.transactions),
            "scenario_anomaly_positive": sum(int(s["is_anomaly"]) for s in builder.scenarios),
            "hard_negative_scenarios": sum(int(not s["is_anomaly"]) for s in builder.scenarios),
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in args.output.iterdir() if p.is_file() and p.name != "checksums.sha256")
    (args.output / "checksums.sha256").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
    print(json.dumps(metadata["row_counts"], indent=2))


if __name__ == "__main__":
    main()
