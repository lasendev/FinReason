#!/usr/bin/env python3
"""Validate structural, financial, temporal and ground-truth invariants."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def dec(text: str) -> Decimal:
    return Decimal(text)


class Checks:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def check(self, name: str, condition: bool, evidence: str) -> None:
        self.results.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})

    @property
    def failures(self) -> list[dict]:
        return [r for r in self.results if r["status"] == "FAIL"]


def validate(data_dir: Path) -> dict:
    expected_files = [
        "accounts.csv", "transactions.csv", "invoices.csv", "scenarios.csv",
        "scenario_transactions.csv", "scenario_invoices.csv", "expected_alerts.csv",
        "expected_rule_firings.csv", "rule_catalog.csv", "rule_test_cases.csv", "metadata.json",
    ]
    checks = Checks()
    missing = [name for name in expected_files if not (data_dir / name).exists()]
    checks.check("required_files", not missing, f"missing={missing}")
    if missing:
        return {"status": "FAIL", "checks": checks.results}

    tables = {name: load_csv(data_dir / name) for name in expected_files if name.endswith(".csv")}
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    accounts, txs, invoices, scenarios = (tables["accounts.csv"], tables["transactions.csv"], tables["invoices.csv"], tables["scenarios.csv"])
    members, invoice_members = tables["scenario_transactions.csv"], tables["scenario_invoices.csv"]
    alerts, firings, rules, rule_cases = tables["expected_alerts.csv"], tables["expected_rule_firings.csv"], tables["rule_catalog.csv"], tables["rule_test_cases.csv"]

    required_columns = {
        "accounts.csv": {"account_id", "country", "base_currency", "opened_at", "opening_balance"},
        "transactions.csv": {"transaction_id", "timestamp", "from_account", "to_account", "amount", "split", "scenario_id", "is_anomaly", "is_laundering"},
        "invoices.csv": {"invoice_id", "issue_date", "due_date", "gross_amount", "scenario_id", "is_anomaly"},
        "scenarios.csv": {"scenario_id", "scenario_type", "is_anomaly", "split", "transaction_count", "invoice_count"},
    }
    for filename, columns in required_columns.items():
        actual = set(tables[filename][0])
        checks.check(f"columns_{filename}", columns <= actual, f"missing={sorted(columns - actual)}")

    keys = {"accounts.csv": "account_id", "transactions.csv": "transaction_id", "invoices.csv": "invoice_id", "scenarios.csv": "scenario_id", "rule_catalog.csv": "rule_id", "rule_test_cases.csv": "test_case_id"}
    for filename, key in keys.items():
        values = [r[key] for r in tables[filename]]
        checks.check(f"unique_{key}", len(values) == len(set(values)) and all(values), f"rows={len(values)}, distinct={len(set(values))}")

    for filename, rows in tables.items():
        expected = metadata["row_counts"].get(filename)
        checks.check(f"metadata_count_{filename}", expected == len(rows), f"metadata={expected}, actual={len(rows)}")

    account_ids = {r["account_id"] for r in accounts}
    tx_ids = {r["transaction_id"] for r in txs}
    invoice_ids = {r["invoice_id"] for r in invoices}
    scenario_ids = {r["scenario_id"] for r in scenarios}
    rule_ids = {r["rule_id"] for r in rules}
    checks.check("transaction_account_fks", all(t["from_account"] in account_ids and t["to_account"] in account_ids for t in txs), f"transactions={len(txs)}")
    checks.check("invoice_account_fks", all(i["supplier_account"] in account_ids and i["customer_account"] in account_ids for i in invoices), f"invoices={len(invoices)}")
    checks.check("scenario_transaction_fks", all(m["scenario_id"] in scenario_ids and m["transaction_id"] in tx_ids for m in members), f"members={len(members)}")
    checks.check("scenario_invoice_fks", all(m["scenario_id"] in scenario_ids and m["invoice_id"] in invoice_ids for m in invoice_members), f"members={len(invoice_members)}")
    checks.check("alert_scenario_fks", all(a["scenario_id"] in scenario_ids for a in alerts), f"alerts={len(alerts)}")
    checks.check("firing_fks", all(f["scenario_id"] in scenario_ids and f["rule_id"] in rule_ids for f in firings), f"firings={len(firings)}")

    split_ranges = {
        "train": (dt("2025-01-01T00:00:00Z"), dt("2025-06-30T23:59:59Z")),
        "validation": (dt("2025-07-01T00:00:00Z"), dt("2025-08-31T23:59:59Z")),
        "test": (dt("2025-09-01T00:00:00Z"), dt("2025-12-31T23:59:59Z")),
    }
    time_ok = all(t["split"] in split_ranges and split_ranges[t["split"]][0] <= dt(t["timestamp"]) <= split_ranges[t["split"]][1] for t in txs)
    checks.check("transaction_time_splits", time_ok, str(Counter(t["split"] for t in txs)))
    opened = {a["account_id"]: dt(a["opened_at"]) for a in accounts}
    checks.check("transactions_after_account_open", all(dt(t["timestamp"]) >= opened[t["from_account"]] and dt(t["timestamp"]) >= opened[t["to_account"]] for t in txs), "all transaction timestamps are on/after both account opening timestamps")
    checks.check("positive_amounts", all(dec(t["amount"]) > 0 for t in txs), f"minimum={min(dec(t['amount']) for t in txs)}")
    checks.check("binary_labels", all(t["is_anomaly"] in {"0", "1"} and t["is_laundering"] in {"0", "1"} for t in txs), "allowed={0,1}")
    checks.check("laundering_implies_anomaly", all(t["is_laundering"] == "0" or t["is_anomaly"] == "1" for t in txs), "no laundering-positive transaction is anomaly-negative")
    checks.check("cross_border_consistency", all(int(t["is_cross_border"]) == int(t["from_country"] != t["to_country"]) for t in txs), "derived flag agrees with countries")

    # Replay the complete ledger and validate all four balance columns.
    balances = {a["account_id"]: dec(a["opening_balance"]) for a in accounts}
    balance_errors = 0
    for tx in sorted(txs, key=lambda x: (x["timestamp"], x["transaction_id"])):
        amount, sender, receiver = dec(tx["amount"]), tx["from_account"], tx["to_account"]
        expected_sender_after = balances[sender] - amount
        expected_receiver_after = balances[receiver] + amount
        if (dec(tx["from_balance_before"]) != balances[sender] or dec(tx["from_balance_after"]) != expected_sender_after or
                dec(tx["to_balance_before"]) != balances[receiver] or dec(tx["to_balance_after"]) != expected_receiver_after):
            balance_errors += 1
        balances[sender], balances[receiver] = expected_sender_after, expected_receiver_after
    checks.check("ledger_balance_reconciliation", balance_errors == 0, f"errors={balance_errors}")
    checks.check("non_negative_balances", all(value >= 0 for value in balances.values()) and all(dec(t["from_balance_after"]) >= 0 for t in txs), f"minimum_final={min(balances.values())}")

    member_counts = Counter(m["scenario_id"] for m in members)
    invoice_member_counts = Counter(m["scenario_id"] for m in invoice_members)
    count_errors = [s["scenario_id"] for s in scenarios if int(s["transaction_count"]) != member_counts[s["scenario_id"]] or int(s["invoice_count"]) != invoice_member_counts[s["scenario_id"]]]
    checks.check("scenario_member_counts", not count_errors, f"mismatches={len(count_errors)}")
    positive_scenarios = {s["scenario_id"] for s in scenarios if s["is_anomaly"] == "1"}
    negative_scenarios = scenario_ids - positive_scenarios
    alerted = {a["scenario_id"] for a in alerts}
    checks.check("positive_scenario_alert_coverage", positive_scenarios <= alerted, f"positive={len(positive_scenarios)}, alerted={len(alerted & positive_scenarios)}")
    checks.check("hard_negative_no_expected_alert", not (negative_scenarios & alerted), f"hard_negatives={len(negative_scenarios)}, incorrectly_alerted={len(negative_scenarios & alerted)}")

    case_counts = Counter(c["rule_id"] for c in rule_cases)
    case_polarities = defaultdict(set)
    for c in rule_cases:
        case_polarities[c["rule_id"]].add(c["expected_fire"])
    checks.check("rule_unit_case_coverage", all(case_counts[r] == 2 and case_polarities[r] == {"0", "1"} for r in rule_ids), f"rules={len(rule_ids)}, test_cases={len(rule_cases)}")
    fired_rules = {f["rule_id"] for f in firings}
    checks.check("scenario_rule_coverage", rule_ids <= fired_rules, f"covered={len(rule_ids & fired_rules)}/{len(rule_ids)}")

    # Scenario-specific semantic assertions for the primary graph typologies.
    tx_by_id = {t["transaction_id"]: t for t in txs}
    scenario_tx = defaultdict(list)
    for member in members:
        scenario_tx[member["scenario_id"]].append((int(member["sequence_number"]), tx_by_id[member["transaction_id"]]))
    semantic_errors = []
    for scenario in scenarios:
        sid, kind = scenario["scenario_id"], scenario["scenario_type"]
        ordered = [x[1] for x in sorted(scenario_tx[sid], key=lambda x: x[0])]
        if kind == "STRUCTURING":
            if len(ordered) < 4 or not all(Decimal("9000") <= dec(t["amount"]) < Decimal("10000") for t in ordered) or (dt(ordered[-1]["timestamp"]) - dt(ordered[0]["timestamp"])).days > 7:
                semantic_errors.append(sid)
        elif kind in {"FAN_OUT", "PEP_GRAPH_PATTERN"}:
            if len({t["to_account"] for t in ordered}) < 4 or len({t["from_account"] for t in ordered}) != 1:
                semantic_errors.append(sid)
        elif kind == "FAN_IN":
            if len({t["from_account"] for t in ordered}) < 4 or len({t["to_account"] for t in ordered}) != 1:
                semantic_errors.append(sid)
        elif kind in {"LAYERING_CHAIN", "CROSS_BORDER_LAYERING"}:
            if len(ordered) < 4 or any(ordered[i]["to_account"] != ordered[i + 1]["from_account"] or dt(ordered[i]["timestamp"]) >= dt(ordered[i + 1]["timestamp"]) for i in range(len(ordered) - 1)):
                semantic_errors.append(sid)
        elif kind == "CIRCULAR_FLOW":
            if len(ordered) < 3 or ordered[0]["from_account"] != ordered[-1]["to_account"]:
                semantic_errors.append(sid)
        elif kind == "RAPID_PASS_THROUGH":
            if len(ordered) != 2 or ordered[0]["to_account"] != ordered[1]["from_account"] or (dt(ordered[1]["timestamp"]) - dt(ordered[0]["timestamp"])).total_seconds() > 1800 or dec(ordered[1]["amount"]) / dec(ordered[0]["amount"]) < Decimal("0.95"):
                semantic_errors.append(sid)
    checks.check("scenario_semantics", not semantic_errors, f"errors={len(semantic_errors)}, examples={semantic_errors[:5]}")

    duplicate_groups = Counter((i["external_invoice_number"], i["supplier_account"], i["customer_account"]) for i in invoices)
    duplicate_scenarios = {i["scenario_id"] for i in invoices if i["anomaly_type"] == "DUPLICATE_INVOICE"}
    valid_duplicate_scenarios = {i["scenario_id"] for i in invoices if duplicate_groups[(i["external_invoice_number"], i["supplier_account"], i["customer_account"])] > 1}
    checks.check("duplicate_invoice_ground_truth", duplicate_scenarios <= valid_duplicate_scenarios, f"covered={len(duplicate_scenarios & valid_duplicate_scenarios)}/{len(duplicate_scenarios)}")
    checks.check("vat_calculation_normal_rows", all(i["is_vat_anomaly"] == "1" or dec(i["reported_vat_amount"]) == dec(i["expected_vat_amount"]) for i in invoices), "all non-VAT-anomaly invoices reconcile")

    summary = {
        "status": "PASS" if not checks.failures else "FAIL",
        "dataset": metadata["dataset_name"], "version": metadata["version"],
        "checks_passed": len(checks.results) - len(checks.failures),
        "checks_failed": len(checks.failures),
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "transaction_split_counts": dict(sorted(Counter(t["split"] for t in txs).items())),
        "transaction_label_counts": {
            "normal": sum(t["is_anomaly"] == "0" for t in txs),
            "anomaly": sum(t["is_anomaly"] == "1" for t in txs),
            "laundering": sum(t["is_laundering"] == "1" for t in txs),
        },
        "scenario_type_counts": dict(sorted(Counter(s["scenario_type"] for s in scenarios).items())),
        "checks": checks.results,
    }
    return summary


def render_markdown(report: dict) -> str:
    lines = ["# FinReason Dataset Quality Report", "", f"Overall status: **{report['status']}**", "",
             f"Checks passed: {report['checks_passed']}", f"Checks failed: {report['checks_failed']}", "",
             "## Dataset summary", "", "| Table | Rows |", "|---|---:|"]
    lines += [f"| {name} | {count:,} |" for name, count in report["row_counts"].items()]
    lines += ["", "## Label summary", "", "| Label | Transactions |", "|---|---:|"]
    lines += [f"| {name} | {count:,} |" for name, count in report["transaction_label_counts"].items()]
    lines += ["", "## Validation checks", "", "| Check | Status | Evidence |", "|---|---|---|"]
    lines += [f"| {r['check']} | {r['status']} | {r['evidence'].replace('|', '/')} |" for r in report["checks"]]
    lines += ["", "The validator is deterministic and can be rerun with `python3 validate_finreason_dataset.py`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = validate(args.data)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.report_dir / "QUALITY_REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "checks_passed", "checks_failed", "transaction_label_counts")}, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
