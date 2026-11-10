# FinReason Dataset Quality Report

Overall status: **PASS**

Checks passed: 43
Checks failed: 0

## Dataset summary

| Table | Rows |
|---|---:|
| accounts.csv | 5,000 |
| transactions.csv | 67,670 |
| invoices.csv | 1,885 |
| scenarios.csv | 1,570 |
| scenario_transactions.csv | 6,269 |
| scenario_invoices.csv | 385 |
| expected_alerts.csv | 1,305 |
| expected_rule_firings.csv | 2,585 |
| rule_catalog.csv | 33 |
| rule_test_cases.csv | 66 |

## Label summary

| Label | Transactions |
|---|---:|
| normal | 62,726 |
| anomaly | 4,944 |
| laundering | 4,659 |

## Validation checks

| Check | Status | Evidence |
|---|---|---|
| required_files | PASS | missing=[] |
| columns_accounts.csv | PASS | missing=[] |
| columns_transactions.csv | PASS | missing=[] |
| columns_invoices.csv | PASS | missing=[] |
| columns_scenarios.csv | PASS | missing=[] |
| unique_account_id | PASS | rows=5000, distinct=5000 |
| unique_transaction_id | PASS | rows=67670, distinct=67670 |
| unique_invoice_id | PASS | rows=1885, distinct=1885 |
| unique_scenario_id | PASS | rows=1570, distinct=1570 |
| unique_rule_id | PASS | rows=33, distinct=33 |
| unique_test_case_id | PASS | rows=66, distinct=66 |
| metadata_count_accounts.csv | PASS | metadata=5000, actual=5000 |
| metadata_count_transactions.csv | PASS | metadata=67670, actual=67670 |
| metadata_count_invoices.csv | PASS | metadata=1885, actual=1885 |
| metadata_count_scenarios.csv | PASS | metadata=1570, actual=1570 |
| metadata_count_scenario_transactions.csv | PASS | metadata=6269, actual=6269 |
| metadata_count_scenario_invoices.csv | PASS | metadata=385, actual=385 |
| metadata_count_expected_alerts.csv | PASS | metadata=1305, actual=1305 |
| metadata_count_expected_rule_firings.csv | PASS | metadata=2585, actual=2585 |
| metadata_count_rule_catalog.csv | PASS | metadata=33, actual=33 |
| metadata_count_rule_test_cases.csv | PASS | metadata=66, actual=66 |
| transaction_account_fks | PASS | transactions=67670 |
| invoice_account_fks | PASS | invoices=1885 |
| scenario_transaction_fks | PASS | members=6269 |
| scenario_invoice_fks | PASS | members=385 |
| alert_scenario_fks | PASS | alerts=1305 |
| firing_fks | PASS | firings=2585 |
| transaction_time_splits | PASS | Counter({'train': 33830, 'test': 22309, 'validation': 11531}) |
| transactions_after_account_open | PASS | all transaction timestamps are on/after both account opening timestamps |
| positive_amounts | PASS | minimum=20.39 |
| binary_labels | PASS | allowed={0,1} |
| laundering_implies_anomaly | PASS | no laundering-positive transaction is anomaly-negative |
| cross_border_consistency | PASS | derived flag agrees with countries |
| ledger_balance_reconciliation | PASS | errors=0 |
| non_negative_balances | PASS | minimum_final=10000.00 |
| scenario_member_counts | PASS | mismatches=0 |
| positive_scenario_alert_coverage | PASS | positive=1305, alerted=1305 |
| hard_negative_no_expected_alert | PASS | hard_negatives=265, incorrectly_alerted=0 |
| rule_unit_case_coverage | PASS | rules=33, test_cases=66 |
| scenario_rule_coverage | PASS | covered=33/33 |
| scenario_semantics | PASS | errors=0, examples=[] |
| duplicate_invoice_ground_truth | PASS | covered=30/30 |
| vat_calculation_normal_rows | PASS | all non-VAT-anomaly invoices reconcile |

The validator is deterministic and can be rerun with `python3 validate_finreason_dataset.py`.
