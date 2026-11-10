# Data dictionary

All timestamps are UTC. Money fields are decimal values with two fractional digits. Boolean fields use `0` and `1`. Empty optional fields are written as empty CSV values.

## `accounts.csv`

| Column | Meaning |
|---|---|
| `account_id` | Unique synthetic account identifier |
| `account_type` | Individual, business, merchant, payroll or charity profile |
| `country` | Synthetic residence/registration country; `XZ` and `XY` are fictional test codes |
| `base_currency` | Account's expected operating currency |
| `opened_at` | Account opening timestamp |
| `risk_band` | Synthetic LOW, MEDIUM or ELEVATED_TEST configuration |
| `is_pep` | Synthetic politically exposed person test flag |
| `industry` | Synthetic customer industry/profile |
| `expected_monthly_volume` | Expected nominal monthly transaction volume |
| `opening_balance` | Balance immediately before the first dataset transaction |
| `is_synthetic` | Always `1` |

## `transactions.csv`

| Column | Meaning |
|---|---|
| `transaction_id` | Unique transaction identifier |
| `timestamp` | Event timestamp in UTC |
| `from_account`, `to_account` | Directed graph endpoints |
| `amount` | Amount in `payment_currency` |
| `payment_currency` | Currency sent |
| `receiving_currency` | Currency received/credited |
| `payment_format` | Bank transfer, card, direct debit, mobile, cash or cheque |
| `transaction_type` | Transfer, payroll, purchase or invoice payment |
| `channel` | Online, mobile, branch or API |
| `from_country`, `to_country` | Denormalised account countries |
| `is_cross_border` | `1` when the endpoint countries differ |
| `from_balance_before`, `from_balance_after` | Sender ledger balances around the transaction |
| `to_balance_before`, `to_balance_after` | Receiver ledger balances around the transaction |
| `invoice_id` | Linked invoice when applicable |
| `description` | Synthetic contextual description |
| `split` | Time-based `train`, `validation` or `test` assignment |
| `scenario_id` | Ground-truth scenario; empty for unstructured background traffic |
| `is_anomaly` | Transaction-level anomaly label |
| `is_laundering` | Transaction-level laundering label |
| `primary_typology` | Primary scenario label or `NORMAL` |
| `expected_severity` | NONE, MEDIUM, HIGH or CRITICAL |
| `is_synthetic` | Always `1` |

## `invoices.csv`

| Column | Meaning |
|---|---|
| `invoice_id` | Unique internal invoice identifier |
| `external_invoice_number` | Business-facing reference used for duplicate checks |
| `issue_date`, `due_date`, `paid_at` | Invoice lifecycle dates |
| `supplier_account`, `customer_account` | Account foreign keys |
| `net_amount` | Pre-VAT amount |
| `vat_rate` | Configured educational VAT percentage |
| `expected_vat_amount` | Computed VAT at the configured rate |
| `reported_vat_amount` | VAT stated on the invoice |
| `gross_amount` | Net plus reported VAT |
| `invoice_currency` | Invoice currency |
| `status` | UNPAID, PAID_ON_TIME or PAID_LATE |
| `is_overdue_as_of_2025_12_31` | Unpaid and past due at the fixed analysis date |
| `days_overdue` | Overdue duration relative to payment or fixed analysis date |
| `is_vat_anomaly`, `is_anomaly` | Invoice-level labels |
| `anomaly_type`, `scenario_id`, `split` | Ground-truth classification and time split |
| `is_synthetic` | Always `1` |

## Ground-truth tables

`scenarios.csv` is the scenario-level truth table. It records the scenario family, anomaly/laundering labels, time bounds, split, difficulty, expected detector, description and member counts.

`scenario_transactions.csv` maps scenarios to ordered transactions. `sequence_number` defines the intended temporal path/order; `transaction_role` identifies fan-out edges, cycle edges, invoice payments and other roles.

`scenario_invoices.csv` maps invoice-oriented scenarios to their core invoices.

`expected_alerts.csv` records the minimum scenario-level alert, detector, type, severity, graph root/target, depth bounds, relevant rules and an explanation keyword.

`expected_rule_firings.csv` records required positive rule firings and their intended chaining order. It is a minimum set, not an assertion that all other rules must remain silent.

## Rule tables

`rule_catalog.csv` defines 33 educational rules with category, priority, human-readable condition, derived fact, severity and recommended action.

`rule_test_cases.csv` contains a positive and negative input-fact case for every rule. `input_facts_json` is a compact specification fixture; adapt it to the final `Fact` class when implementing the engine.

