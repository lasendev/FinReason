# FinReason Synthetic Financial Crime Dataset

Version 1.0.0 is a deterministic, fully synthetic dataset for evaluating graph-search and symbolic-rule approaches to explainable financial anomaly detection. It contains no real people, accounts or transactions.

## Contents

| File | Purpose |
|---|---|
| `data/accounts.csv` | Synthetic account profiles and opening balances |
| `data/transactions.csv` | Timestamped directed financial transactions with transaction-level labels |
| `data/invoices.csv` | Receivables and VAT facts for invoice-oriented rules |
| `data/scenarios.csv` | Scenario-level ground truth, split and difficulty |
| `data/scenario_transactions.csv` | Ordered transaction membership for each scenario |
| `data/scenario_invoices.csv` | Invoice membership for each invoice scenario |
| `data/expected_alerts.csv` | Minimum alert that a correct detector should produce for every positive scenario |
| `data/expected_rule_firings.csv` | Required rule firings for positive scenarios; additional lower-level firings may be valid |
| `data/rule_catalog.csv` | Reference catalogue for 33 educational production rules |
| `data/rule_test_cases.csv` | One positive and one negative unit case for every rule |
| `data/metadata.json` | Version, seed, counts and dataset-level metadata |
| `data/checksums.sha256` | SHA-256 checksums for generated files |
| `generate_finreason_dataset.py` | Dependency-free deterministic generator |
| `validate_finreason_dataset.py` | Dependency-free quality and semantic validator |
| `QUALITY_REPORT.md` | Human-readable results of the latest validation run |
| `FinReason_Data_Quality.ipynb` | Inspectable companion notebook for loading and checking the package |

See `DATA_DICTIONARY.md` for column definitions and `METHODOLOGY.md` for the generation and evaluation design.

## Dataset scale

The default release contains:

- 5,000 accounts
- 67,670 transactions
- 1,885 invoices
- 1,570 labelled scenarios
- 4,944 anomaly-positive transactions
- 4,659 laundering-positive transactions
- 265 legitimate hard-negative scenarios
- 33 rules and 66 rule unit cases

Exact counts are recorded in `data/metadata.json`.

## Scenario families

Graph and temporal cases include fan-out, fan-in, time-respecting layering chains, circular flows, rapid pass-through, combined fan-out/fan-in, cross-border layering and graph-plus-rule escalation.

Transaction and aggregate cases include structuring, unusual frequency, elevated-risk fictional routes, high-value cash, new-account activity, PEP test cases, round amounts and balance drain.

Invoice cases include 30/60/90-day overdue receivables, VAT mismatches, invalid configured VAT rates, duplicate invoices, overpayment and foreign-currency invoices.

Legitimate look-alikes include payroll fan-out, merchant fan-in, documented high-value payments, single near-threshold payments and supply-chain paths. These are important for testing false positives.

## Recommended evaluation protocol

1. Develop rules and traversal logic using the `train` split.
2. Tune non-statutory thresholds using `validation` only.
3. Report final results once on `test`.
4. Evaluate transaction labels and scenario labels separately.
5. Convert a detected graph path to transaction-level predictions only through an explicitly documented mapping.
6. Report synthetic results separately from IBM AML and PaySim results. Do not combine them into a single headline score.
7. Include precision, recall, F1, false-positive rate, scenario detection rate, runtime, peak memory and audit-trail completeness.
8. Include ablations: rules only, graph only, statistical baseline and graph plus rules.

For graph evaluation, use `scenario_transactions.csv` to recover the ordered ground-truth path. Search must respect timestamp order. A static path that requires travelling backward in time is not a valid match.

## Quick start

Generate the release again:

```bash
python3 generate_finreason_dataset.py
```

Validate it:

```bash
python3 validate_finreason_dataset.py
```

Both scripts use only the Python standard library. The default generator seed is `20250713`. Custom smaller builds are supported:

```bash
python3 generate_finreason_dataset.py --accounts 1000 --normal-transactions 10000 --normal-invoices 300 --output data_small
python3 validate_finreason_dataset.py --data data_small --report-dir validation_small
```

## Important interpretation notes

- This dataset is synthetic and intended for coursework, testing and demonstration.
- `XZ` and `XY` are fictional elevated-risk country codes created for deterministic tests. They are not ISO codes and do not describe real jurisdictions.
- Monetary thresholds and VAT rates are educational configuration values, not statements of current law.
- An anomaly is not automatically money laundering. The fields `is_anomaly` and `is_laundering` are deliberately separate.
- `expected_rule_firings.csv` gives the minimum required firings for each scenario. A well-designed engine may also fire valid low-level rules, such as a high-value rule inside a more complex graph case.
- Synthetic performance is not evidence of real-world effectiveness. Use the external IBM AML and PaySim datasets for evaluation and discuss their label and domain limitations.

## Public sharing

Upload the entire `finreason_synthetic_dataset` folder or the provided ZIP file. Keep the documentation, generator, validation report and checksums with the CSV files so users can understand and verify the release. If Drive previews truncate a large CSV, users should download it rather than treating the preview as the full file.

## Licence

Released under Creative Commons Attribution 4.0 International. See `LICENSE.txt` for attribution details.
