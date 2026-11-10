# Generation methodology

## Design objective

The dataset fills gaps left by row-labelled public fraud datasets. It supplies exact ground truth for paths, cycles, fan structures, chained rule conclusions and invoice facts while preserving a realistic distinction between transaction-, path-, scenario- and invoice-level labels.

## Reproducibility

Generation uses Python's local `random.Random` instance with seed `20250713`. Amounts are created as integer cents. The observation period is calendar year 2025, and the receivables analysis date is fixed at 31 December 2025. Generated files include SHA-256 checksums.

## Time splits

- Train: 1 January–30 June 2025
- Validation: 1 July–31 August 2025
- Test: 1 September–31 December 2025

Splits are time based rather than random. Scenario transactions remain in their scenario's temporal region. This reduces leakage from closely related pattern transactions appearing in different splits.

## Normal traffic

Background amounts follow a bounded log-normal distribution and are combined with controlled account, currency, country, channel and payment-format variation. Invoice payments are linked through `invoice_id`. Ledger balances are replayed chronologically after generation; opening balances are raised where necessary to prevent impossible negative balances.

## Positive scenarios

Every positive scenario has an entry in `expected_alerts.csv`. Graph scenarios preserve ordered membership in `scenario_transactions.csv`. Temporal chains use increasing timestamps, cycle paths return to the origin, and rapid pass-through cases retain at least 95% of the incoming value within 30 minutes.

Structuring cases contain four to seven payments between the same endpoints, each between 9,000 and 10,000 synthetic GBP units, within seven days. These are coursework configuration values, not legal reporting thresholds.

## Hard negatives

The dataset deliberately contains legitimate structures that resemble suspicious graph patterns: payroll fan-out, merchant fan-in, documented high-value transfers, a single near-threshold invoice payment and supply-chain paths. These cases are labelled non-anomalous and have no expected alerts. They are designed to expose detectors that rely only on graph degree, path length or amount thresholds without context.

## Knowledge representation coverage

The reference catalogue contains 33 rules across transaction facts, temporal aggregates, graph facts, derived/chained facts and invoice facts. Each rule has one positive and one negative unit fixture. Scenario-level firing records specify the minimum facts required for the intended explanation chain.

## Known limitations

- Synthetic account and transaction distributions cannot reproduce all real banking behaviour.
- Currency values are not exchange-rate normalised; production comparisons require a dated FX table.
- PEP and country-risk values are fictional test inputs, not external compliance assertions.
- Background generation can naturally satisfy low-level rule conditions. This is intentional and makes false-positive analysis possible.
- Scenario labels identify planted patterns. They do not prove criminal intent.
- The dataset must supplement, not replace, evaluation on IBM AML, PaySim or another external source.

