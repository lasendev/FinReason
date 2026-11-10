"""Auditable knowledge-representation models and the FinReason rule catalogue.

Section 7 deliberately separates rule modelling from the forward-chaining engine in
Section 8.  Conditions are named, deterministic Python predicates; no condition is
loaded from CSV and executed dynamically.  External sources support the *reason to
monitor* while project thresholds remain clearly labelled as experimental settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional


SEVERITIES = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
BASIS_TYPES = {
    "GUIDANCE_ALIGNED_INDICATOR",
    "PROJECT_OPERATIONAL_RULE",
    "ACCOUNTING_CONTROL_HEURISTIC",
    "DERIVED_ESCALATION",
}
REFERENCE_SNAPSHOT_DATE = "2026-07-29"


def _frozen_mapping(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return f"{prefix}-{sha256(encoded).hexdigest()[:16].upper()}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _number(context: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = context.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SourceReference:
    source_id: str
    title: str
    publisher: str
    url_or_path: str
    source_type: str
    published_or_updated: str
    accessed_on: str = REFERENCE_SNAPSHOT_DATE
    relevance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "publisher": self.publisher,
            "url_or_path": self.url_or_path,
            "source_type": self.source_type,
            "published_or_updated": self.published_or_updated,
            "accessed_on": self.accessed_on,
            "relevance": self.relevance,
        }


@dataclass(frozen=True)
class Fact:
    """An immutable assertion with record-level or rule-level provenance."""

    fact_id: str
    fact_type: str
    subject_id: str
    value: Any = True
    observed_at: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source_dataset: Optional[str] = None
    source_record_ids: tuple[str, ...] = ()
    derived_by_rule_id: Optional[str] = None
    parent_fact_ids: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.fact_id or not self.fact_type or not self.subject_id:
            raise ValueError("fact_id, fact_type and subject_id are required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.derived_by_rule_id and not self.parent_fact_ids:
            raise ValueError("a derived fact must identify at least one parent fact")
        if self.parent_fact_ids and not self.derived_by_rule_id:
            raise ValueError("parent facts require derived_by_rule_id")
        if not self.derived_by_rule_id and not self.source_record_ids:
            raise ValueError("an observed fact must identify at least one source record")
        object.__setattr__(self, "attributes", _frozen_mapping(self.attributes))
        object.__setattr__(self, "source_record_ids", tuple(dict.fromkeys(self.source_record_ids)))
        object.__setattr__(self, "parent_fact_ids", tuple(dict.fromkeys(self.parent_fact_ids)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "fact_type": self.fact_type,
            "subject_id": self.subject_id,
            "value": self.value,
            "observed_at": self.observed_at,
            "attributes": dict(self.attributes),
            "source_dataset": self.source_dataset,
            "source_record_ids": list(self.source_record_ids),
            "derived_by_rule_id": self.derived_by_rule_id,
            "parent_fact_ids": list(self.parent_fact_ids),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    category: str
    priority: int
    condition_summary: str
    condition_id: str
    derived_fact_type: str
    severity: str
    recommended_action: str
    basis_type: str
    parameter_origin: str
    source_ids: tuple[str, ...]
    rationale: str
    limitation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"invalid severity: {self.severity}")
        if self.basis_type not in BASIS_TYPES:
            raise ValueError(f"invalid basis_type: {self.basis_type}")
        if self.condition_id not in CONDITION_EVALUATORS:
            raise ValueError(f"no executable condition registered for {self.condition_id}")
        if not self.source_ids:
            raise ValueError(f"{self.rule_id} must cite at least one source")
        object.__setattr__(self, "source_ids", tuple(dict.fromkeys(self.source_ids)))
        object.__setattr__(self, "parameters", _frozen_mapping(self.parameters))

    def evaluate(self, context: Mapping[str, Any]) -> bool:
        """Evaluate only the named, reviewed predicate registered in this module."""

        return bool(CONDITION_EVALUATORS[self.condition_id](context, self.parameters))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "fact_category": self.category,
            "priority": self.priority,
            "condition_summary": self.condition_summary,
            "condition_id": self.condition_id,
            "derived_fact": self.derived_fact_type,
            "severity": self.severity,
            "recommended_action": self.recommended_action,
            "basis_type": self.basis_type,
            "parameter_origin": self.parameter_origin,
            "source_ids": list(self.source_ids),
            "rationale": self.rationale,
            "limitation": self.limitation,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class Explanation:
    explanation_id: str
    conclusion_fact_id: str
    rule_id: str
    source_fact_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    reference_source_ids: tuple[str, ...]
    narrative: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "conclusion_fact_id": self.conclusion_fact_id,
            "rule_id": self.rule_id,
            "source_fact_ids": list(self.source_fact_ids),
            "source_record_ids": list(self.source_record_ids),
            "reference_source_ids": list(self.reference_source_ids),
            "narrative": self.narrative,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class KnowledgeAlert:
    alert_id: str
    alert_type: str
    subject_id: str
    severity: str
    conclusion_fact_ids: tuple[str, ...]
    explanation_ids: tuple[str, ...]
    recommended_action: str
    status: str = "REVIEW_REQUIRED"
    legal_determination: bool = False

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"invalid severity: {self.severity}")
        if self.legal_determination:
            raise ValueError("FinReason alerts are indicators, not legal determinations")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "subject_id": self.subject_id,
            "severity": self.severity,
            "conclusion_fact_ids": list(self.conclusion_fact_ids),
            "explanation_ids": list(self.explanation_ids),
            "recommended_action": self.recommended_action,
            "status": self.status,
            "legal_determination": self.legal_determination,
        }


class KnowledgeBase:
    """Validated fact/rule store; inference scheduling is added in Section 8."""

    def __init__(
        self,
        rules: Iterable[Rule] = (),
        references: Optional[Mapping[str, SourceReference]] = None,
    ) -> None:
        self.references = dict(references or {})
        self.rules: dict[str, Rule] = {}
        self.facts: dict[str, Fact] = {}
        self.explanations: dict[str, Explanation] = {}
        self._fact_fingerprints: dict[str, str] = {}
        for rule in rules:
            self.add_rule(rule)

    def add_rule(self, rule: Rule) -> None:
        if rule.rule_id in self.rules:
            raise ValueError(f"duplicate rule_id: {rule.rule_id}")
        missing_sources = set(rule.source_ids) - set(self.references)
        if missing_sources:
            raise ValueError(f"{rule.rule_id} has unknown sources: {sorted(missing_sources)}")
        self.rules[rule.rule_id] = rule

    def add_fact(self, fact: Fact) -> Fact:
        if fact.fact_id in self.facts:
            if self.facts[fact.fact_id] != fact:
                raise ValueError(f"conflicting fact_id: {fact.fact_id}")
            return self.facts[fact.fact_id]
        missing_parents = set(fact.parent_fact_ids) - set(self.facts)
        if missing_parents:
            raise ValueError(f"missing parent facts: {sorted(missing_parents)}")
        if fact.derived_by_rule_id and fact.derived_by_rule_id not in self.rules:
            raise ValueError(f"unknown derivation rule: {fact.derived_by_rule_id}")
        fingerprint = _stable_id("FP", {
            "fact_type": fact.fact_type,
            "subject_id": fact.subject_id,
            "value": fact.value,
            "observed_at": fact.observed_at,
            "attributes": dict(fact.attributes),
            "parents": fact.parent_fact_ids,
            "rule": fact.derived_by_rule_id,
        })
        if fingerprint in self._fact_fingerprints:
            return self.facts[self._fact_fingerprints[fingerprint]]
        self.facts[fact.fact_id] = fact
        self._fact_fingerprints[fingerprint] = fact.fact_id
        return fact

    def evaluate_rule(self, rule_id: str, context: Mapping[str, Any]) -> bool:
        return self.rules[rule_id].evaluate(context)

    def derive_fact(
        self,
        rule_id: str,
        subject_id: str,
        context: Mapping[str, Any],
        parent_fact_ids: Iterable[str],
        observed_at: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
    ) -> tuple[Fact, Explanation]:
        rule = self.rules[rule_id]
        parents = tuple(dict.fromkeys(parent_fact_ids))
        if not parents:
            raise ValueError("at least one parent fact is required")
        missing = set(parents) - set(self.facts)
        if missing:
            raise ValueError(f"missing parent facts: {sorted(missing)}")
        if not rule.evaluate(context):
            raise ValueError(f"condition for {rule_id} is not satisfied")
        record_ids = tuple(dict.fromkeys(
            record_id
            for parent_id in parents
            for record_id in self.facts[parent_id].source_record_ids
        ))
        fact_payload = {
            "rule_id": rule_id,
            "subject_id": subject_id,
            "parents": parents,
            "derived_fact": rule.derived_fact_type,
        }
        fact = self.add_fact(Fact(
            fact_id=_stable_id("FACT", fact_payload),
            fact_type=rule.derived_fact_type,
            subject_id=subject_id,
            value=True,
            observed_at=observed_at,
            attributes={
                "condition_id": rule.condition_id,
                "severity": rule.severity,
                **dict(attributes or {}),
            },
            source_dataset="DERIVED",
            source_record_ids=record_ids,
            derived_by_rule_id=rule_id,
            parent_fact_ids=parents,
        ))
        explanation_payload = {"fact_id": fact.fact_id, "rule_id": rule_id, "parents": parents}
        explanation = Explanation(
            explanation_id=_stable_id("EXP", explanation_payload),
            conclusion_fact_id=fact.fact_id,
            rule_id=rule_id,
            source_fact_ids=parents,
            source_record_ids=record_ids,
            reference_source_ids=rule.source_ids,
            narrative=(
                f"{rule.name} fired because the reviewed condition was satisfied: "
                f"{rule.condition_summary}. The result is an indicator for human review, "
                "not proof of criminal conduct."
            ),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.explanations[explanation.explanation_id] = explanation
        return fact, explanation

    def provenance_chain(self, fact_id: str) -> dict[str, Any]:
        if fact_id not in self.facts:
            raise KeyError(fact_id)
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()

        def visit(current_id: str) -> None:
            if current_id in seen:
                return
            seen.add(current_id)
            fact = self.facts[current_id]
            for parent_id in fact.parent_fact_ids:
                visit(parent_id)
            ordered.append(fact.to_dict())

        visit(fact_id)
        return {"conclusion_fact_id": fact_id, "facts": ordered}


# Each condition is explicit, total (missing data -> False), and side-effect free.
def _r001(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _number(c, "amount_gbp") >= _number(p, "high_value_gbp", 100_000)


def _r002(c: Mapping[str, Any], _: Mapping[str, Any]) -> bool:
    source, target = c.get("from_country"), c.get("to_country")
    return bool(source and target and source != target)


def _r003(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    elevated = set(p.get("elevated_test_countries", ("XZ", "XY")))
    return c.get("from_country") in elevated or c.get("to_country") in elevated


def _r004(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _r001(c, p) and str(c.get("payment_format", "")).upper() == "CASH"


def _r005(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    amount = _number(c, "amount_gbp")
    return _number(p, "near_threshold_low_gbp", 9_000) <= amount < _number(p, "near_threshold_high_gbp", 10_000)


def _r006(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _number(c, "near_threshold_count_7d") >= _number(p, "minimum_count", 4)


def _r007(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _number(c, "outgoing_count_24h") >= _number(p, "minimum_count", 12)


def _r008(c: Mapping[str, Any], _: Mapping[str, Any]) -> bool:
    value = c.get("timestamp")
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return False
    return 0 <= parsed.astimezone(timezone.utc).hour < 4


def _r009(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _number(c, "account_age_days", 10**9) < _number(p, "new_account_days", 180) and _r001(c, p)


def _r010(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _as_bool(c.get("is_pep")) and _r001(c, p)


def _r011(c: Mapping[str, Any], _: Mapping[str, Any]) -> bool:
    paid, received = c.get("payment_currency"), c.get("receiving_currency")
    return bool(paid and received and paid != received)


def _r012(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    amount = _number(c, "amount_gbp")
    divisor = _number(p, "round_divisor_gbp", 10_000)
    return amount >= _number(p, "minimum_round_amount_gbp", 50_000) and divisor > 0 and isclose(amount % divisor, 0.0, abs_tol=0.01)


def _r013(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    balance = _number(c, "from_balance_before")
    return balance > 0 and _number(c, "amount_gbp") / balance >= _number(p, "minimum_balance_fraction", 0.80)


def _r014(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _number(c, "distinct_successors_24h") >= _number(p, "minimum_counterparties", 4)


def _r015(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _number(c, "distinct_predecessors_24h") >= _number(p, "minimum_counterparties", 4)


def _r016(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return (
        _number(c, "layering_depth") >= _number(p, "minimum_depth", 4)
        and _as_bool(c.get("timestamps_strictly_increasing"))
        and _number(c, "minimum_step_retention") >= _number(p, "minimum_retention", 0.95)
    )


def _r017(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    depth = _number(c, "cycle_depth")
    return (
        _as_bool(c.get("returns_to_origin"))
        and _as_bool(c.get("timestamps_strictly_increasing"))
        and _number(p, "minimum_cycle_depth", 3) <= depth <= _number(p, "maximum_depth", 6)
    )


def _r018(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    delay = _number(c, "pass_through_minutes", float("inf"))
    return 0 <= delay <= _number(p, "maximum_minutes", 30) and _number(c, "amount_retention") >= _number(p, "minimum_retention", 0.95)


def _r019(c: Mapping[str, Any], _: Mapping[str, Any]) -> bool:
    return _as_bool(c.get("fan_out_pattern")) and _as_bool(c.get("fan_in_pattern"))


def _r020(c: Mapping[str, Any], _: Mapping[str, Any]) -> bool:
    return _as_bool(c.get("layering_pattern")) and _as_bool(c.get("cross_border_tx"))


def _r021(c: Mapping[str, Any], _: Mapping[str, Any]) -> bool:
    graph_present = any(_as_bool(c.get(key)) for key in (
        "fan_out_pattern", "fan_in_pattern", "layering_pattern", "circular_flow", "rapid_pass_through"
    ))
    return _as_bool(c.get("is_pep")) and graph_present


def _r022(c: Mapping[str, Any], _: Mapping[str, Any]) -> bool:
    return _as_bool(c.get("structuring_pattern")) and _as_bool(c.get("fan_out_pattern"))


def _r023(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    indicators = {str(v) for v in c.get("independent_risk_facts", ()) if v}
    return len(indicators) >= int(_number(p, "minimum_indicators", 2))


def _unpaid_overdue(c: Mapping[str, Any], days: float) -> bool:
    return str(c.get("status", "")).upper() == "UNPAID" and _number(c, "days_overdue") >= days


def _r024(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _unpaid_overdue(c, _number(p, "minimum_days_overdue", 30))


def _r025(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _unpaid_overdue(c, _number(p, "minimum_days_overdue", 60))


def _r026(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _unpaid_overdue(c, _number(p, "minimum_days_overdue", 90))


def _r027(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    difference = abs(_number(c, "reported_vat_amount") - _number(c, "expected_vat_amount"))
    return difference > _number(p, "vat_tolerance_gbp", 0.01)


def _r028(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    if c.get("vat_rate") is None:
        return False
    return _number(c, "vat_rate", -1) not in {float(v) for v in p.get("configured_vat_rates", (0, 5, 20))}


def _r029(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _number(c, "same_number_supplier_customer_count") >= _number(p, "minimum_duplicate_count", 2)


def _r030(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _number(c, "linked_payment_amount") > _number(c, "gross_amount") + _number(p, "amount_tolerance_gbp", 0.01)


def _r031(c: Mapping[str, Any], _: Mapping[str, Any]) -> bool:
    invoice, customer = c.get("invoice_currency"), c.get("customer_base_currency")
    return bool(invoice and customer and invoice != customer)


def _r032(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    return _as_bool(c.get("overdue_90")) and _number(c, "gross_amount") >= _number(p, "high_value_overdue_gbp", 500_000)


def _r033(c: Mapping[str, Any], p: Mapping[str, Any]) -> bool:
    facts = {str(v) for v in c.get("high_or_critical_derived_facts", ()) if v}
    return len(facts) >= int(_number(p, "minimum_high_critical_facts", 3))


CONDITION_EVALUATORS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], bool]] = {
    f"R{i:03d}": evaluator
    for i, evaluator in enumerate((
        _r001, _r002, _r003, _r004, _r005, _r006, _r007, _r008, _r009,
        _r010, _r011, _r012, _r013, _r014, _r015, _r016, _r017, _r018,
        _r019, _r020, _r021, _r022, _r023, _r024, _r025, _r026, _r027,
        _r028, _r029, _r030, _r031, _r032, _r033,
    ), start=1)
}


RULE_PARAMETERS: dict[str, dict[str, Any]] = {
    "R001": {"high_value_gbp": 100_000},
    "R003": {"elevated_test_countries": ["XZ", "XY"]},
    "R004": {"high_value_gbp": 100_000},
    "R005": {"near_threshold_low_gbp": 9_000, "near_threshold_high_gbp": 10_000},
    "R006": {"minimum_count": 4, "window_days": 7},
    "R007": {"minimum_count": 12, "window_hours": 24},
    "R009": {"new_account_days": 180, "high_value_gbp": 100_000},
    "R010": {"high_value_gbp": 100_000},
    "R012": {"minimum_round_amount_gbp": 50_000, "round_divisor_gbp": 10_000},
    "R013": {"minimum_balance_fraction": 0.80},
    "R014": {"minimum_counterparties": 4, "window_hours": 24},
    "R015": {"minimum_counterparties": 4, "window_hours": 24},
    "R016": {"minimum_depth": 4, "minimum_retention": 0.95, "window_hours": 24},
    "R017": {"minimum_cycle_depth": 3, "maximum_depth": 6, "window_hours": 24},
    "R018": {"maximum_minutes": 30, "minimum_retention": 0.95},
    "R023": {"minimum_indicators": 2},
    "R024": {"minimum_days_overdue": 30},
    "R025": {"minimum_days_overdue": 60},
    "R026": {"minimum_days_overdue": 90},
    "R027": {"vat_tolerance_gbp": 0.01},
    "R028": {"configured_vat_rates": [0, 5, 20]},
    "R029": {"minimum_duplicate_count": 2},
    "R030": {"amount_tolerance_gbp": 0.01},
    "R032": {"high_value_overdue_gbp": 500_000},
    "R033": {"minimum_high_critical_facts": 3},
}


def load_reference_catalogue() -> dict[str, SourceReference]:
    references = [
        SourceReference("FATF_REC_2025", "The FATF Recommendations (amended October 2025)", "FATF", "https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html", "INTERNATIONAL_STANDARD", "2025-10", relevance="Risk-based CDD, ongoing monitoring, PEP and suspicious-transaction framework."),
        SourceReference("FATF_PEP", "FATF Guidance: Politically Exposed Persons (Recommendations 12 and 22)", "FATF", "https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Peps-r12-r22.html", "OFFICIAL_GUIDANCE", "2013-06", relevance="PEP measures are preventive and risk-based; PEP status alone does not imply criminality."),
        SourceReference("UK_MLR_R33", "Money Laundering Regulations 2017, regulation 33", "UK legislation", "https://www.legislation.gov.uk/uksi/2017/692/regulation/33", "LEGISLATION", "2017", relevance="Enhanced due diligence for complex, unusually large or unusual patterns without apparent economic/legal purpose."),
        SourceReference("FCA_FCG3", "Financial Crime Guide, Chapter 3", "Financial Conduct Authority", "https://www.handbook.fca.org.uk/handbook/FCG/3/?view=chapter", "REGULATOR_GUIDANCE", "2024-11-29", relevance="Risk-sensitive transaction monitoring, customer context and network-aware review."),
        SourceReference("HMRC_AML_INDICATORS", "AMLG11100: Reporting Suspicious Activity", "HM Revenue & Customs", "https://www.gov.uk/hmrc-internal-manuals/anti-money-laundering-guidance-for-supervised-businesses/amlg11100", "GOVERNMENT_GUIDANCE", "2026", relevance="Size/frequency deviations, unusually large cash/FX activity and unexpected fund movements as contextual indicators."),
        SourceReference("FINCEN_STRUCTURING", "FinCEN Ruling 2005-6: Suspicious Activity Reporting (Structuring)", "FinCEN", "https://www.fincen.gov/resources/statutes-regulations/administrative-rulings/suspicious-activity-reporting-structuring", "REGULATOR_RULING", "2005-07-15", relevance="Structuring means breaking up transactions to evade applicable reporting/recordkeeping requirements; monitoring must be risk-commensurate."),
        SourceReference("FINCEN_FUNNEL", "FinCEN Advisory FIN-2014-A005: Funnel Accounts and TBML", "FinCEN", "https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2014-a005", "REGULATOR_ADVISORY", "2014-05-28", relevance="Multiple inflows, consolidation and rapid onward movement; indicators should be assessed with customer context."),
        SourceReference("FATF_TBML", "Trade-Based Money Laundering: Risk Indicators", "FATF and Egmont Group", "https://www.fatf-gafi.org/en/publications/Methodsandtrends/Trade-based-money-laundering-indicators.html", "OFFICIAL_TYPOLOGY_GUIDANCE", "2021-03-11", relevance="Trade-document, invoice, business-structure and transaction risk indicators."),
        SourceReference("GOVUK_VAT", "VAT rates", "UK Government", "https://www.gov.uk/vat-rates", "GOVERNMENT_REFERENCE", "accessed current page", relevance="UK headline VAT rates are 20%, 5% and 0%; applicability depends on goods/services and exemptions."),
        SourceReference("IFRS9_ECL", "IFRS 9 Financial Instruments", "IFRS Foundation", "https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf?bypass=on", "ACCOUNTING_STANDARD", "2021 issued edition", relevance="Rebuttable 30-day significant-credit-risk and 90-day default presumptions; not AML determinations."),
        SourceReference("W3C_PROV", "PROV-O: The PROV Ontology", "W3C", "https://www.w3.org/TR/prov-o/", "TECHNICAL_STANDARD", "2013-04-30", relevance="Derived entities should retain links to generating activities and source entities."),
        SourceReference("NIST_AI_RMF", "Artificial Intelligence Risk Management Framework 1.0", "NIST", "https://www.nist.gov/itl/ai-risk-management-framework", "RISK_FRAMEWORK", "2023-01", relevance="Supports valid, accountable, transparent, explainable and interpretable AI."),
        SourceReference("FINREASON_METHOD", "FinReason synthetic dataset methodology", "FinReason coursework", "finreason_synthetic_dataset/METHODOLOGY.md", "PROJECT_SPECIFICATION", "v1.0.0", relevance="Defines synthetic scenarios, experimental parameters, labels and known limitations."),
    ]
    return {reference.source_id: reference for reference in references}


def _governance(
    basis: str,
    parameter_origin: str,
    sources: tuple[str, ...],
    rationale: str,
    limitation: str,
) -> dict[str, Any]:
    return {
        "basis_type": basis,
        "parameter_origin": parameter_origin,
        "source_ids": sources,
        "rationale": rationale,
        "limitation": limitation,
    }


RULE_GOVERNANCE: dict[str, dict[str, Any]] = {
    "R001": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("UK_MLR_R33", "FCA_FCG3", "FINREASON_METHOD"), "Unusually large transactions warrant contextual review; £100,000 makes 'large' reproducible in this dataset.", "£100,000 is not a universal reporting threshold and value alone is not suspicious."),
    "R002": _governance("GUIDANCE_ALIGNED_INDICATOR", "DIRECT_DATA_RELATION", ("FATF_REC_2025", "FCA_FCG3", "FINREASON_METHOD"), "Geographic context is relevant to risk assessment, so differing endpoint countries create a low-level fact.", "Cross-border activity is common and is not suspicious without additional context."),
    "R003": _governance("PROJECT_OPERATIONAL_RULE", "SYNTHETIC_TEST_CONFIGURATION", ("FATF_REC_2025", "FINREASON_METHOD"), "Higher-risk geography is a recognised contextual factor; XZ and XY provide deterministic test values.", "XZ and XY are fictional and must never be presented as real high-risk jurisdictions."),
    "R004": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("HMRC_AML_INDICATORS", "UK_MLR_R33", "FINREASON_METHOD"), "Guidance identifies unusually large cash activity as a potential indicator; the amount makes the test reproducible.", "The amount is experimental; the rule creates review evidence, not a reporting duty."),
    "R005": _governance("PROJECT_OPERATIONAL_RULE", "SYNTHETIC_TEST_CONFIGURATION", ("FINCEN_STRUCTURING", "FINREASON_METHOD"), "Near-threshold values are retained as ingredients for detecting repeated splitting behaviour.", "The £9k–£10k band borrows a typology shape but is not a UK legal threshold; one payment must not imply structuring."),
    "R006": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FINCEN_STRUCTURING", "FINREASON_METHOD"), "Repeated sub-threshold transactions can indicate deliberate evasion; aggregation distinguishes a pattern from one payment.", "Intent and applicable jurisdictional obligations require human investigation; four in seven days is experimental."),
    "R007": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("HMRC_AML_INDICATORS", "FCA_FCG3", "FINREASON_METHOD"), "Transaction frequency that departs from expected behaviour is a monitoring signal.", "Twelve payments in 24 hours is a coursework parameter and may be normal for many businesses."),
    "R008": _governance("PROJECT_OPERATIONAL_RULE", "PROJECT_CONFIGURATION", ("FCA_FCG3", "FINREASON_METHOD"), "Time-of-day can help characterise activity when compared with a customer's normal behaviour.", "00:00–03:59 UTC is not inherently suspicious and ignores local time unless contextualised."),
    "R009": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FCA_FCG3", "FINREASON_METHOD"), "A large transaction soon after onboarding can merit closer comparison with the known customer profile.", "The 180-day and £100k limits are experimental and do not establish wrongdoing."),
    "R010": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FATF_PEP", "FINREASON_METHOD"), "FATF calls for additional risk-based measures for PEP relationships, particularly around material activity.", "PEP measures are preventive; PEP status and value do not imply criminal conduct."),
    "R011": _governance("GUIDANCE_ALIGNED_INDICATOR", "DIRECT_DATA_RELATION", ("HMRC_AML_INDICATORS", "FCA_FCG3", "FINREASON_METHOD"), "Currency conversion is a contextual feature useful for understanding foreign-currency fund movement.", "Currency conversion is ordinary activity and remains a low-severity fact."),
    "R012": _governance("PROJECT_OPERATIONAL_RULE", "PROJECT_CONFIGURATION", ("HMRC_AML_INDICATORS", "FINREASON_METHOD"), "Large round amounts can be useful explainable features when combined with customer and transaction context.", "The divisibility test is heuristic, currency-dependent and weak in isolation."),
    "R013": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FINCEN_FUNNEL", "HMRC_AML_INDICATORS", "FINREASON_METHOD"), "A very large proportion of available balance moving out can represent unexpected movement or rapid depletion.", "The 80% fraction is experimental and can occur legitimately."),
    "R014": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FINCEN_FUNNEL", "FCA_FCG3", "FINREASON_METHOD"), "Network-aware monitoring can identify dispersal to multiple counterparties.", "Four successors in 24 hours is project-defined; payroll and other legitimate fan-out require context."),
    "R015": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FINCEN_FUNNEL", "FCA_FCG3", "FINREASON_METHOD"), "Multiple inflows and consolidation are documented funnel-account features.", "Four predecessors in 24 hours is project-defined; merchant collection can be legitimate."),
    "R016": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FATF_REC_2025", "FCA_FCG3", "FINREASON_METHOD"), "Chronological multi-account movement operationalises layering as an explainable graph pattern.", "Depth, 24-hour window and 95% retention are experimental, not a statutory definition of layering."),
    "R017": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FATF_REC_2025", "FCA_FCG3", "FINREASON_METHOD"), "A time-respecting return of funds to origin is an unusually complex network pattern suitable for review.", "A cycle can have a lawful purpose; depth and time limits are project settings."),
    "R018": _governance("GUIDANCE_ALIGNED_INDICATOR", "PROJECT_CONFIGURATION", ("FINCEN_FUNNEL", "FINREASON_METHOD"), "Funnel guidance describes little time between receipt and onward movement, motivating a pass-through fact.", "Thirty minutes and 95% retention are synthetic test parameters, not regulatory cut-offs."),
    "R019": _governance("DERIVED_ESCALATION", "COMPOSITE_REASONING", ("FINCEN_FUNNEL", "FCA_FCG3", "FINREASON_METHOD"), "Combining dispersal and consolidation yields stronger context than either degree pattern alone.", "Both components can still arise legitimately; the conclusion requires review."),
    "R020": _governance("DERIVED_ESCALATION", "COMPOSITE_REASONING", ("FATF_REC_2025", "FCA_FCG3", "FINREASON_METHOD"), "Combining a validated temporal chain with cross-border context creates a more specific explanation.", "Cross-border layering remains an indicator, not proof or automatic SAR logic."),
    "R021": _governance("DERIVED_ESCALATION", "COMPOSITE_REASONING", ("FATF_PEP", "FCA_FCG3", "FINREASON_METHOD"), "A PEP relationship plus an independently observed graph pattern supports enhanced human review.", "PEP status must never be treated as evidence of criminality."),
    "R022": _governance("DERIVED_ESCALATION", "COMPOSITE_REASONING", ("FINCEN_STRUCTURING", "FINCEN_FUNNEL", "FINREASON_METHOD"), "Two independently computed pattern facts provide a clearer multi-stage risk narrative.", "Synthetic thresholds and labels do not establish intent to evade controls."),
    "R023": _governance("DERIVED_ESCALATION", "PROJECT_CONFIGURATION", ("FINCEN_FUNNEL", "FCA_FCG3", "FINREASON_METHOD"), "Official guidance stresses evaluating indicators in combination and against expected activity.", "Two indicators is a project escalation threshold; correlated duplicates must not be counted as independent."),
    "R024": _governance("ACCOUNTING_CONTROL_HEURISTIC", "IFRS_REBUTTABLE_PRESUMPTION", ("IFRS9_ECL", "FINREASON_METHOD"), "Thirty days past due is an IFRS 9 rebuttable credit-risk presumption and a useful receivables ageing fact.", "This is a credit-risk/accounting fact, not an AML or fraud conclusion."),
    "R025": _governance("ACCOUNTING_CONTROL_HEURISTIC", "PROJECT_CONFIGURATION", ("IFRS9_ECL", "FINREASON_METHOD"), "A 60-day bucket provides an intermediate operational ageing escalation.", "IFRS 9 does not make 60 days a universal legal threshold; this is project-defined."),
    "R026": _governance("ACCOUNTING_CONTROL_HEURISTIC", "IFRS_REBUTTABLE_PRESUMPTION", ("IFRS9_ECL", "FINREASON_METHOD"), "IFRS 9 includes a rebuttable presumption that default occurs no later than 90 days past due.", "The presumption can be rebutted and is not an AML determination."),
    "R027": _governance("ACCOUNTING_CONTROL_HEURISTIC", "DIRECT_RECONCILIATION", ("GOVUK_VAT", "FATF_TBML", "FINREASON_METHOD"), "Recomputing expected VAT provides a traceable arithmetic control; invoice inconsistencies are relevant to trade-document review.", "A mismatch may be error, rounding or special tax treatment; £0.01 tolerance is a data-control setting."),
    "R028": _governance("ACCOUNTING_CONTROL_HEURISTIC", "UK_VAT_REFERENCE_WITH_CONTEXT", ("GOVUK_VAT", "FINREASON_METHOD"), "The synthetic list mirrors UK headline 0%, 5% and 20% VAT rates.", "Actual applicability depends on supply, date, place and exemptions; the rule is invalid outside the configured UK-like dataset."),
    "R029": _governance("ACCOUNTING_CONTROL_HEURISTIC", "DIRECT_RECONCILIATION", ("FATF_TBML", "FINREASON_METHOD"), "Repeated invoice numbers for the same parties can reveal duplicate billing or multiple-invoicing risk.", "Duplicates may be corrections or data-entry issues and require document review."),
    "R030": _governance("ACCOUNTING_CONTROL_HEURISTIC", "DIRECT_RECONCILIATION", ("FATF_TBML", "FINREASON_METHOD"), "Comparing linked payment with invoice gross value detects an explainable overpayment inconsistency.", "Overpayment may reflect credit notes, deposits, FX or aggregation; payment linkage must be verified."),
    "R031": _governance("ACCOUNTING_CONTROL_HEURISTIC", "DIRECT_DATA_RELATION", ("FATF_TBML", "FINREASON_METHOD"), "Currency mismatch is useful trade and settlement context for document review.", "Foreign-currency invoicing is normal and therefore remains low severity."),
    "R032": _governance("DERIVED_ESCALATION", "PROJECT_CONFIGURATION", ("IFRS9_ECL", "FINREASON_METHOD"), "Combining a 90-day ageing fact with material exposure prioritises receivables review.", "£500,000 is an experimental materiality value, not a universal accounting threshold."),
    "R033": _governance("DERIVED_ESCALATION", "PROJECT_CONFIGURATION", ("FCA_FCG3", "NIST_AI_RMF", "FINREASON_METHOD"), "Three distinct high/critical facts create a transparent prioritisation rule for human review.", "The count is project-defined; correlated facts and underlying evidence must be reviewed before action."),
}


def load_rules(catalogue_path: Path | str) -> list[Rule]:
    import csv

    with Path(catalogue_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rules: list[Rule] = []
    for row in rows:
        rule_id = row["rule_id"]
        if rule_id not in RULE_GOVERNANCE:
            raise ValueError(f"missing governance metadata for {rule_id}")
        governance = RULE_GOVERNANCE[rule_id]
        rules.append(Rule(
            rule_id=rule_id,
            name=row["rule_name"],
            category=row["fact_category"],
            priority=int(row["priority"]),
            condition_summary=row["condition_summary"],
            condition_id=rule_id,
            derived_fact_type=row["derived_fact"],
            severity=row["severity"],
            recommended_action=row["recommended_action"],
            parameters=RULE_PARAMETERS.get(rule_id, {}),
            **governance,
        ))
    return rules


def build_rule_test_contexts() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Return semantically meaningful positive/negative contexts for all 33 rules."""

    return {
        "R001": ({"amount_gbp": 100_000}, {"amount_gbp": 99_999.99}),
        "R002": ({"from_country": "GB", "to_country": "US"}, {"from_country": "GB", "to_country": "GB"}),
        "R003": ({"from_country": "GB", "to_country": "XZ"}, {"from_country": "GB", "to_country": "US"}),
        "R004": ({"amount_gbp": 120_000, "payment_format": "CASH"}, {"amount_gbp": 120_000, "payment_format": "BANK_TRANSFER"}),
        "R005": ({"amount_gbp": 9_500}, {"amount_gbp": 10_000}),
        "R006": ({"near_threshold_count_7d": 4}, {"near_threshold_count_7d": 3}),
        "R007": ({"outgoing_count_24h": 12}, {"outgoing_count_24h": 11}),
        "R008": ({"timestamp": "2025-01-01T03:59:00Z"}, {"timestamp": "2025-01-01T04:00:00Z"}),
        "R009": ({"account_age_days": 179, "amount_gbp": 100_000}, {"account_age_days": 180, "amount_gbp": 100_000}),
        "R010": ({"is_pep": True, "amount_gbp": 100_000}, {"is_pep": False, "amount_gbp": 100_000}),
        "R011": ({"payment_currency": "GBP", "receiving_currency": "USD"}, {"payment_currency": "GBP", "receiving_currency": "GBP"}),
        "R012": ({"amount_gbp": 50_000}, {"amount_gbp": 50_001}),
        "R013": ({"amount_gbp": 800, "from_balance_before": 1_000}, {"amount_gbp": 799.99, "from_balance_before": 1_000}),
        "R014": ({"distinct_successors_24h": 4}, {"distinct_successors_24h": 3}),
        "R015": ({"distinct_predecessors_24h": 4}, {"distinct_predecessors_24h": 3}),
        "R016": ({"layering_depth": 4, "timestamps_strictly_increasing": True, "minimum_step_retention": 0.95}, {"layering_depth": 4, "timestamps_strictly_increasing": False, "minimum_step_retention": 0.95}),
        "R017": ({"cycle_depth": 3, "returns_to_origin": True, "timestamps_strictly_increasing": True}, {"cycle_depth": 3, "returns_to_origin": False, "timestamps_strictly_increasing": True}),
        "R018": ({"pass_through_minutes": 30, "amount_retention": 0.95}, {"pass_through_minutes": 31, "amount_retention": 0.95}),
        "R019": ({"fan_out_pattern": True, "fan_in_pattern": True}, {"fan_out_pattern": True, "fan_in_pattern": False}),
        "R020": ({"layering_pattern": True, "cross_border_tx": True}, {"layering_pattern": True, "cross_border_tx": False}),
        "R021": ({"is_pep": True, "layering_pattern": True}, {"is_pep": True, "layering_pattern": False}),
        "R022": ({"structuring_pattern": True, "fan_out_pattern": True}, {"structuring_pattern": True, "fan_out_pattern": False}),
        "R023": ({"independent_risk_facts": ["HIGH_VALUE_TX", "CROSS_BORDER_TX"]}, {"independent_risk_facts": ["HIGH_VALUE_TX", "HIGH_VALUE_TX"]}),
        "R024": ({"status": "UNPAID", "days_overdue": 30}, {"status": "PAID_LATE", "days_overdue": 40}),
        "R025": ({"status": "UNPAID", "days_overdue": 60}, {"status": "UNPAID", "days_overdue": 59}),
        "R026": ({"status": "UNPAID", "days_overdue": 90}, {"status": "UNPAID", "days_overdue": 89}),
        "R027": ({"reported_vat_amount": 200.02, "expected_vat_amount": 200.00}, {"reported_vat_amount": 200.01, "expected_vat_amount": 200.00}),
        "R028": ({"vat_rate": 17}, {"vat_rate": 20}),
        "R029": ({"same_number_supplier_customer_count": 2}, {"same_number_supplier_customer_count": 1}),
        "R030": ({"linked_payment_amount": 120.02, "gross_amount": 120.00}, {"linked_payment_amount": 120.01, "gross_amount": 120.00}),
        "R031": ({"invoice_currency": "USD", "customer_base_currency": "GBP"}, {"invoice_currency": "GBP", "customer_base_currency": "GBP"}),
        "R032": ({"overdue_90": True, "gross_amount": 500_000}, {"overdue_90": True, "gross_amount": 499_999.99}),
        "R033": ({"high_or_critical_derived_facts": ["A", "B", "C"]}, {"high_or_critical_derived_facts": ["A", "A", "B"]}),
    }
