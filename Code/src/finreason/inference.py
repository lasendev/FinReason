"""Deterministic Match -> Resolve -> Act forward chaining for FinReason."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional

from .knowledge import Explanation, Fact, KnowledgeBase


def _freeze(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _stable_key(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class RuleActivation:
    """A fully bound candidate rule firing produced during Match."""

    rule_id: str
    subject_id: str
    context: Mapping[str, Any]
    parent_fact_ids: tuple[str, ...]
    observed_at: Optional[str] = None
    output_attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id or not self.subject_id:
            raise ValueError("rule_id and subject_id are required")
        if not self.parent_fact_ids:
            raise ValueError("an activation requires supporting parent facts")
        object.__setattr__(self, "context", _freeze(self.context))
        object.__setattr__(self, "output_attributes", _freeze(self.output_attributes))
        object.__setattr__(self, "parent_fact_ids", tuple(dict.fromkeys(self.parent_fact_ids)))

    @property
    def activation_key(self) -> str:
        return _stable_key({
            "rule_id": self.rule_id,
            "subject_id": self.subject_id,
            "context": dict(self.context),
            "parent_fact_ids": self.parent_fact_ids,
            "observed_at": self.observed_at,
            "output_attributes": dict(self.output_attributes),
        })


@dataclass(frozen=True)
class InferenceEvent:
    cycle: int
    agenda_position: int
    rule_id: str
    rule_priority: int
    subject_id: str
    activation_key: str
    parent_fact_ids: tuple[str, ...]
    derived_fact_id: str
    explanation_id: str
    new_fact_added: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "agenda_position": self.agenda_position,
            "rule_id": self.rule_id,
            "rule_priority": self.rule_priority,
            "subject_id": self.subject_id,
            "activation_key": self.activation_key,
            "parent_fact_ids": list(self.parent_fact_ids),
            "derived_fact_id": self.derived_fact_id,
            "explanation_id": self.explanation_id,
            "new_fact_added": self.new_fact_added,
        }


Matcher = Callable[[KnowledgeBase], Iterable[RuleActivation]]


class ForwardChainer:
    """Forward chainer with explicit phases, priority resolution and fixpoint safety."""

    def __init__(self, knowledge_base: KnowledgeBase, matchers: Iterable[Matcher]) -> None:
        self.knowledge_base = knowledge_base
        self.matchers = tuple(matchers)
        self.fired_activation_keys: set[str] = set()
        self.trace: list[InferenceEvent] = []
        self.cycle_summaries: list[dict[str, Any]] = []
        self.fixpoint_reached = False

    def match(self) -> list[RuleActivation]:
        """Match phase: bind valid, not-yet-fired rule activations."""

        unique: dict[str, RuleActivation] = {}
        for matcher in self.matchers:
            for activation in matcher(self.knowledge_base):
                if activation.rule_id not in self.knowledge_base.rules:
                    raise ValueError(f"matcher produced unknown rule {activation.rule_id}")
                if set(activation.parent_fact_ids) - set(self.knowledge_base.facts):
                    continue
                if activation.activation_key in self.fired_activation_keys:
                    continue
                if not self.knowledge_base.evaluate_rule(activation.rule_id, activation.context):
                    continue
                unique.setdefault(activation.activation_key, activation)
        return list(unique.values())

    def resolve(self, matches: Iterable[RuleActivation]) -> list[RuleActivation]:
        """Resolve phase: priority descending, then stable deterministic tie-breaks."""

        return sorted(
            matches,
            key=lambda activation: (
                -self.knowledge_base.rules[activation.rule_id].priority,
                activation.rule_id,
                activation.subject_id,
                activation.activation_key,
            ),
        )

    def act(self, activation: RuleActivation, *, cycle: int, agenda_position: int) -> InferenceEvent:
        """Act phase: derive one fact and its explanation, preserving all provenance."""

        fact_count_before = len(self.knowledge_base.facts)
        fact, explanation = self.knowledge_base.derive_fact(
            activation.rule_id,
            activation.subject_id,
            activation.context,
            activation.parent_fact_ids,
            observed_at=activation.observed_at,
            attributes=activation.output_attributes,
        )
        self.fired_activation_keys.add(activation.activation_key)
        event = InferenceEvent(
            cycle=cycle,
            agenda_position=agenda_position,
            rule_id=activation.rule_id,
            rule_priority=self.knowledge_base.rules[activation.rule_id].priority,
            subject_id=activation.subject_id,
            activation_key=activation.activation_key,
            parent_fact_ids=activation.parent_fact_ids,
            derived_fact_id=fact.fact_id,
            explanation_id=explanation.explanation_id,
            new_fact_added=len(self.knowledge_base.facts) > fact_count_before,
        )
        self.trace.append(event)
        return event

    def run(self, *, maximum_cycles: int = 100) -> list[InferenceEvent]:
        if maximum_cycles < 1:
            raise ValueError("maximum_cycles must be at least one")
        self.fixpoint_reached = False
        for cycle in range(1, maximum_cycles + 1):
            matched = self.match()
            agenda = self.resolve(matched)
            if not agenda:
                self.fixpoint_reached = True
                self.cycle_summaries.append({
                    "cycle": cycle,
                    "matched": 0,
                    "resolved": 0,
                    "acted": 0,
                    "new_facts": 0,
                    "termination": "FIXPOINT",
                })
                return list(self.trace)
            events = [
                self.act(activation, cycle=cycle, agenda_position=position)
                for position, activation in enumerate(agenda, start=1)
            ]
            self.cycle_summaries.append({
                "cycle": cycle,
                "matched": len(matched),
                "resolved": len(agenda),
                "acted": len(events),
                "new_facts": sum(event.new_fact_added for event in events),
                "termination": "CONTINUE",
            })
        remaining = self.resolve(self.match())
        if remaining:
            raise RuntimeError(
                f"forward chaining did not reach a fixpoint within {maximum_cycles} cycles"
            )
        self.fixpoint_reached = True
        return list(self.trace)

    def trace_records(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.trace]


def static_matcher(activations: Iterable[RuleActivation]) -> Matcher:
    """Return a matcher for precomputed observed-evidence activations."""

    stored = tuple(activations)

    def matcher(_: KnowledgeBase) -> Iterable[RuleActivation]:
        return stored

    return matcher

