import tempfile
import unittest
from pathlib import Path

from finreason.knowledge import (
    CONDITION_EVALUATORS,
    RULE_GOVERNANCE,
    Explanation,
    Fact,
    KnowledgeAlert,
    KnowledgeBase,
    build_rule_test_contexts,
    load_reference_catalogue,
    load_rules,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "finreason_synthetic_dataset" / "data" / "rule_catalog.csv"


class KnowledgeRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.references = load_reference_catalogue()
        cls.rules = load_rules(CATALOGUE)

    def test_all_33_catalogue_rules_are_executable_and_governed(self):
        self.assertEqual(len(self.rules), 33)
        self.assertEqual({r.rule_id for r in self.rules}, set(CONDITION_EVALUATORS))
        self.assertEqual({r.rule_id for r in self.rules}, set(RULE_GOVERNANCE))
        for rule in self.rules:
            self.assertTrue(set(rule.source_ids) <= set(self.references))
            self.assertTrue(rule.rationale)
            self.assertTrue(rule.limitation)

    def test_every_rule_has_real_positive_and_negative_boundary_case(self):
        contexts = build_rule_test_contexts()
        self.assertEqual(set(contexts), {r.rule_id for r in self.rules})
        for rule in self.rules:
            positive, negative = contexts[rule.rule_id]
            with self.subTest(rule_id=rule.rule_id, expected=True):
                self.assertTrue(rule.evaluate(positive))
            with self.subTest(rule_id=rule.rule_id, expected=False):
                self.assertFalse(rule.evaluate(negative))

    def test_observed_fact_requires_record_provenance(self):
        with self.assertRaises(ValueError):
            Fact("FACT-1", "TX_CONTEXT", "TX-1")

    def test_derived_fact_requires_parent_provenance(self):
        with self.assertRaises(ValueError):
            Fact("FACT-1", "HIGH_VALUE_TX", "TX-1", derived_by_rule_id="R001")

    def test_knowledge_base_preserves_complete_provenance(self):
        kb = KnowledgeBase(self.rules, self.references)
        raw = kb.add_fact(Fact(
            fact_id="FACT-RAW-1",
            fact_type="TRANSACTION_CONTEXT",
            subject_id="TX-1",
            attributes={"amount_gbp": 125_000},
            source_dataset="finreason",
            source_record_ids=("TX-1",),
        ))
        conclusion, explanation = kb.derive_fact(
            "R001", "TX-1", {"amount_gbp": 125_000}, (raw.fact_id,)
        )
        self.assertEqual(conclusion.parent_fact_ids, (raw.fact_id,))
        self.assertEqual(conclusion.source_record_ids, ("TX-1",))
        self.assertEqual(explanation.conclusion_fact_id, conclusion.fact_id)
        self.assertTrue(explanation.reference_source_ids)
        chain = kb.provenance_chain(conclusion.fact_id)
        self.assertEqual([f["fact_id"] for f in chain["facts"]], [raw.fact_id, conclusion.fact_id])

    def test_duplicate_facts_are_suppressed(self):
        kb = KnowledgeBase(self.rules, self.references)
        first = kb.add_fact(Fact("FACT-A", "TX_CONTEXT", "TX-1", source_dataset="x", source_record_ids=("1",)))
        second = kb.add_fact(Fact("FACT-B", "TX_CONTEXT", "TX-1", source_dataset="x", source_record_ids=("1",)))
        self.assertEqual(first.fact_id, second.fact_id)
        self.assertEqual(len(kb.facts), 1)

    def test_alert_cannot_claim_legal_determination(self):
        with self.assertRaises(ValueError):
            KnowledgeAlert("A-1", "HIGH_VALUE_TX", "TX-1", "MEDIUM", ("F-1",), ("E-1",), "Review", legal_determination=True)


if __name__ == "__main__":
    unittest.main()
