import unittest
from pathlib import Path

from finreason.inference import ForwardChainer, RuleActivation, static_matcher
from finreason.knowledge import Fact, KnowledgeBase, load_reference_catalogue, load_rules


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "finreason_synthetic_dataset" / "data" / "rule_catalog.csv"


class ForwardChainingTests(unittest.TestCase):
    def setUp(self):
        self.references = load_reference_catalogue()
        self.rules = load_rules(CATALOGUE)
        self.kb = KnowledgeBase(self.rules, self.references)

    def observed(self, fact_id, fact_type, subject, record_id, attributes=None):
        return self.kb.add_fact(Fact(
            fact_id=fact_id,
            fact_type=fact_type,
            subject_id=subject,
            attributes=attributes or {},
            source_dataset="test",
            source_record_ids=(record_id,),
        ))

    def test_repeated_match_does_not_create_firing_loop(self):
        raw = self.observed("RAW-1", "TRANSACTION_CONTEXT", "TX-1", "TX-1")
        activation = RuleActivation("R001", "TX-1", {"amount_gbp": 100_000}, (raw.fact_id,))
        engine = ForwardChainer(self.kb, [static_matcher([activation])])
        events = engine.run(maximum_cycles=5)
        self.assertEqual(len(events), 1)
        self.assertTrue(engine.fixpoint_reached)
        self.assertEqual(engine.cycle_summaries[-1]["termination"], "FIXPOINT")

    def test_resolve_orders_higher_priority_first(self):
        raw = self.observed("RAW-1", "TRANSACTION_CONTEXT", "TX-1", "TX-1")
        activations = [
            RuleActivation("R002", "TX-1", {"from_country": "GB", "to_country": "US"}, (raw.fact_id,)),
            RuleActivation("R001", "TX-1", {"amount_gbp": 100_000}, (raw.fact_id,)),
        ]
        engine = ForwardChainer(self.kb, [static_matcher(activations)])
        resolved = engine.resolve(engine.match())
        self.assertEqual([a.rule_id for a in resolved], ["R001", "R002"])

    def test_false_condition_is_not_added_to_agenda(self):
        raw = self.observed("RAW-1", "TRANSACTION_CONTEXT", "TX-1", "TX-1")
        activation = RuleActivation("R001", "TX-1", {"amount_gbp": 99_999}, (raw.fact_id,))
        engine = ForwardChainer(self.kb, [static_matcher([activation])])
        self.assertEqual(engine.match(), [])

    def test_two_stage_chain_preserves_recursive_provenance(self):
        graph = self.observed("RAW-G", "GRAPH_EVIDENCE", "ACC-1", "TX-G")
        cross = self.observed("RAW-X", "TRANSACTION_CONTEXT", "ACC-1", "TX-G")
        seeds = [
            RuleActivation("R016", "ACC-1", {
                "layering_depth": 4,
                "timestamps_strictly_increasing": True,
                "minimum_step_retention": 0.97,
            }, (graph.fact_id,)),
            RuleActivation("R002", "ACC-1", {
                "from_country": "GB", "to_country": "US",
            }, (cross.fact_id,)),
        ]

        def cross_border_layering(kb):
            layering = [f for f in kb.facts.values() if f.fact_type == "LAYERING_PATTERN"]
            borders = [f for f in kb.facts.values() if f.fact_type == "CROSS_BORDER_TX"]
            return [
                RuleActivation("R020", left.subject_id, {
                    "layering_pattern": True, "cross_border_tx": True,
                }, (left.fact_id, right.fact_id))
                for left in layering for right in borders
                if left.subject_id == right.subject_id
            ]

        engine = ForwardChainer(self.kb, [static_matcher(seeds), cross_border_layering])
        events = engine.run(maximum_cycles=5)
        self.assertEqual([e.rule_id for e in events], ["R016", "R002", "R020"])
        conclusion = next(f for f in self.kb.facts.values() if f.fact_type == "CROSS_BORDER_LAYERING")
        chain = self.kb.provenance_chain(conclusion.fact_id)
        self.assertEqual({f["fact_id"] for f in chain["facts"]}, {
            "RAW-G", "RAW-X", events[0].derived_fact_id, events[1].derived_fact_id, conclusion.fact_id,
        })

    def test_output_attributes_are_retained(self):
        raw = self.observed("RAW-1", "GRAPH_EVIDENCE", "ACC-1", "TX-1")
        activation = RuleActivation(
            "R014", "ACC-1", {"distinct_successors_24h": 4}, (raw.fact_id,),
            output_attributes={"window_hours": 24, "counterparties": ["A", "B", "C", "D"]},
        )
        engine = ForwardChainer(self.kb, [static_matcher([activation])])
        engine.run()
        fact = next(f for f in self.kb.facts.values() if f.fact_type == "FAN_OUT_PATTERN")
        self.assertEqual(fact.attributes["window_hours"], 24)


if __name__ == "__main__":
    unittest.main()
