"""Tests for triage plan grouping and ordering."""
import unittest

from agent_actions import propose_lifecycle_transitions
from agent_service import _risk_rank, normalize_session
from agent_triage import GROUP_LABELS, build_triage_plan


def _contact(cid, cls, risk, lifecycle="ai_detected"):
    return {"id": cid, "cls": cls, "class_risk": risk, "lifecycle": lifecycle,
            "status": "Review", "frame": 1, "conf": 0.8}


class TestBuildTriagePlan(unittest.TestCase):
    def test_groups_by_proposed_action(self):
        contacts = [
            _contact("A", "wreck", "Critical"),
            _contact("B", "rock", "Low"),
            _contact("C", "pipeline", "High", "rov_scheduled"),
        ]
        plan = build_triage_plan(contacts)
        self.assertEqual(plan["total"], 3)
        self.assertFalse(plan["empty"])
        actions = [g["action"] for g in plan["groups"]]
        self.assertEqual(actions, ["rov_scheduled", "desk_review", "rov_confirmed"])
        self.assertEqual(plan["groups"][0]["label"], GROUP_LABELS["rov_scheduled"])
        self.assertEqual(plan["groups"][0]["count"], 1)
        self.assertEqual(plan["groups"][1]["count"], 1)
        self.assertEqual(plan["groups"][2]["count"], 1)

    def test_critical_first_within_group(self):
        contacts = [
            _contact("low", "rock", "Low"),
            _contact("crit", "wreck", "Critical"),
            _contact("high", "pipeline", "High"),
        ]
        plan = build_triage_plan(contacts)
        rov_group = next(g for g in plan["groups"] if g["action"] == "rov_scheduled")
        risks = [p["class_risk"] for p in rov_group["proposals"]]
        self.assertEqual(risks, ["Critical", "High"])

    def test_risk_tally(self):
        plan = build_triage_plan([
            _contact("A", "wreck", "Critical"),
            _contact("B", "rock", "Low"),
        ])
        self.assertEqual(plan["risk_tally"]["Critical"], 1)
        self.assertEqual(plan["risk_tally"]["Low"], 1)

    def test_empty_state(self):
        plan = build_triage_plan([_contact("X", "wreck", "Critical", "rov_confirmed")])
        self.assertTrue(plan["empty"])
        self.assertEqual(plan["total"], 0)
        self.assertIn("up to date", plan["summary"].lower())

    def test_inherits_guardrails(self):
        plan = build_triage_plan([_contact("A", "wreck", "Critical")])
        self.assertTrue(plan["propose_only"])
        self.assertTrue(plan["clearance_not_proposed"])
        for g in plan["groups"]:
            for p in g["proposals"]:
                self.assertNotIn(p["proposed_state"], ("clearance_issued", "cleared"))

    def test_normalize_session_integration(self):
        ctx = normalize_session(
            frames=[{"dets": [
                {"cls": "wreck", "conf": 0.9, "contact_id": "CT-1-01"},
                {"cls": "rock", "conf": 0.6, "contact_id": "CT-2-01"},
            ]}],
            lifecycle={"items": {}},
        )
        plan = build_triage_plan(ctx["contacts"])
        self.assertGreaterEqual(plan["total"], 2)

    def test_ordering_matches_propose_tool(self):
        contacts = [_contact("Z", "wreck", "High"), _contact("A", "wreck", "Critical")]
        raw = propose_lifecycle_transitions({}, contacts)
        plan = build_triage_plan(contacts)
        rov = next(g for g in plan["groups"] if g["action"] == "rov_scheduled")
        self.assertEqual(
            [p["contact_id"] for p in rov["proposals"]],
            [p["contact_id"] for p in raw["proposals"] if p["proposed_state"] == "rov_scheduled"],
        )
        ranks = [_risk_rank(p["class_risk"]) for p in rov["proposals"]]
        self.assertEqual(ranks, sorted(ranks, reverse=True))


if __name__ == "__main__":
    unittest.main()
