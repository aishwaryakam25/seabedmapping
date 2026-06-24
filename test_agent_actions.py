"""Tests for lifecycle proposals and clearance guardrails."""
import unittest

from agent_actions import NEVER_PROPOSE, propose_lifecycle_transitions
from agent_service import normalize_session


def _contact(cid, cls, risk, lifecycle="ai_detected", status="Review"):
    return {"id": cid, "cls": cls, "class_risk": risk, "lifecycle": lifecycle,
            "status": status, "frame": 1, "conf": 0.8}


class TestProposeLifecycleTransitions(unittest.TestCase):
    def test_high_risk_to_rov_scheduled(self):
        out = propose_lifecycle_transitions({}, [_contact("CT-1-01", "wreck", "Critical")])
        self.assertEqual(out["proposals"][0]["proposed_state"], "rov_scheduled")
        self.assertTrue(out["clearance_not_proposed"])

    def test_low_risk_to_desk_review(self):
        out = propose_lifecycle_transitions({}, [_contact("CT-1-02", "rock", "Low")])
        self.assertEqual(out["proposals"][0]["proposed_state"], "desk_review")

    def test_skips_terminal(self):
        out = propose_lifecycle_transitions({}, [
            _contact("A", "wreck", "Critical", "rov_confirmed"),
            _contact("B", "wreck", "Critical", "clearance_issued"),
        ])
        self.assertEqual(out["count"], 0)

    def test_rov_scheduled_to_confirmed(self):
        out = propose_lifecycle_transitions({}, [_contact("X", "pipeline", "High", "rov_scheduled")])
        self.assertEqual(out["proposals"][0]["proposed_state"], "rov_confirmed")

    def test_never_clearance(self):
        for state in NEVER_PROPOSE:
            out = propose_lifecycle_transitions({}, [_contact("X", "wreck", "Critical", state)])
            for p in out.get("proposals") or []:
                self.assertNotIn(p["proposed_state"], NEVER_PROPOSE)

    def test_lifecycle_from_session(self):
        ctx = normalize_session(
            frames=[{"dets": [{"cls": "wreck", "conf": 0.9, "contact_id": "CT-9-01"}]}],
            lifecycle={"items": {"CT-9-01": {"state": "ai_detected"}}},
        )
        out = propose_lifecycle_transitions({}, ctx["contacts"])
        self.assertEqual(out["proposals"][0]["current_lifecycle"], "ai_detected")


if __name__ == "__main__":
    unittest.main()
