"""Tests for ROV dive-plan sequencing and briefs."""
import unittest

from agent_rov import build_dive_plan, plan_to_text, _tooling_for
from agent_service import normalize_session


def _contact(cid, cls, risk, frame=1, lifecycle="rov_scheduled", conf=0.8, **kw):
    c = {"id": cid, "cls": cls, "class_risk": risk, "frame": frame, "lifecycle": lifecycle,
         "status": "Review", "conf": conf}
    c.update(kw)
    return c


class TestBuildDivePlan(unittest.TestCase):
    def test_empty_when_no_rov_scheduled(self):
        plan = build_dive_plan([_contact("A", "wreck", "Critical", lifecycle="ai_detected")])
        self.assertTrue(plan["empty"])
        self.assertIn("Triage", plan["summary"])

    def test_sequences_by_frame(self):
        contacts = [
            _contact("F3", "rock", "Low", frame=3),
            _contact("F1", "wreck", "Critical", frame=1),
            _contact("F2", "pipeline", "High", frame=2),
        ]
        plan = build_dive_plan(contacts)
        frames = [d["frame"] for d in plan["dives"]]
        self.assertEqual(frames, [1, 2, 3])

    def test_critical_first_within_frame(self):
        contacts = [
            _contact("low", "rock", "Low", frame=2, conf=0.6),
            _contact("crit", "wreck", "Critical", frame=2, conf=0.9),
        ]
        plan = build_dive_plan(contacts)
        self.assertEqual(plan["dive_count"], 1)
        ids = [t["contact_id"] for t in plan["dives"][0]["targets"]]
        self.assertEqual(ids[0], "crit")

    def test_class_tooling(self):
        self.assertIn("multibeam", _tooling_for("wreck").lower())
        self.assertIn("tracker", _tooling_for("pipeline").lower())
        self.assertIn("profiler", _tooling_for("scour").lower())

    def test_co_located_notes(self):
        contacts = [
            _contact("A", "rock", "Low", frame=4),
            _contact("B", "boulder", "Medium", frame=4),
        ]
        plan = build_dive_plan(contacts)
        notes = " ".join(plan["dives"][0]["notes"]).lower()
        self.assertIn("co-located", notes)

    def test_guardrails(self):
        plan = build_dive_plan([_contact("X", "wreck", "Critical")])
        self.assertTrue(plan["verification_only"])
        self.assertTrue(plan["clearance_not_issued"])

    def test_plan_to_text(self):
        plan = build_dive_plan([_contact("X", "wreck", "Critical")])
        text = plan_to_text(plan)
        self.assertIn("DIVE 1", text)
        self.assertIn("verification only", text.lower())

    def test_lifecycle_integration(self):
        ctx = normalize_session(
            frames=[{"dets": [
                {"cls": "wreck", "conf": 0.9, "contact_id": "CT-1-01", "lifecycle": "rov_scheduled"},
            ]}],
            lifecycle={"items": {"CT-1-01": {"state": "rov_scheduled"}}},
        )
        plan = build_dive_plan(ctx["contacts"])
        self.assertEqual(plan["target_count"], 1)


if __name__ == "__main__":
    unittest.main()
