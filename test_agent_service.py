"""Offline tests for agent session parsing and tools."""
import unittest

from agent_service import (
    execute_tool, explain_contact_logic, normalize_session, risk_score,
    tool_query_contacts, tool_recommend_followup, tool_survey_summary,
)

SAMPLE_FRAMES = [
    {
        "name": "line12_001.png", "natW": 1024, "natH": 1257,
        "dets": [
            {"cls": "wreck", "conf": 0.91, "box": [120, 200, 280, 520], "cxn": 0.42, "cyn": 0.35,
             "wnorm": 0.15, "status": "Review", "contact_id": "CT-1-01"},
            {"cls": "rock", "conf": 0.62, "box": [600, 400, 680, 480], "cxn": 0.62, "cyn": 0.55,
             "wnorm": 0.08, "status": "Verified", "contact_id": "CT-1-02"},
        ],
    },
    {
        "name": "line12_002.png", "natW": 1024, "natH": 1257,
        "dets": [{"class": "pipeline", "confidence": 0.78, "box": [300, 100, 900, 180],
                  "cxn": 0.58, "cyn": 0.12, "wnorm": 0.58, "status": "Review"}],
    },
]

SAMPLE_SESSION = {
    "frames": 2, "hasData": True, "survey_line": "L12_20240515_1045",
    "detections": [
        {"cls": "boulder", "conf": 0.71, "frame": 1, "x": 30, "y": 20, "range": 45, "status": "Review"},
        {"cls": "debris", "conf": 0.44, "frame": 2, "x": 100, "y": 35, "status": "Rejected"},
    ],
}


class TestNormalizeSession(unittest.TestCase):
    def test_frames_schema(self):
        ctx = normalize_session(frames=SAMPLE_FRAMES, survey_info={"range": "150 m"})
        self.assertEqual(len(ctx["contacts"]), 3)
        wreck = next(c for c in ctx["contacts"] if c["id"] == "CT-1-01")
        self.assertEqual(wreck["class_risk"], "Critical")

    def test_session_fallback(self):
        ctx = normalize_session(session=SAMPLE_SESSION)
        self.assertEqual(len(ctx["contacts"]), 2)

    def test_lifecycle_from_items(self):
        ctx = normalize_session(frames=SAMPLE_FRAMES, lifecycle={"items": {"CT-1-01": {"state": "desk_review"}}})
        wreck = next(c for c in ctx["contacts"] if c["id"] == "CT-1-01")
        self.assertEqual(wreck["lifecycle"], "desk_review")


class TestTools(unittest.TestCase):
    def setUp(self):
        self.ctx = normalize_session(frames=SAMPLE_FRAMES)

    def test_survey_summary(self):
        out = tool_survey_summary(self.ctx)
        self.assertEqual(out["totals"]["detections"], 3)

    def test_query_and_explain(self):
        out = tool_query_contacts(self.ctx, {"risk": "Critical"})
        self.assertEqual(out["matched"], 1)
        ex = execute_tool("explain_contact", {"contact_id": "CT-1-01"}, self.ctx)
        self.assertIn("summary", ex)

    def test_recommend_followup(self):
        out = tool_recommend_followup(self.ctx, {"min_risk": "High"})
        self.assertGreaterEqual(out["count"], 1)

    def test_risk_score(self):
        self.assertAlmostEqual(risk_score("wreck", 0.9), 0.9)


if __name__ == "__main__":
    unittest.main()
