# Survey Operations Console — Client & Technical Overview

## Purpose

The Survey Ops layer closes the loop between **AI sonar detection** and **field verification**. After NautiCAI scores contacts on side-scan frames, survey engineers triage risk, queue ROV dives, capture findings, and export deliverables that feed the survey PDF.

## End-to-end workflow

1. **Load imagery** — Data Review (`/review`): import frames, set survey metadata.
2. **Run detection** — YOLO sonar model; confidence threshold in sidebar (`set_conf` in localStorage).
3. **Risk triage** — Dashboard and Detections aggregate Critical/High contacts.
4. **ROV follow-up** — ROV Ops console (`/rov`): flag → schedule → in-progress → close.
5. **Deliverables** — Multi-frame PDF from Data Review; Section 05 ROV Follow-up Plan includes queue chart and table.

Track live progress on **Survey Workflow** (`/workflow`).

## ROV Ops console (`/rov`)

- **Risk score** = class weight × confidence (0–1). Weights are tunable in `assets/seabed_survey_ops.js` → `CONFIG.classRisk`.
- **Tiers**: High ≥ 0.65, Medium ≥ 0.35, Low below (thresholds `highT` / `medT`).
- **States**: Unreviewed → Flagged → Scheduled → In progress → Closed.
- **Findings panel**: reclassification, measured size, condition, outcome, ROV notes.
- **Persistence**: `localStorage` key `seabed_rov_workflow`.
- **Export**: `rov_deployment_queue.json` — handoff for dive planning; also cached as `seabed_rov_workflow_export` for PDF generation.

Without an active session, the console renders **demo contacts** so the UI is immediately usable.

## Data integration

| Key | Purpose |
|-----|---------|
| `nauticai_session` | Detections from Data Review (`detections[]`: cls, conf, frame, x, y, range) |
| `nauticai_review_frames` | Frame count / imagery |
| `nauticai_survey_info` | Project, survey line, vessel |
| `nauticai_theme` | Dark/light (`body.light` CSS vars) |
| `seabed_rov_workflow` | Per-contact workflow state and findings |
| `seabed_rov_workflow_export` | Last exported queue JSON for PDF |

Field mapping accepts `confidence` as 0–1 or 0–100. Configure in `SeabedSurveyOps.CONFIG`.

## PDF — Section 05 ROV Follow-up Plan

Multi-frame survey reports (`POST /report-multi`) include:

- Matplotlib horizontal bar chart of workflow states
- Summary counts (flagged, scheduled, in progress, closed)
- Queue table (up to 14 rows; full list in JSON export)

If no queue is supplied, high-risk detections (Critical/High) are listed as **recommended** for ROV.

## Bell notifications

The header bell badge shows the higher of **high-risk detection count** or **open ROV high-risk follow-ups**, matching survey urgency.

## For clients

NautiCAI does not replace ROV inspection for high-consequence contacts. The platform **prioritises** where to send the ROV, **documents** what was found, and **packages** results into survey-ready PDFs — reducing manual triage time while keeping humans in the verification loop.
