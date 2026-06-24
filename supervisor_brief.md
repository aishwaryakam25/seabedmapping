# NautiCAI Seabed PoC — Supervisor Brief

**Date:** June 2026  
**Project:** Side-scan sonar object detection & survey intelligence (SEABED PoC)  
**Demo URL (local):** http://127.0.0.1:8000/review

---

## One-page summary

### What we built

An end-to-end **proof-of-concept** for AI-assisted seabed mapping: load side-scan sonar frames, run a trained **YOLOv8** detector, review contacts in a survey console, triage ROV follow-up, and export **client-ready PDF reports** — plus **10 differentiated survey-intelligence features** not found in generic object-detection demos.

### Model & data

| Item | Detail |
|------|--------|
| Architecture | YOLOv8 (Ultralytics) |
| Classes | 8 — rock, boulder, wreck, pipeline, debris, cable, scour, obstruction |
| Training set | **2,400 images** (800 synthetic + 1,600 UATD) |
| Split | 1,480 train / 920 val |
| Training | 20 epochs on CPU |
| Best weights | `runs/detect/sonar_yolo/weights/best.pt` |
| Val mAP50 | **~0.97** (epoch 20) |
| Val mAP50-95 | **~0.68** |

### Application modules (live)

| Module | Route | Purpose |
|--------|-------|---------|
| Data Review | `/review` | Load frames, run inference, verify/reject contacts |
| Survey Intelligence | `/intelligence` | 10 novel analytics features (see below) |
| Dashboard | `/dashboard` | KPIs, class/risk breakdown, ROV ROI |
| Survey Map | `/map` | Plan view, corridor risk, contact layers |
| 3D Seabed Twin | `/twin` | Per-frame 3D contact pins |
| ROV Ops | `/rov` | Flag → schedule → close dive workflow |
| Survey Workflow | `/workflow` | Pipeline overview |
| Reports | `/reports` | PDF generation hub |

### Ten differentiated features (Survey Intelligence)

1. **Acoustic shadow → height & burial** — estimates object height and burial state from box geometry  
2. **Cross-line contact fusion** — merges contacts seen on multiple lines by position  
3. **Contact lifecycle & clearance certificate** — AI → desk review → ROV → clearance PDF  
4. **Site fingerprint learning** — analyst rejections flag similar false positives  
5. **Corridor risk heatmap** — chainage risk along cable/turbine routes  
6. **Contractor survey diff** — compare two saved survey snapshots  
7. **Explainable sonar attribution** — human-readable cues per detection  
8. **ROV dollar & carbon saved** — ROI from auto-accept triage  
9. **Temporal 4D change detection** — diff survey snapshots over time  
10. **Live install urgency** — rank contacts by proximity to installation date  

### Deliverables

- Multi-frame **survey PDF** (detections, risk, ROV plan, intelligence summary)  
- Standalone **clearance certificate** PDF (`/intelligence` → Issue clearance)  
- GeoJSON / CSV contact export  
- ROV deployment queue JSON  

### PoC limitations (transparent)

- Positions and ranges are **illustrative** (frame-derived), not full GIS/geo-referenced survey.  
- Shadow/burial physics are **heuristic** — suitable for demo, not certified metrology.  
- Persistence is **browser localStorage** — no production database yet.  
- Deployment benchmark on **GPU/TensorRT** is documented; full GPU numbers require a CUDA machine or Colab.

### Suggested next steps (production path)

1. Geo-referenced ingestion (NAV / layback / beam geometry)  
2. GPU deployment (TensorRT INT8 on vessel edge or Jetson)  
3. Backend database + multi-user survey projects  
4. Calibration of shadow-height model against ROV ground truth  
5. Tiled inference for continuous survey strips  

---

## Email draft (copy & send)

**Subject:** NautiCAI Seabed Mapping PoC — demo ready (YOLO sonar detection + survey intelligence)

Dear [Name],

I wanted to share an update on the **NautiCAI seabed mapping proof-of-concept** we have been developing.

We now have a working end-to-end demo that covers the full survey workflow: **load side-scan sonar frames → AI detection → analyst review → ROV triage → PDF deliverables**. The detector is a **YOLOv8 model** trained on **2,400 images** (synthetic + UATD), reaching approximately **97% mAP50** on the validation split across **8 seabed object classes**.

Beyond standard bounding-box detection, we implemented **ten survey-intelligence features** aimed at offshore wind and cable-route workflows — including cross-line contact fusion, corridor risk heatmaps, explainable sonar attribution, ROV cost/CO₂ savings estimates, 4D change detection between surveys, and automated clearance certificate generation.

**Key capabilities at a glance:**

- **Data Review console** — multi-frame upload, confidence thresholding, measurement tool, mosaic view  
- **Survey Intelligence hub** — differentiated analytics in one place (`/intelligence`)  
- **ROV Ops console** — queue high-risk contacts for verification  
- **PDF reports** — multi-frame survey report with ROV follow-up plan and intelligence summary  

The system runs locally via a single Python server (`python server.py`) and is suitable for a **live walkthrough** at your convenience.

I have attached / linked a one-page brief with technical detail. Happy to schedule a short demo or discuss the path from PoC to pilot deployment (GPU edge inference, geo-referencing, and ROV calibration).

Best regards,  
[Your name]  
[Your role / affiliation]

---

## Quick demo script (5–10 min)

1. Open **/review** → load 2–4 sonar frames → show detections and detail panel (height, burial, attribution).  
2. Mark one contact **Verified**, reject one **false positive** → show site fingerprint warning on similar contact.  
3. Open **/intelligence** → fusion table, corridor risk, ROI, set install date → urgency ranking.  
4. Open **/rov** → flag a high-risk contact → schedule.  
5. Back to **/review** → **Generate report (PDF)** → show ROV + intelligence sections.  
6. **/intelligence** → **Issue clearance certificate**.  

---

## Benchmark reference

See `benchmark_results.csv` (generated by `yolo_export_benchmark.py`) for PyTorch vs ONNX vs INT8 latency and mAP comparison on this machine.
