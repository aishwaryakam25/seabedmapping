"""
report.py -- builds the multi-page nauticAi "Seabed Object Detection Report" PDF
from a sonar image and a list of detections.

    build_report(image_bytes, detections, meta) -> bytes  (a PDF)

detections: list of {"cls": str, "conf": float, "box": [x1,y1,x2,y2]}
meta:       optional dict (project, location, survey_id, sensor, vessel,
            operator, frame_m, threshold, filename)

Dependencies:  reportlab, matplotlib, opencv-python, numpy
Class-agnostic: any class names work; known seabed classes get fixed colours
and risk levels, everything else falls back sensibly.
"""
import io
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, PageBreak, Image, Table, TableStyle, Flowable)
from reportlab.pdfgen import canvas

# ---- palette ---------------------------------------------------------------
NAVY = colors.HexColor('#0b2a38'); CYAN = colors.HexColor('#16b4c9')
CYAND = colors.HexColor('#0e8fa3'); MUT = colors.HexColor('#5d7480')
LINE = colors.HexColor('#dbe5e8'); ZEB = colors.HexColor('#f3f8f9')
RISKCOL = {'Critical': colors.HexColor('#b3261e'), 'High': colors.HexColor('#d2691e'),
           'Medium': colors.HexColor('#b5791f'), 'Low': colors.HexColor('#15803d')}

# class -> hex colour (BGR derived on the fly for cv2); unknown -> hashed palette
CLASSHEX = {'rock': '#b5791f', 'boulder': '#c2410c', 'wreck': '#be185d',
            'pipeline': '#2f7fce', 'cable': '#0aa0b4', 'cable_crossing': '#0aa0b4',
            'debris': '#8b5cf6', 'uxo': '#b3261e', 'mine': '#b3261e',
            'anchor': '#0e8fa3', 'obstruction': '#566270', 'scour': '#15803d'}
FALLBACK = ['#0e7c86', '#c2410c', '#3b6fb6', '#7c3aed', '#b5791f', '#0891b2',
            '#be185d', '#15803d', '#4b5563', '#9333ea']
# class -> risk; unknown -> Medium
CLASSRISK = {'wreck': 'Critical', 'uxo': 'Critical', 'mine': 'Critical',
             'pipeline': 'High', 'cable': 'High', 'cable_crossing': 'High',
             'obstruction': 'High', 'boulder': 'Medium', 'anchor': 'Medium',
             'debris': 'Medium', 'scour': 'Medium', 'rock': 'Low'}


def _key(c):
    return str(c).lower().replace(' ', '_')


def class_hex(c):
    k = _key(c)
    if k in CLASSHEX:
        return CLASSHEX[k]
    h = 0
    for ch in k:
        h = (h * 31 + ord(ch)) & 0xffffffff
    return FALLBACK[h % len(FALLBACK)]


def risk_of(c):
    return CLASSRISK.get(_key(c), 'Medium')


def _bgr(hexv):
    hexv = hexv.lstrip('#')
    return (int(hexv[4:6], 16), int(hexv[2:4], 16), int(hexv[0:2], 16))


def _annotated_png(img, dets):
    a = img.copy()
    for d in dets:
        b = d['box']; col = _bgr(class_hex(d['cls']))
        cv2.rectangle(a, (b[0], b[1]), (b[2], b[3]), col, 2)
        t = "%s %d%%" % (str(d['cls']).upper(), round(d['conf'] * 100))
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y0 = max(th + 6, b[1])
        cv2.rectangle(a, (b[0], y0 - th - 6), (b[0] + tw + 6, y0), col, -1)
        cv2.putText(a, t, (b[0] + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode('.png', a)
    return io.BytesIO(buf.tobytes())


def _crop_png(img, box, size=(190, 140)):
    h, w = img.shape[:2]; p = 8
    x1 = max(0, box[0] - p); y1 = max(0, box[1] - p)
    x2 = min(w, box[2] + p); y2 = min(h, box[3] + p)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((size[1], size[0], 3), np.uint8)
    crop = cv2.resize(crop, size)
    ok, buf = cv2.imencode('.png', crop)
    return io.BytesIO(buf.tobytes())


def _pie_png(counts):
    labels = list(counts); sizes = [counts[k] for k in labels]
    cols = [class_hex(k) for k in labels]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=150)
    _, _, at = ax.pie(sizes, labels=[l.title() for l in labels], colors=cols,
                      autopct='%1.0f%%', startangle=90, pctdistance=0.74,
                      textprops={'fontsize': 9, 'color': '#0b2a38'},
                      wedgeprops={'linewidth': 1.6, 'edgecolor': 'white'})
    for a in at:
        a.set_color('white'); a.set_fontweight('bold'); a.set_fontsize(8.5)
    ax.set_aspect('equal'); fig.patch.set_alpha(0); plt.tight_layout()
    bio = io.BytesIO(); fig.savefig(bio, format='png', transparent=True,
                                    bbox_inches='tight'); plt.close(fig)
    bio.seek(0); return bio


def build_report(image_bytes, detections, meta=None):
    meta = meta or {}
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    IMGH, IMGW = img.shape[:2]
    GW = float(meta.get('frame_m', 120.0)); GH = GW * IMGH / IMGW

    def sizem(b):
        return ((b[2] - b[0]) / IMGW * GW, (b[3] - b[1]) / IMGH * GH)

    dets = list(detections)
    n = len(dets)
    avg = (sum(d['conf'] for d in dets) / n) if n else 0.0
    counts = {}
    for d in dets:
        counts[d['cls']] = counts.get(d['cls'], 0) + 1
    risks = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
    for d in dets:
        risks[risk_of(d['cls'])] += 1
    hi = risks['Critical'] + risks['High']

    PAGEW, PAGEH = A4; LM = RM = 16 * mm; PW = PAGEW - LM - RM
    ss = getSampleStyleSheet()

    def stl(nm, parent=None, **k):
        return ParagraphStyle(nm, parent=parent or ss['Normal'], **k)
    H1 = stl('H1', fontSize=14.5, textColor=NAVY, leading=17, fontName='Helvetica-Bold')
    SECN = stl('SECN', fontSize=9, textColor=CYAND, fontName='Helvetica-Bold')
    BODY = stl('BODY', fontSize=10, leading=15, textColor=colors.HexColor('#1c2b33'))
    SMALL = stl('SM', fontSize=8, leading=11, textColor=MUT)
    CELL = stl('CELL', fontSize=9.2, leading=12, textColor=colors.HexColor('#1c2b33'))
    CELLB = stl('CELLB', fontSize=9.2, leading=12, textColor=NAVY, fontName='Helvetica-Bold')
    CELLR = stl('CELLR', parent=CELL, alignment=TA_RIGHT)
    CELLH = stl('CELLH', fontSize=8.6, leading=11, textColor=colors.white, fontName='Helvetica-Bold')
    CELLHR = stl('CELLHR', parent=CELLH, alignment=TA_RIGHT)
    WHT = stl('WHT', fontSize=9.2, leading=12, textColor=colors.white, fontName='Helvetica-Bold', alignment=TA_CENTER)

    class Rule(Flowable):
        def __init__(s, w, c=CYAN, t=1.4): s.w = w; s.c = c; s.t = t
        def wrap(s, a, b): return (s.w, s.t)
        def draw(s):
            s.canv.setStrokeColor(s.c); s.canv.setLineWidth(s.t); s.canv.line(0, 0, s.w, 0)

    def section(num, title):
        return [Spacer(1, 2), Paragraph('SECTION %s' % num, SECN), Paragraph(title, H1), Rule(PW), Spacer(1, 9)]

    def hdr_table(rows, col0=46 * mm, zebra=False):
        data = [[Paragraph(k, CELL), Paragraph('<b>%s</b>' % v, CELLB)] for k, v in rows]
        t = Table(data, colWidths=[col0, PW - col0])
        sty = [('LINEBELOW', (0, 0), (-1, -1), 0.4, LINE), ('TOPPADDING', (0, 0), (-1, -1), 5.5),
               ('BOTTOMPADDING', (0, 0), (-1, -1), 5.5), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
        if zebra:
            for r in range(0, len(data), 2):
                sty.append(('BACKGROUND', (0, r), (-1, r), ZEB))
        t.setStyle(TableStyle(sty)); return t

    ann = _annotated_png(img, dets)
    story = [Spacer(1, 54 * mm)]
    cover = [['Survey ID', meta.get('survey_id', 'SURV-2026-001')],
             ['Project', meta.get('project', 'Offshore Wind Site Assessment')],
             ['Location', meta.get('location', 'North Sea Sector A')],
             ['Sensor', meta.get('sensor', 'Side-scan sonar (SSS)')],
             ['Source image', meta.get('filename', 'sonar_frame')],
             ['Generated', datetime.now().strftime('%d %B %Y, %H:%M')]]
    story += [hdr_table(cover, 46 * mm, True), Spacer(1, 11),
              Image(ann, width=PW, height=PW * IMGH / IMGW), Spacer(1, 5),
              Paragraph('Analysed side-scan frame. Positions, ranges and object sizes in this report are '
                        'illustrative \u2014 the source imagery is not geo-referenced.', SMALL), PageBreak()]

    # 01 exec + 02 overview
    story += section('01', 'Executive Summary')
    es = [['Survey frame', '%.0f \u00d7 %.0f m (illustrative)' % (GW, GH)],
          ['Image resolution', '%d \u00d7 %d px' % (IMGW, IMGH)],
          ['Objects detected', str(n)],
          ['High-risk (Critical / High)', str(hi)],
          ['Medium / Low risk', '%d / %d' % (risks['Medium'], risks['Low'])],
          ['Average confidence', '%.1f%%' % (avg * 100)]]
    cls_summary = ', '.join('%s (%d)' % (c.title(), counts[c]) for c in sorted(counts)) or 'no objects'
    story += [hdr_table(es, 70 * mm, True), Spacer(1, 9),
              Paragraph('The system analysed the uploaded side-scan imagery and detected %d object(s): %s. '
                        '%d were assessed as high-risk. %s' % (
                            n, cls_summary, hi,
                            'Further inspection is recommended for the high-risk contacts before installation activity.'
                            if hi else 'No high-risk contacts were flagged in this frame.'), BODY), Spacer(1, 12)]

    story += section('02', 'Detection Overview')
    if n:
        crows = [[Paragraph('Class', CELLH), Paragraph('Count', CELLHR), Paragraph('Risk', CELLH)]]
        sty = [('BACKGROUND', (0, 0), (-1, 0), NAVY), ('GRID', (0, 0), (-1, -1), 0.4, LINE),
               ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
               ('ALIGN', (1, 0), (1, -1), 'RIGHT')]
        ri = 1
        for c in sorted(counts):
            crows.append([Paragraph(c.title(), CELL), Paragraph(str(counts[c]), CELLR), Paragraph(risk_of(c), CELL)])
            if ri % 2 == 0:
                sty.append(('BACKGROUND', (0, ri), (-1, ri), ZEB))
            ri += 1
        crows.append([Paragraph('Total', CELLB), Paragraph('<b>%d</b>' % n, CELLR), Paragraph('', CELL)])
        sty.append(('LINEABOVE', (0, len(crows) - 1), (-1, len(crows) - 1), 0.8, NAVY))
        ctbl = Table(crows, colWidths=[34 * mm, 20 * mm, 28 * mm]); ctbl.setStyle(TableStyle(sty))
        two = Table([[ctbl, Image(_pie_png(counts), width=74 * mm, height=74 * mm)]],
                    colWidths=[84 * mm, PW - 84 * mm])
        two.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('ALIGN', (1, 0), (1, 0), 'CENTER')]))
        story += [two]
    else:
        story += [Paragraph('No objects were detected in this frame.', BODY)]
    story += [PageBreak()]

    # 03 image
    story += section('03', 'Sonar Image Analysis')
    story += [Image(_annotated_png(img, dets), width=PW, height=PW * IMGH / IMGW), Spacer(1, 5),
              Paragraph('Detected objects with bounding boxes and class confidence. '
                        'Box colours correspond to object class.', SMALL), PageBreak()]

    # 04 detection table
    story += section('04', 'Detection Details')
    if n:
        hd = [Paragraph('ID', CELLH), Paragraph('Class', CELLH), Paragraph('Conf', CELLHR),
              Paragraph('Risk', WHT), Paragraph('Location (px)', CELLH), Paragraph('Est. size', CELLHR)]
        data = [hd]
        sty = [('BACKGROUND', (0, 0), (-1, 0), NAVY), ('GRID', (0, 0), (-1, -1), 0.4, LINE),
               ('TOPPADDING', (0, 0), (-1, -1), 5.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5.5),
               ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
               ('ALIGN', (3, 0), (3, -1), 'CENTER'), ('ALIGN', (5, 0), (5, -1), 'RIGHT')]
        order = sorted(range(n), key=lambda i: -dets[i]['conf'])
        for row, i in enumerate(order, 1):
            d = dets[i]; b = d['box']; rk = risk_of(d['cls']); sx, sy = sizem(b)
            cxp = (b[0] + b[2]) // 2; cyp = (b[1] + b[3]) // 2
            data.append([Paragraph('OBJ-%03d' % row, CELLB), Paragraph(str(d['cls']).title(), CELL),
                         Paragraph('%d%%' % round(d['conf'] * 100), CELLR),
                         Paragraph('<font color="white"><b>%s</b></font>' % rk, WHT),
                         Paragraph('X:%d  Y:%d' % (cxp, cyp), CELL),
                         Paragraph('%.1f \u00d7 %.1f m' % (sx, sy), CELLR)])
            sty.append(('BACKGROUND', (3, row), (3, row), RISKCOL[rk]))
            if row % 2 == 0:
                for cc in (0, 1, 2, 4, 5):
                    sty.append(('BACKGROUND', (cc, row), (cc, row), ZEB))
        dt = Table(data, colWidths=[20 * mm, 26 * mm, 16 * mm, 22 * mm, 32 * mm, PW - 116 * mm],
                   repeatRows=1)
        dt.setStyle(TableStyle(sty))
        story += [dt]
    else:
        story += [Paragraph('No detections to list.', BODY)]
    story += [PageBreak()]

    # 05 gallery
    story += section('05', 'Object Snapshots')
    if n:
        cells = []
        for row, i in enumerate(sorted(range(n), key=lambda i: -dets[i]['conf']), 1):
            d = dets[i]; rk = risk_of(d['cls']); hexr = '#' + RISKCOL[rk].hexval()[2:]
            cap = Table([[Paragraph('<b>%s #%d</b>' % (str(d['cls']).title(), row), CELL)],
                         [Paragraph('Confidence %d%%  \u00b7  <font color="%s"><b>%s</b></font>' % (
                             round(d['conf'] * 100), hexr, rk), SMALL)]])
            cap.setStyle(TableStyle([('TOPPADDING', (0, 0), (-1, -1), 1), ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                                     ('LEFTPADDING', (0, 0), (-1, -1), 6)]))
            cell = Table([[Image(_crop_png(img, d['box']), width=53 * mm, height=39 * mm)], [cap]])
            cell.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 0.5, LINE), ('TOPPADDING', (0, 1), (0, 1), 4),
                                      ('BOTTOMPADDING', (0, 1), (0, 1), 6), ('TOPPADDING', (0, 0), (0, 0), 0),
                                      ('LEFTPADDING', (0, 0), (0, 0), 0), ('RIGHTPADDING', (0, 0), (0, 0), 0)]))
            cells.append(cell)
        rows_g = [cells[k:k + 3] for k in range(0, len(cells), 3)]
        for r in rows_g:
            while len(r) < 3:
                r.append('')
        gal = Table(rows_g, colWidths=[PW / 3] * 3)
        gal.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 5),
                                 ('BOTTOMPADDING', (0, 0), (-1, -1), 5), ('LEFTPADDING', (0, 0), (-1, -1), 3),
                                 ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
        story += [gal]
    else:
        story += [Paragraph('No object snapshots available.', BODY)]
    story += [PageBreak()]

    # 06 risk + 07 recommendations
    story += section('06', 'Risk Assessment')
    txts = {'Critical': 'Critical contacts \u2014 treat as hard constraints pending inspection.',
            'High': 'High-risk linear/obstruction features \u2014 maintain clearance and verify before route works.',
            'Medium': 'Medium-risk targets \u2014 may require clearance.',
            'Low': 'Low-risk isolated targets \u2014 note for foundation suitability.'}
    present = [lvl for lvl in ['Critical', 'High', 'Medium', 'Low'] if risks[lvl] > 0]
    if present:
        rrows = []
        for lvl in present:
            items = [d['cls'] for d in dets if risk_of(d['cls']) == lvl]
            cnt = ', '.join('%s \u00d7%d' % (t.title(), items.count(t)) for t in sorted(set(items)))
            rrows.append([Paragraph('<font color="white"><b>%s</b></font>' % lvl, WHT),
                          Paragraph(txts[lvl], CELL), Paragraph(cnt, SMALL)])
        rsty = [('GRID', (0, 0), (-1, -1), 0.4, LINE), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('ALIGN', (0, 0), (0, -1), 'CENTER')]
        for r, lvl in enumerate(present):
            rsty.append(('BACKGROUND', (0, r), (0, r), RISKCOL[lvl]))
        rt = Table(rrows, colWidths=[24 * mm, PW - 74 * mm, 50 * mm]); rt.setStyle(TableStyle(rsty))
        story += [rt]
    else:
        story += [Paragraph('No risk-rated contacts in this frame.', BODY)]
    story += [Spacer(1, 12)]

    story += section('07', 'Recommendations')
    recs = []
    present_classes = set(_key(c) for c in counts)
    if {'uxo', 'mine'} & present_classes:
        recs.append('Suspected UXO present \u2014 stop work in the vicinity and arrange a specialist assessment.')
    if 'wreck' in present_classes:
        recs.append('Conduct an ROV inspection of the wreck contact before any installation activity.')
    if {'pipeline', 'cable', 'cable_crossing'} & present_classes:
        recs.append('Verify the pipeline / cable crossing manually and confirm separation distances.')
    if 'boulder' in present_classes:
        recs.append('Avoid foundation or turbine placement near boulder clusters; consider clearance.')
    if 'debris' in present_classes:
        recs.append('Remove or route around debris prior to cable laying.')
    if 'rock' in present_classes:
        recs.append('Rocky seabed noted \u2014 assess foundation suitability for the affected area.')
    if not recs:
        recs.append('No high-priority targets detected; continue routine survey coverage.')
    recs.append('Perform an additional survey pass around the highest-risk detections.')
    rec_rows = [[Paragraph('<font color="%s"><b>%d</b></font>' % ('#' + CYAND.hexval()[2:], i), CELLB),
                 Paragraph(r, BODY)] for i, r in enumerate(recs, 1)]
    rect = Table(rec_rows, colWidths=[10 * mm, PW - 10 * mm])
    rect.setStyle(TableStyle([('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                              ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -2), 0.3, LINE)]))
    story += [rect, PageBreak()]

    # 08 appendix
    story += section('08', 'Appendix \u2014 Technical Details')
    ap = [['Model', meta.get('model', 'YOLOv8 sonar detection model')],
          ['Input resolution', '%d \u00d7 %d px' % (IMGW, IMGH)],
          ['Detection threshold', str(meta.get('threshold', '0.25'))],
          ['Objects detected', str(n)],
          ['Average confidence', '%.1f%%' % (avg * 100)],
          ['Datum', 'Local grid (m) \u2014 illustrative'],
          ['Report generated', datetime.now().strftime('%d %B %Y, %H:%M')]]
    story += [hdr_table(ap, 55 * mm, True), Spacer(1, 12),
              Paragraph('<b>Disclaimer.</b> Positions, ranges and object sizes are illustrative and derived from '
                        'each detection\u2019s position within the frame. The source imagery is not geo-referenced. '
                        'This is a proof-of-concept demonstration, not a survey-grade deliverable.', SMALL)]

    # ---- page furniture -----------------------------------------------------
    def wordmark(c, x, y, size):
        c.setFont('Helvetica-Bold', size); c.setFillColor(colors.white); c.drawString(x, y, 'nautic')
        w = c.stringWidth('nautic', 'Helvetica-Bold', size); c.setFillColor(CYAN); c.drawString(x + w, y, 'Ai')

    def deco(c, page, total):
        c.saveState()
        if page == 1:
            c.setFillColor(NAVY); c.rect(0, PAGEH - 72 * mm, PAGEW, 72 * mm, fill=1, stroke=0)
            c.setFillColor(CYAN); c.rect(0, PAGEH - 72 * mm, PAGEW, 2.2, fill=1, stroke=0)
            wordmark(c, LM, PAGEH - 34 * mm, 30)
            c.setFont('Helvetica', 8.5); c.setFillColor(CYAN)
            c.drawString(LM, PAGEH - 40 * mm, 'INTELLIGENT MARITIME AWARENESS')
            c.setFont('Helvetica-Bold', 16); c.setFillColor(colors.white)
            c.drawString(LM, PAGEH - 54 * mm, 'Seabed Object Detection Report')
            c.setFont('Helvetica', 10); c.setFillColor(colors.HexColor('#9fc6cf'))
            c.drawString(LM, PAGEH - 61 * mm, 'Automated side-scan sonar analysis  \u00b7  Proof of concept')
        else:
            c.setFillColor(NAVY); c.rect(0, PAGEH - 15 * mm, PAGEW, 15 * mm, fill=1, stroke=0)
            c.setFillColor(CYAN); c.rect(0, PAGEH - 15 * mm, PAGEW, 1.6, fill=1, stroke=0)
            wordmark(c, LM, PAGEH - 10.3 * mm, 12)
            c.setFont('Helvetica', 8.2); c.setFillColor(colors.HexColor('#9fc6cf'))
            c.drawRightString(PAGEW - RM, PAGEH - 10 * mm, 'SEABED OBJECT DETECTION REPORT')
        c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(LM, 13 * mm, PAGEW - RM, 13 * mm)
        c.setFont('Helvetica', 7.6); c.setFillColor(MUT)
        c.drawString(LM, 9 * mm, 'nauticAi  \u00b7  Seabed Object Detection  \u00b7  Proof of Concept')
        c.drawRightString(PAGEW - RM, 9 * mm, 'Page %d of %d' % (page, total))
        c.restoreState()

    class NumCanvas(canvas.Canvas):
        def __init__(self, *a, **k):
            canvas.Canvas.__init__(self, *a, **k); self._saved = []
        def showPage(self):
            self._saved.append(dict(self.__dict__)); self._startPage()
        def save(self):
            tot = len(self._saved)
            for s in self._saved:
                self.__dict__.update(s); deco(self, self._pageNumber, tot); canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

    out = io.BytesIO()
    frame = Frame(LM, 18 * mm, PW, PAGEH - 18 * mm - 26 * mm, id='f')
    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=LM, rightMargin=RM,
                          topMargin=26 * mm, bottomMargin=18 * mm,
                          title='Seabed Object Detection Report - nauticAi')
    doc.addPageTemplates([PageTemplate(id='main', frames=[frame])])
    doc.build(story, canvasmaker=NumCanvas)
    out.seek(0)
    return out.getvalue()
