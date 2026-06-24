"""
report.py -- builds the multi-page NautiCAI "Seabed Object Detection Report" PDF
from a sonar image and a list of detections.

    build_report(image_bytes, detections, meta) -> bytes  (a PDF)

detections: list of {"cls": str, "conf": float, "box": [x1,y1,x2,y2]}
meta:       optional dict (project, location, survey_id, sensor, vessel,
            operator, frame_m, threshold, filename)

Dependencies:  reportlab, matplotlib, opencv-python, numpy
Class-agnostic: any class names work; known seabed classes get fixed colours
and risk levels, everything else falls back sensibly.
"""
import base64
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from brand import BRAND, BRAND_PREFIX, BRAND_SUFFIX, TAGLINE, WEBSITE, WEBSITE_LABEL

_log = logging.getLogger(__name__)

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, PageBreak, Image, Table, TableStyle, Flowable)
from reportlab.pdfgen import canvas

_HERE = Path(__file__).resolve().parent
_LOGO_PATHS = (
    _HERE / "assets" / "nauticai-logo.png",
    _HERE / "assets" / "nauticai-logo-nav.png",
)
COVER_BAND = 78 * mm

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


def _logo_path(nav=False):
    """nav=True prefers the horizontal wordmark (better on navy PDF headers)."""
    nav_p = _HERE / "assets" / "nauticai-logo-nav.png"
    full_p = _HERE / "assets" / "nauticai-logo.png"
    order = (nav_p, full_p) if nav else (full_p, nav_p)
    for p in order:
        if p.exists():
            return p
    return None


def _wordmark(c, x, y, size):
    c.setFont("Helvetica-Bold", size)
    c.setFillColor(colors.white)
    c.drawString(x, y, BRAND_PREFIX)
    w = c.stringWidth(BRAND_PREFIX, "Helvetica-Bold", size)
    c.setFillColor(CYAN)
    c.drawString(x + w, y, BRAND_SUFFIX)


def _draw_website_link(c, x, y, font_size=7.5, link_name=None):
    c.setFont("Helvetica", font_size)
    c.setFillColor(CYAN)
    c.drawString(x, y, WEBSITE_LABEL)
    if link_name:
        tw = c.stringWidth(WEBSITE_LABEL, "Helvetica", font_size)
        c.linkURL(
            WEBSITE,
            (x, y - 1 * mm, x + tw, y + font_size * 0.45 + 1 * mm),
            relative=0,
            name=link_name,
        )


def _draw_header_logo(c, x, y, max_h, max_w=None, nav=True):
    """Draw logo clipped to header band. Returns (width, height)."""
    path = _logo_path(nav=nav)
    if not path:
        _wordmark(c, x, y, 11)
        return 40 * mm, 5 * mm
    ir = ImageReader(str(path))
    iw, ih = ir.getSize()
    logo_h = max_h
    logo_w = logo_h * iw / ih
    if max_w and logo_w > max_w:
        logo_w = max_w
        logo_h = logo_w * ih / iw
    c.drawImage(ir, x, y, width=logo_w, height=logo_h, mask="auto", preserveAspectRatio=True)
    return logo_w, logo_h


def _draw_cover_logo_block(c, x, y_top, logo_h_mm, site_below=True, site_size=7.5, link_name=None):
    """Cover-page logo + optional website (kept inside the navy band). Returns (width, bottom_y)."""
    logo_h = logo_h_mm * mm
    y_bot = y_top - logo_h
    logo_w, _ = _draw_header_logo(c, x, y_bot, logo_h, nav=True)
    bottom = y_bot
    if site_below:
        site_y = y_bot - 4 * mm
        _draw_website_link(c, x, site_y, site_size, link_name)
        bottom = site_y - 2.5 * mm
    return logo_w, bottom


def _draw_inner_header(c, pagew, pageh, lm, rm):
    """Compact running header — logo + title only; website stays in the footer."""
    header_h = 16 * mm
    y0 = pageh - header_h
    c.setFillColor(NAVY)
    c.rect(0, y0, pagew, header_h, fill=1, stroke=0)
    c.setFillColor(CYAN)
    c.rect(0, pageh - 2, pagew, 2, fill=1, stroke=0)
    logo_h = 8 * mm
    logo_y = y0 + (header_h - logo_h) / 2
    _draw_header_logo(c, lm, logo_y, logo_h, max_w=55 * mm, nav=True)
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#b8d4db"))
    c.drawRightString(pagew - rm, y0 + header_h / 2 - 3, "SEABED OBJECT DETECTION REPORT")


def _pdf_page_furniture(brand, cover_subtitle):
    PAGEW, PAGEH = A4
    LM = RM = 16 * mm

    def deco(c, page, total):
        c.saveState()
        if page == 1:
            c.setFillColor(NAVY)
            c.rect(0, PAGEH - COVER_BAND, PAGEW, COVER_BAND, fill=1, stroke=0)
            c.setFillColor(CYAN)
            c.rect(0, PAGEH - COVER_BAND, PAGEW, 2.2, fill=1, stroke=0)
            _, bottom = _draw_cover_logo_block(
                c, LM, PAGEH - 18 * mm, 10, site_below=True, site_size=8, link_name="cover_site_link"
            )
            tag_y = bottom - 4 * mm
            c.setFont("Helvetica", 8.5)
            c.setFillColor(CYAN)
            c.drawString(LM, tag_y, TAGLINE.upper())
            c.setFont("Helvetica-Bold", 16)
            c.setFillColor(colors.white)
            c.drawString(LM, tag_y - 11 * mm, "Seabed Object Detection Report")
            c.setFont("Helvetica", 10)
            c.setFillColor(colors.HexColor("#9fc6cf"))
            c.drawString(LM, tag_y - 18 * mm, cover_subtitle)
        else:
            _draw_inner_header(c, PAGEW, PAGEH, LM, RM)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.5)
        c.line(LM, 13 * mm, PAGEW - RM, 13 * mm)
        c.setFont("Helvetica", 7.6)
        c.setFillColor(MUT)
        c.drawString(
            LM,
            9 * mm,
            "%s  \u00b7  %s  \u00b7  Seabed Object Detection  \u00b7  Proof of Concept"
            % (brand, WEBSITE_LABEL),
        )
        c.drawRightString(PAGEW - RM, 9 * mm, "Page %d of %d" % (page, total))
        c.restoreState()

    class NumCanvas(canvas.Canvas):
        def __init__(self, *a, **k):
            canvas.Canvas.__init__(self, *a, **k)
            self._saved = []

        def showPage(self):
            self._saved.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            tot = len(self._saved)
            for s in self._saved:
                self.__dict__.update(s)
                deco(self, self._pageNumber, tot)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

    return NumCanvas


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


def _cap_cls(name):
    return ("" + str(name)).replace("_", " ").title()


def _clamp_box(box, w, h):
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1 = max(0, min(w - 2, x1))
    y1 = max(0, min(h - 2, y1))
    x2 = max(x1 + 2, min(w, x2))
    y2 = max(y1 + 2, min(h, y2))
    return [x1, y1, x2, y2]


def _det_boxes_for_image(dets, img_w, img_h, nat_w=None, nat_h=None):
    """Map detection boxes to the actual image pixels (fixes thumbnail vs full-res mismatch)."""
    nw = int(nat_w or img_w) or img_w
    nh = int(nat_h or img_h) or img_h
    sx = img_w / max(1, nw)
    sy = img_h / max(1, nh)
    out = []
    for d in dets:
        nd = dict(d)
        b = d.get("box") or [0, 0, 0, 0]
        if d.get("cxn") is not None and d.get("cyn") is not None:
            cx = float(d["cxn"]) * img_w
            cy = float(d["cyn"]) * img_h
            obw = max(4, (b[2] - b[0]) * sx) if len(b) >= 4 and b[2] > b[0] else float(d.get("wnorm", 0.06) or 0.06) * img_w
            obh = max(4, (b[3] - b[1]) * sy) if len(b) >= 4 and b[3] > b[1] else obw * 0.72
            nd["box"] = _clamp_box([cx - obw / 2, cy - obh / 2, cx + obw / 2, cy + obh / 2], img_w, img_h)
        else:
            nd["box"] = _clamp_box([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], img_w, img_h)
        out.append(nd)
    return out


def _annotated_png(img, dets, nat_w=None, nat_h=None):
    h, w = img.shape[:2]
    scaled = _det_boxes_for_image(dets, w, h, nat_w, nat_h)
    return io.BytesIO(cv2.imencode(".png", annotate_image(img, scaled))[1].tobytes())


def _fit_image_flowable(png_bio, pw, max_h):
    """Scale a raster to fit ReportLab page width and height (portrait sonar safe)."""
    from reportlab.lib.utils import ImageReader

    png_bio.seek(0)
    ir = ImageReader(png_bio)
    iw, ih = ir.getSize()
    if not iw or not ih:
        png_bio.seek(0)
        return Image(png_bio, width=min(pw, max_h), height=min(max_h, pw * 0.75))
    scale = min(pw / float(iw), max_h / float(ih))
    w, h = iw * scale, ih * scale
    png_bio.seek(0)
    return Image(png_bio, width=w, height=h)


def annotate_image(img, dets, font_scale=None, thickness=None):
    """Draw boxes with labels that avoid overlapping when detections are close."""
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    if font_scale is None:
        font_scale = max(0.55, min(w, h) / 650.0)
    if thickness is None:
        thickness = max(2, int(min(w, h) / 350))
    pad = max(4, int(font_scale * 6))
    box_th = max(2, thickness)
    placed = []
    items = []

    for d in dets:
        b = _clamp_box(d["box"], w, h)
        cls = str(d["cls"])
        conf = d.get("conf")
        text = "%s %d%%" % (_cap_cls(cls), round(float(conf) * 100)) if conf is not None else _cap_cls(cls)
        tw, tht = cv2.getTextSize(text, font, font_scale, thickness)[0]
        lw, lh = tw + pad * 2, tht + pad * 2
        bx1, by1, bx2, by2 = b

        def rect(x, y):
            return [x, y, x + lw, y + lh]

        def fits(r):
            return not any(
                not (r[2] + 2 <= p[0] or p[2] + 2 <= r[0] or r[3] + 2 <= p[1] or p[3] + 2 <= r[1])
                for p in placed
            )

        lx = max(0, min(bx1, w - lw))
        ly = max(0, by1 - lh - 3)
        r = rect(lx, ly)
        for attempt in range(48):
            if fits(r):
                break
            if attempt < 16:
                ly -= lh + 2
                if ly < 0:
                    ly = by2 + 3
            elif attempt < 32:
                ly = by2 + 3
                lx = min(lx + max(10, lw // 3), w - lw)
            else:
                ly = min(by2 + 3, h - lh)
                lx = min(bx1, w - lw)
            r = rect(max(0, lx), max(0, min(ly, h - lh)))
        placed.append(r)
        items.append((b, text, r[0], r[1], lw, lh, class_hex(cls)))

    for b, text, lx, ly, lw, lh, hexcol in items:
        col = _bgr(hexcol)
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), col, box_th)
        cv2.rectangle(out, (lx, ly), (lx + lw, ly + lh), col, -1)
        cv2.putText(
            out, text, (lx + pad, ly + lh - pad - 1), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return out


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


def _rov_status_png(counts):
    """Horizontal bar chart of ROV workflow states for PDF embedding."""
    labels = ['Flagged', 'Scheduled', 'In progress', 'Closed']
    keys = ['flagged', 'scheduled', 'in_progress', 'closed']
    sizes = [int(counts.get(k, 0) or 0) for k in keys]
    cols = ['#b3261e', '#0e8fa3', '#b5791f', '#15803d']
    if not any(sizes):
        labels, sizes, cols = ['Awaiting triage'], [1], ['#c5d0d6']
    fig, ax = plt.subplots(figsize=(5.4, 2.6), dpi=150)
    y = range(len(labels))
    ax.barh(list(y), sizes, color=cols[: len(labels)], height=0.55)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9, color='#0b2a38')
    ax.set_xlabel('Contacts', fontsize=8.5, color='#5d7480')
    ax.tick_params(axis='x', labelsize=8, colors='#5d7480')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('none')
    fig.patch.set_alpha(0)
    plt.tight_layout()
    bio = io.BytesIO()
    fig.savefig(bio, format='png', transparent=True, bbox_inches='tight')
    plt.close(fig)
    bio.seek(0)
    return bio


def _resolve_rov_queue(meta, all_dets):
    """ROV deployment queue from client workflow export or high-risk detections."""
    rq = meta.get('rov_queue')
    if isinstance(rq, dict) and rq.get('queue'):
        return list(rq['queue'])
    items = []
    for i, d in enumerate(all_dets):
        rk = risk_of(d['cls'])
        if rk not in ('Critical', 'High'):
            continue
        items.append(
            {
                'contact_id': d.get('id') or ('OBJ-%03d' % (i + 1)),
                'class': d['cls'],
                'confidence': d['conf'],
                'survey_risk': rk,
                'frame': d.get('_frame', 1),
                'workflow_state': 'recommended',
                'position': {'range_m': d.get('range')},
            }
        )
    return items


def _rov_followup_story(queue, status_summary, PW, BODY, SMALL, CELL, CELLB, CELLH, CELLHR, CELLR, WHT, LINE, ZEB):
    """ReportLab flowables for the ROV follow-up section."""
    story = []
    nq = len(queue)
    flagged = sum(1 for q in queue if (q.get('workflow_state') or '') in ('flagged', 'recommended'))
    story.append(
        Paragraph(
            '%d contact(s) are queued or recommended for ROV verification. '
            'The chart summarises workflow state counts from the survey-ops console '
            '(or inferred high-risk contacts when no queue was supplied).' % nq,
            BODY,
        )
    )
    story.append(Spacer(1, 8))
    chart_w = min(PW * 0.46, 88 * mm)
    summary_rows = [
        ['Flagged / recommended', str(status_summary.get('flagged', flagged))],
        ['Scheduled', str(status_summary.get('scheduled', 0))],
        ['In progress', str(status_summary.get('in_progress', 0))],
        ['Closed', str(status_summary.get('closed', 0))],
    ]
    sum_tbl = Table(
        [[Paragraph(k, CELL), Paragraph('<b>%s</b>' % v, CELLB)] for k, v in summary_rows],
        colWidths=[42 * mm, 22 * mm],
    )
    sum_tbl.setStyle(
        TableStyle(
            [
                ('LINEBELOW', (0, 0), (-1, -1), 0.4, LINE),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]
        )
    )
    top = Table(
        [[sum_tbl, Image(_rov_status_png(status_summary), width=chart_w, height=chart_w * 0.48)]],
        colWidths=[66 * mm, PW - 66 * mm],
    )
    top.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    story.append(top)
    story.append(Spacer(1, 10))
    if not queue:
        story.append(Paragraph('No ROV follow-up contacts for this survey.', BODY))
        return story
    hd = [
        Paragraph('ID', CELLH),
        Paragraph('Class', CELLH),
        Paragraph('Risk', CELLH),
        Paragraph('Frame', CELLHR),
        Paragraph('Conf', CELLHR),
        Paragraph('State', CELLH),
    ]
    data = [hd]
    sty = [
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('GRID', (0, 0), (-1, -1), 0.4, LINE),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('ALIGN', (3, 0), (4, -1), 'RIGHT'),
    ]
    show = queue[:14]
    for ri, q in enumerate(show, 1):
        rk = q.get('survey_risk') or risk_of(q.get('class', 'object'))
        st = (q.get('workflow_state') or 'flagged').replace('_', ' ').title()
        data.append(
            [
                Paragraph(str(q.get('contact_id', '—')), CELLB),
                Paragraph(str(q.get('class', '')).title(), CELL),
                Paragraph('<font color="white"><b>%s</b></font>' % rk, WHT),
                Paragraph(str(q.get('frame', '—')), CELLR),
                Paragraph('%d%%' % round(float(q.get('confidence', 0)) * 100), CELLR),
                Paragraph(st, CELL),
            ]
        )
        sty.append(('BACKGROUND', (2, ri), (2, ri), RISKCOL.get(rk, RISKCOL['Medium'])))
        if ri % 2 == 0:
            for cc in (0, 1, 3, 4, 5):
                sty.append(('BACKGROUND', (cc, ri), (cc, ri), ZEB))
    qt = Table(data, colWidths=[28 * mm, 26 * mm, 22 * mm, 16 * mm, 16 * mm, PW - 108 * mm])
    qt.setStyle(TableStyle(sty))
    story.append(qt)
    if len(queue) > len(show):
        story.append(Spacer(1, 4))
        story.append(Paragraph('… and %d more contact(s) in the full deployment queue export.' % (len(queue) - len(show)), SMALL))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            'Export the full queue from ROV Ops as <b>rov_deployment_queue.json</b> for dive planning systems.',
            SMALL,
        )
    )
    return story


FRAME_TWIN_W = 120.0
FRAME_TWIN_H = 60.0
HEAT_STOPS = np.array(
    [
        [0.12, 0.23, 0.54],
        [0.12, 0.61, 0.84],
        [0.25, 0.68, 0.41],
        [0.90, 0.82, 0.23],
        [0.88, 0.34, 0.31],
    ]
)


def _hex_to_rgb01(hexv):
    hexv = hexv.lstrip("#")
    return (
        int(hexv[0:2], 16) / 255.0,
        int(hexv[2:4], 16) / 255.0,
        int(hexv[4:6], 16) / 255.0,
    )


def _terrain_y(x, z):
    return (
        6.0 * np.sin(x * 0.05) * np.cos(z * 0.045)
        + 3.0 * np.sin(x * 0.11 + 1.2) * np.cos(z * 0.09 + 0.4)
        + 1.4 * np.sin((x + z) * 0.18)
        - 0.018 * x
    )


def _det_twin_points(dets, img_w, img_h):
    pts = []
    for d in dets:
        b = d["box"]
        cxn = ((b[0] + b[2]) / 2) / max(1, img_w)
        cyn = ((b[1] + b[3]) / 2) / max(1, img_h)
        pts.append(
            {
                "cls": d["cls"],
                "conf": float(d.get("conf", 0)),
                "px": cxn * FRAME_TWIN_W - FRAME_TWIN_W / 2,
                "pz": cyn * FRAME_TWIN_H - FRAME_TWIN_H / 2,
            }
        )
    return pts


def _heat_rgba_grid(norm2d):
    t = np.clip(norm2d, 0, 1)
    f = t * (len(HEAT_STOPS) - 1)
    i = np.floor(f).astype(int)
    i = np.clip(i, 0, len(HEAT_STOPS) - 2)
    k = (f - i)[..., np.newaxis]
    a = HEAT_STOPS[i]
    b = HEAT_STOPS[i + 1]
    rgb = a + (b - a) * k
    rgba = np.ones((*rgb.shape[:-1], 4))
    rgba[..., :3] = rgb
    return rgba


def _heatmap_3d_png(dets, img_w, img_h, title="3D detection density"):
    """Render seabed twin heatmap — high-res, bright, crisp 2D labels."""
    pad = 40.0
    wt = FRAME_TWIN_W + pad * 2
    ht = FRAME_TWIN_H + pad * 2
    max_dim = max(wt, ht)
    nx = max(96, int(round(wt / 2)))
    nz = max(68, int(round(ht / 2)))
    x = np.linspace(-wt / 2, wt / 2, nx)
    z = np.linspace(-ht / 2, ht / 2, nz)
    X, Z = np.meshgrid(x, z)
    Y = _terrain_y(X, Z)
    min_h = float(Y.min())
    max_h = float(Y.max())

    pts = _det_twin_points(dets, img_w, img_h)
    sig = max(8.0, max_dim * 0.035)
    inv2s2 = 1.0 / (2.0 * sig * sig)
    heat = np.zeros_like(X, dtype=np.float64)
    for p in pts:
        dx = X - p["px"]
        dz = Z - p["pz"]
        heat += p["conf"] * np.exp(-(dx * dx + dz * dz) * inv2s2)
    heat_max = max(float(heat.max()), 1e-6)
    heat_norm = heat / heat_max
    heat_rgb = _heat_rgba_grid(heat_norm)[..., :3]

    ls = LightSource(azdeg=225, altdeg=44)
    dxs = float(x[1] - x[0]) if len(x) > 1 else 1.0
    dzs = float(z[1] - z[0]) if len(z) > 1 else 1.0
    hill = ls.hillshade(Y, vert_exag=1.35, dx=dxs, dy=dzs)
    lit_rgb = np.clip(heat_rgb * (0.68 + 0.52 * hill[..., np.newaxis]), 0, 1)
    facecolors = np.dstack([lit_rgb, np.ones_like(hill)])

    fig = plt.figure(figsize=(10.5, 7.0), dpi=220, facecolor="#04101a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#04101a")

    ax.plot_surface(
        X, Z, Y,
        facecolors=facecolors,
        rstride=1, cstride=1,
        shade=False,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        zorder=1,
    )

    y_water = max_h + 20.0
    xw = np.linspace(-wt * 0.7, wt * 0.7, 20)
    zw = np.linspace(-ht * 0.7, ht * 0.7, 14)
    Xw, Zw = np.meshgrid(xw, zw)
    Yw = np.full_like(Xw, y_water)
    ax.plot_surface(
        Xw, Zw, Yw,
        color="#0c4a63",
        alpha=0.16,
        shade=False,
        linewidth=0,
        edgecolor="none",
        zorder=2,
    )

    marker_meta = []
    for p in pts:
        ty = float(_terrain_y(np.array([p["px"]]), np.array([p["pz"]]))[0])
        top = ty + 9.0
        hexcol = class_hex(p["cls"])
        rgb = _hex_to_rgb01(hexcol)
        ax.plot(
            [p["px"], p["px"]],
            [p["pz"], p["pz"]],
            [ty, top],
            color=rgb,
            alpha=0.85,
            linewidth=2.8,
            zorder=4,
            solid_capstyle="round",
        )
        ax.scatter(
            [p["px"]], [p["pz"]], [top],
            color=rgb,
            s=160,
            depthshade=False,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )
        marker_meta.append((p, top, hexcol))

    ax.set_xlim(-wt / 2, wt / 2)
    ax.set_ylim(-ht / 2, ht / 2)
    ax.set_zlim(min_h - 6, y_water + 10)
    try:
        ax.set_box_aspect((wt, ht, (max_h - min_h) + 34))
    except Exception:
        pass
    ax.set_axis_off()
    ax.view_init(elev=36, azim=-46)
    ax.dist = 9.2

    ax.text2D(0.02, 0.97, title, transform=ax.transAxes, color="#eaf3f5", fontsize=12, fontweight="bold")
    ax.text2D(0.02, 0.02, "Low", transform=ax.transAxes, color="#b8c8d0", fontsize=9, fontweight="bold")
    ax.text2D(0.20, 0.02, "High", transform=ax.transAxes, color="#b8c8d0", fontsize=9, fontweight="bold")
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax_inset = fig.add_axes([0.26, 0.035, 0.34, 0.034])
    ax_inset.imshow(_heat_rgba_grid(grad), aspect="auto")
    ax_inset.set_axis_off()
    ax_inset.set_title("Detection density", color="#b8c8d0", fontsize=9, fontweight="bold", pad=3)

    bio = io.BytesIO()
    fig.savefig(bio, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.18, dpi=220)
    plt.close(fig)
    bio.seek(0)
    return bio


def _heatmap_det_table(dets, pw):
    if not dets:
        return []
    ss = getSampleStyleSheet()
    cell = ParagraphStyle("HMT", parent=ss["Normal"], fontSize=9.5, leading=13)
    cellb = ParagraphStyle("HMTB", parent=cell, fontName="Helvetica-Bold", fontSize=9.5)
    rows = [
        [
            Paragraph("Class", cellb),
            Paragraph("Confidence", cellb),
            Paragraph("Risk", cellb),
        ]
    ]
    sty = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    for ri, d in enumerate(sorted(dets, key=lambda x: -x["conf"]), 1):
        rk = risk_of(d["cls"])
        hexcol = class_hex(d["cls"])
        rows.append([
            Paragraph('<font color="%s"><b>%s</b></font>' % (hexcol, _cap_cls(d["cls"])), cell),
            Paragraph("<b>%d%%</b>" % round(float(d["conf"]) * 100), cell),
            Paragraph(rk, cell),
        ])
        if ri % 2 == 0:
            sty.append(("BACKGROUND", (0, ri), (-1, ri), ZEB))
    tbl = Table(rows, colWidths=[pw * 0.38, pw * 0.28, pw * 0.24])
    tbl.setStyle(TableStyle(sty))
    return [Spacer(1, 6), Paragraph("<b>Defects in this frame</b>", cellb), Spacer(1, 4), tbl]


def _heatmap_story(dets, img_w, img_h, pw, title="3D detection density", include_table=False):
    hm_h = pw * 0.68
    cap = (
        "Heat colour = detection density. Coloured pins mark each defect — class names are in the detection table."
        if dets
        else "Baseline seabed twin surface — no defects detected in this frame."
    )
    ss = getSampleStyleSheet()
    hm_body = ParagraphStyle(
        "HM_BODY", parent=ss["Normal"], fontSize=10, leading=15,
        textColor=colors.HexColor("#1c2b33"),
    )
    hm_small = ParagraphStyle(
        "HM_SM", parent=ss["Normal"], fontSize=8.5, leading=12, textColor=MUT,
    )
    story = [
        Spacer(1, 6),
        Paragraph("<b>3D seabed digital twin — detection heatmap</b>", hm_body),
        Spacer(1, 4),
        Image(_heatmap_3d_png(dets, img_w, img_h, title=title), width=pw, height=hm_h),
        Spacer(1, 4),
        Paragraph(cap, hm_small),
    ]
    if include_table:
        story += _heatmap_det_table(dets, pw)
    return story


TWIN_VIEW_W = 120.0
TWIN_VIEW_H = 60.0


def _scaled_dets_for_frame(fr):
    raw = fr["image_bytes"]
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return [], 0, 0
    imgh, imgw = img.shape[:2]
    nat_w = fr.get("nat_w") or fr.get("natW")
    nat_h = fr.get("nat_h") or fr.get("natH")
    scaled = _det_boxes_for_image(fr.get("detections") or [], imgw, imgh, nat_w, nat_h)
    return scaled, imgw, imgh


def _twin_points_from_scaled(scaled, imgw, imgh, frame_idx=1):
    pts = []
    for d in scaled:
        if d.get("cxn") is not None and d.get("cyn") is not None:
            cxn, cyn = float(d["cxn"]), float(d["cyn"])
        else:
            b = d["box"]
            cxn = ((b[0] + b[2]) / 2) / max(1, imgw)
            cyn = ((b[1] + b[3]) / 2) / max(1, imgh)
        pts.append(
            {
                "cls": d["cls"],
                "conf": float(d.get("conf", 0)),
                "x": cxn * TWIN_VIEW_W,
                "y": cyn * TWIN_VIEW_H,
                "frame": frame_idx,
            }
        )
    return pts


def _twin_frame_from_survey(fr, fi):
    scaled, imgw, imgh = _scaled_dets_for_frame(fr)
    return {
        "index": fi,
        "name": fr.get("filename", "Frame %d" % fi),
        "points": _twin_points_from_scaled(scaled, imgw, imgh, fi),
    }


def _twin_screenshot_png(frame_list, base_url=None, viewport=(1440, 900), frame_idx=None):
    """Capture the live /twin page (markers + labels) via headless Chromium."""
    if not frame_list:
        return None
    base_url = (base_url or os.environ.get("NAUTICAI_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    # Per-frame PDF capture sends one frame; twin.html indexes frameList[0] when ?frame=1.
    # Renumber to index 1 so markers are not filtered out (points carried frame=4 etc.).
    capture_list = frame_list
    twin_frame_param = 1
    if len(frame_list) == 1:
        fl = frame_list[0]
        pts = []
        for p in fl.get("points") or []:
            pt = dict(p)
            pt["frame"] = 1
            pts.append(pt)
        capture_list = [{
            "index": 1,
            "name": fl.get("name", "Frame 1"),
            "points": pts,
        }]
        twin_frame_param = 1
    else:
        twin_frame_param = int(frame_idx or frame_list[0].get("index", 1) or 1)

    data = {"frames": len(capture_list), "frameList": capture_list, "points": []}
    for fl in capture_list:
        data["points"].extend(fl.get("points") or [])

    payload = base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode("utf-8")).decode().rstrip("=")
    url = "%s/twin?frame=%d&embed=1&screenshot=1#twin=%s" % (base_url, twin_frame_param, payload)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_selector('[data-twin-ready="1"]', timeout=35000)
            page.wait_for_timeout(1200)
            shot = page.screenshot(type="png", full_page=False)
            browser.close()
            bio = io.BytesIO(shot)
            bio.seek(0)
            return bio
    except Exception as exc:
        _log.warning("3D twin screenshot failed (%s): %s", base_url, exc)
        return None


def _twin_story_for_frame(fr, fi, nf, pw, meta=None):
    """PDF flowables: real /twin screenshot, or a clear placeholder if Chromium unavailable."""
    meta = meta or {}
    twin_frame = _twin_frame_from_survey(fr, fi)
    png = _twin_screenshot_png([twin_frame], base_url=meta.get("base_url"))
    ss = getSampleStyleSheet()
    body = ParagraphStyle(
        "TWIN_BODY", parent=ss["Normal"], fontSize=10, leading=15,
        textColor=colors.HexColor("#1c2b33"),
    )
    small = ParagraphStyle(
        "TWIN_SM", parent=ss["Normal"], fontSize=8.5, leading=12, textColor=MUT,
    )
    title = "Frame %d of %d \u2014 3D seabed twin" % (fi, nf)
    story = [
        Spacer(1, 6),
        Paragraph("<b>3D seabed digital twin</b>", body),
        Spacer(1, 4),
    ]
    if png:
        story += [
            Image(png, width=pw, height=pw * 0.62),
            Spacer(1, 4),
            Paragraph(
                "Live twin view for <b>%s</b> \u2014 heatmap, markers and class labels match the "
                "interactive viewer at /twin." % twin_frame["name"],
                small,
            ),
        ]
    else:
        story += [
            Paragraph(
                "<b>%s</b> \u2014 twin screenshot unavailable. Install once: "
                "<font face='Courier'>pip install playwright</font> then "
                "<font face='Courier'>playwright install chromium</font>, "
                "restart the server, and regenerate. "
                "Or open the live twin in your browser: "
                "<font color='#0e8fa3'>/twin?frame=%d</font>." % (title, fi),
                body,
            ),
            Spacer(1, 6),
            Paragraph(
                "%d detection(s) mapped on this frame." % len(twin_frame.get("points") or []),
                small,
            ),
        ]
    return story


def build_report(image_bytes, detections, meta=None):
    meta = meta or {}
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    IMGH, IMGW = img.shape[:2]
    GW = float(meta.get('frame_m', 120.0)); GH = GW * IMGH / IMGW

    def sizem(b):
        return ((b[2] - b[0]) / IMGW * GW, (b[3] - b[1]) / IMGH * GH)

    brand = meta.get("brand", BRAND)
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
    story = [Spacer(1, 60 * mm)]
    cover = [['Survey ID', meta.get('survey_id', 'SURV-2026-001')],
             ['Project', meta.get('project', 'Offshore Wind Site Assessment')],
             ['Location', meta.get('location', 'North Sea Sector A')],
             ['Vessel', meta.get('vessel', '\u2014')],
             ['Sensor', meta.get('sensor', 'Side-scan sonar (SSS)')],
             ['Source image', meta.get('filename', 'sonar_frame')],
             ['Generated', datetime.now().strftime('%d %B %Y, %H:%M')]]
    story += [hdr_table(cover, 46 * mm, True), Spacer(1, 11),
              _fit_image_flowable(ann, PW, PAGEH - 75 * mm), Spacer(1, 5),
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
    nat_w = meta.get("nat_w") or meta.get("natW")
    nat_h = meta.get("nat_h") or meta.get("natH")
    scaled = _det_boxes_for_image(dets, IMGW, IMGH, nat_w, nat_h)
    story += section('03', 'Sonar Image Analysis')
    story += [_fit_image_flowable(_annotated_png(img, dets, nat_w, nat_h), PW, PAGEH - 50 * mm), Spacer(1, 5),
              Paragraph('Detected objects with bounding boxes and class confidence. '
                        'Box colours correspond to object class.', SMALL)]
    if n:
        story += _heatmap_det_table(scaled, PW)
    story += [PageBreak()]
    story += section('03.1', '3D Seabed Twin')
    twin_fr = {
        "image_bytes": image_bytes,
        "detections": dets,
        "filename": meta.get("filename", "sonar_frame"),
        "nat_w": nat_w,
        "nat_h": nat_h,
    }
    story += _twin_story_for_frame(twin_fr, 1, 1, PW, meta)
    story += [PageBreak()]

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
    ap = [['Brand', brand],
          ['Model', meta.get('model', 'YOLOv8 sonar detection model')],
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

    NumCanvas = _pdf_page_furniture(
        brand, "Automated side-scan sonar analysis  \u00b7  Proof of concept"
    )

    out = io.BytesIO()
    frame = Frame(LM, 18 * mm, PW, PAGEH - 18 * mm - 26 * mm, id='f')
    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=LM, rightMargin=RM,
                          topMargin=26 * mm, bottomMargin=18 * mm,
                          title='Seabed Object Detection Report - %s' % brand)
    doc.addPageTemplates([PageTemplate(id='main', frames=[frame])])
    doc.build(story, canvasmaker=NumCanvas)
    out.seek(0)
    return out.getvalue()


def build_survey_report(frames_data, meta=None):
    """Build one PDF covering every analysed frame.

    frames_data: list of dicts with keys image_bytes, detections, filename
    """
    meta = meta or {}
    if not frames_data:
        raise ValueError("no frames")
    if len(frames_data) == 1:
        fm = dict(meta)
        fm["filename"] = frames_data[0].get("filename", meta.get("filename", "sonar_frame"))
        return build_report(frames_data[0]["image_bytes"], frames_data[0]["detections"], fm)

    brand = meta.get("brand", BRAND)
    nf = len(frames_data)
    all_dets = []
    for fi, fr in enumerate(frames_data, 1):
        for d in fr.get("detections") or []:
            all_dets.append(dict(d, _frame=fi, _frame_name=fr.get("filename", "frame_%d" % fi)))

    n = len(all_dets)
    avg = (sum(d["conf"] for d in all_dets) / n) if n else 0.0
    counts = {}
    for d in all_dets:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    risks = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for d in all_dets:
        risks[risk_of(d["cls"])] += 1
    hi = risks["Critical"] + risks["High"]

    PAGEW, PAGEH = A4
    LM = RM = 16 * mm
    PW = PAGEW - LM - RM
    ss = getSampleStyleSheet()

    def stl(nm, parent=None, **k):
        return ParagraphStyle(nm, parent=parent or ss["Normal"], **k)

    H1 = stl("H1", fontSize=14.5, textColor=NAVY, leading=17, fontName="Helvetica-Bold")
    SECN = stl("SECN", fontSize=9, textColor=CYAND, fontName="Helvetica-Bold")
    BODY = stl("BODY", fontSize=10, leading=15, textColor=colors.HexColor("#1c2b33"))
    SMALL = stl("SM", fontSize=8, leading=11, textColor=MUT)
    CELL = stl("CELL", fontSize=9.2, leading=12, textColor=colors.HexColor("#1c2b33"))
    CELLB = stl("CELLB", fontSize=9.2, leading=12, textColor=NAVY, fontName="Helvetica-Bold")
    CELLR = stl("CELLR", parent=CELL, alignment=TA_RIGHT)
    CELLH = stl("CELLH", fontSize=8.6, leading=11, textColor=colors.white, fontName="Helvetica-Bold")
    CELLHR = stl("CELLHR", parent=CELLH, alignment=TA_RIGHT)
    WHT = stl("WHT", fontSize=9.2, leading=12, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER)

    class Rule(Flowable):
        def __init__(s, w, c=CYAN, t=1.4):
            s.w = w
            s.c = c
            s.t = t

        def wrap(s, a, b):
            return (s.w, s.t)

        def draw(s):
            s.canv.setStrokeColor(s.c)
            s.canv.setLineWidth(s.t)
            s.canv.line(0, 0, s.w, 0)

    def section(num, title):
        return [Spacer(1, 2), Paragraph("SECTION %s" % num, SECN), Paragraph(title, H1), Rule(PW), Spacer(1, 9)]

    def hdr_table(rows, col0=46 * mm, zebra=False):
        data = [[Paragraph(k, CELL), Paragraph("<b>%s</b>" % v, CELLB)] for k, v in rows]
        t = Table(data, colWidths=[col0, PW - col0])
        sty = [
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 5.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        if zebra:
            for r in range(0, len(data), 2):
                sty.append(("BACKGROUND", (0, r), (-1, r), ZEB))
        t.setStyle(TableStyle(sty))
        return t

    def frame_detection_table(img, dets, frame_label):
        if not dets:
            return [Paragraph("No objects detected in %s." % frame_label, BODY)]
        IMGH, IMGW = img.shape[:2]
        GW = float(meta.get("frame_m", 120.0))
        GH = GW * IMGH / IMGW

        def sizem(b):
            return ((b[2] - b[0]) / IMGW * GW, (b[3] - b[1]) / IMGH * GH)

        hd = [
            Paragraph("ID", CELLH),
            Paragraph("Class", CELLH),
            Paragraph("Conf", CELLHR),
            Paragraph("Risk", WHT),
            Paragraph("Location (px)", CELLH),
            Paragraph("Est. size", CELLHR),
        ]
        data = [hd]
        sty = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 5.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (2, 0), (2, -1), "RIGHT"),
            ("ALIGN", (3, 0), (3, -1), "CENTER"),
            ("ALIGN", (5, 0), (5, -1), "RIGHT"),
        ]
        order = sorted(range(len(dets)), key=lambda i: -dets[i]["conf"])
        for row, i in enumerate(order, 1):
            d = dets[i]
            b = d["box"]
            rk = risk_of(d["cls"])
            sx, sy = sizem(b)
            cxp = (b[0] + b[2]) // 2
            cyp = (b[1] + b[3]) // 2
            data.append(
                [
                    Paragraph("OBJ-%03d" % row, CELLB),
                    Paragraph(str(d["cls"]).title(), CELL),
                    Paragraph("%d%%" % round(d["conf"] * 100), CELLR),
                    Paragraph('<font color="white"><b>%s</b></font>' % rk, WHT),
                    Paragraph("X:%d  Y:%d" % (cxp, cyp), CELL),
                    Paragraph("%.1f \u00d7 %.1f m" % (sx, sy), CELLR),
                ]
            )
            sty.append(("BACKGROUND", (3, row), (3, row), RISKCOL[rk]))
            if row % 2 == 0:
                for cc in (0, 1, 2, 4, 5):
                    sty.append(("BACKGROUND", (cc, row), (cc, row), ZEB))
        dt = Table(
            data,
            colWidths=[20 * mm, 26 * mm, 16 * mm, 22 * mm, 32 * mm, PW - 116 * mm],
            repeatRows=1,
        )
        dt.setStyle(TableStyle(sty))
        return [dt]

    story = [Spacer(1, 60 * mm)]
    cover = [
        ["Survey ID", meta.get("survey_id", "SURV-2026-001")],
        ["Project", meta.get("project", "Offshore Wind Site Assessment")],
        ["Location", meta.get("location", "North Sea Sector A")],
        ["Vessel", meta.get("vessel", "\u2014")],
        ["Sensor", meta.get("sensor", "Side-scan sonar (SSS)")],
        ["Frames analysed", str(nf)],
        ["Total detections", str(n)],
        ["Generated", datetime.now().strftime("%d %B %Y, %H:%M")],
    ]
    story += [hdr_table(cover, 46 * mm, True), Spacer(1, 11), PageBreak()]

    story += section("01", "Executive Summary")
    es = [
        ["Survey frames", str(nf)],
        ["Total objects detected", str(n)],
        ["High-risk (Critical / High)", str(hi)],
        ["Medium / Low risk", "%d / %d" % (risks["Medium"], risks["Low"])],
        ["Average confidence", "%.1f%%" % (avg * 100)],
    ]
    cls_summary = ", ".join("%s (%d)" % (c.title(), counts[c]) for c in sorted(counts)) or "no objects"
    story += [
        hdr_table(es, 70 * mm, True),
        Spacer(1, 9),
        Paragraph(
            "The system analysed %d side-scan frame(s) and detected %d object(s) in total: %s. "
            "%d were assessed as high-risk." % (nf, n, cls_summary, hi),
            BODY,
        ),
        Spacer(1, 10),
    ]

    frows = [[Paragraph("Frame", CELLH), Paragraph("Source image", CELLH), Paragraph("Detections", CELLHR)]]
    fst = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (2, 0), (2, -1), "RIGHT"),
    ]
    for ri, fr in enumerate(frames_data, 1):
        nd = len(fr.get("detections") or [])
        frows.append(
            [
                Paragraph("%d / %d" % (ri, nf), CELLB),
                Paragraph(fr.get("filename", "frame_%d" % ri), CELL),
                Paragraph(str(nd), CELLR),
            ]
        )
        if ri % 2 == 0:
            fst.append(("BACKGROUND", (0, ri), (-1, ri), ZEB))
    ft = Table(frows, colWidths=[22 * mm, PW - 42 * mm, 20 * mm])
    ft.setStyle(TableStyle(fst))
    story += [Paragraph("<b>Frames in this survey</b>", CELLB), Spacer(1, 4), ft, PageBreak()]

    story += section("02", "Detection Overview (all frames)")
    if n:
        crows = [[Paragraph("Class", CELLH), Paragraph("Count", CELLHR), Paragraph("Risk", CELLH)]]
        csty = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ]
        ri = 1
        for c in sorted(counts):
            crows.append([Paragraph(c.title(), CELL), Paragraph(str(counts[c]), CELLR), Paragraph(risk_of(c), CELL)])
            if ri % 2 == 0:
                csty.append(("BACKGROUND", (0, ri), (-1, ri), ZEB))
            ri += 1
        crows.append([Paragraph("Total", CELLB), Paragraph("<b>%d</b>" % n, CELLR), Paragraph("", CELL)])
        csty.append(("LINEABOVE", (0, len(crows) - 1), (-1, len(crows) - 1), 0.8, NAVY))
        ctbl = Table(crows, colWidths=[34 * mm, 20 * mm, 28 * mm])
        ctbl.setStyle(TableStyle(csty))
        two = Table([[ctbl, Image(_pie_png(counts), width=74 * mm, height=74 * mm)]], colWidths=[84 * mm, PW - 84 * mm])
        two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (1, 0), (1, 0), "CENTER")]))
        story += [two, PageBreak()]
    else:
        story += [Paragraph("No objects were detected across the survey frames.", BODY), PageBreak()]

    for fi, fr in enumerate(frames_data, 1):
        raw = fr["image_bytes"]
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        dets = list(fr.get("detections") or [])
        IMGH, IMGW = img.shape[:2]
        nat_w = fr.get("nat_w") or fr.get("natW")
        nat_h = fr.get("nat_h") or fr.get("natH")
        scaled = _det_boxes_for_image(dets, IMGW, IMGH, nat_w, nat_h)
        fname = fr.get("filename", "frame_%d" % fi)
        story += section("03.%d" % fi, "Frame %d of %d \u2014 %s" % (fi, nf, fname))
        frame_img_h = PAGEH - 18 * mm - 26 * mm - 45 * mm
        story += [
            _fit_image_flowable(_annotated_png(img, dets, nat_w, nat_h), PW, frame_img_h),
            Spacer(1, 5),
            Paragraph(
                "%d detection(s) in this frame. Positions and sizes are illustrative." % len(scaled),
                SMALL,
            ),
        ]
        story += frame_detection_table(img, scaled, "frame %d" % fi)
        story += [PageBreak()]
        story += section(
            "03.%d twin" % fi,
            "Frame %d of %d \u2014 3D seabed twin" % (fi, nf),
        )
        story += _twin_story_for_frame(fr, fi, nf, PW, meta)
        story += [PageBreak()]

    story += section("04", "Risk Assessment")
    txts = {
        "Critical": "Critical contacts \u2014 treat as hard constraints pending inspection.",
        "High": "High-risk linear/obstruction features \u2014 maintain clearance and verify before route works.",
        "Medium": "Medium-risk targets \u2014 may require clearance.",
        "Low": "Low-risk isolated targets \u2014 note for foundation suitability.",
    }
    present = [lvl for lvl in ["Critical", "High", "Medium", "Low"] if risks[lvl] > 0]
    if present:
        rrows = []
        for lvl in present:
            items = [d["cls"] for d in all_dets if risk_of(d["cls"]) == lvl]
            cnt = ", ".join("%s \u00d7%d" % (t.title(), items.count(t)) for t in sorted(set(items)))
            rrows.append(
                [
                    Paragraph('<font color="white"><b>%s</b></font>' % lvl, WHT),
                    Paragraph(txts[lvl], CELL),
                    Paragraph(cnt, SMALL),
                ]
            )
        rsty = [
            ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ]
        for r, lvl in enumerate(present):
            rsty.append(("BACKGROUND", (0, r), (0, r), RISKCOL[lvl]))
        rt = Table(rrows, colWidths=[24 * mm, PW - 74 * mm, 50 * mm])
        rt.setStyle(TableStyle(rsty))
        story += [rt]
    else:
        story += [Paragraph("No risk-rated contacts in this survey.", BODY)]
    story += [Spacer(1, 12)]

    rov_queue = _resolve_rov_queue(meta, all_dets)
    rov_summary = {}
    rq_meta = meta.get('rov_queue')
    if isinstance(rq_meta, dict) and rq_meta.get('summary'):
        rov_summary = rq_meta['summary']
    else:
        rov_summary = {
            'flagged': sum(
                1
                for q in rov_queue
                if (q.get('workflow_state') or '') in ('flagged', 'recommended')
            ),
            'scheduled': 0,
            'in_progress': 0,
            'closed': 0,
        }
    story += section("05", "ROV Follow-up Plan")
    story += _rov_followup_story(
        rov_queue, rov_summary, PW, BODY, SMALL, CELL, CELLB, CELLH, CELLHR, CELLR, WHT, LINE, ZEB
    )
    story += [PageBreak()]

    story += section("06", "Recommendations")
    recs = []
    if counts.get("uxo") or counts.get("mine"):
        n_uxo = counts.get("uxo", 0) + counts.get("mine", 0)
        recs.append(
            "%d suspected UXO contact(s) \u2014 stop work in the vicinity and arrange a specialist assessment." % n_uxo
        )
    if counts.get("wreck"):
        recs.append(
            "%d wreck contact(s) \u2014 conduct ROV inspection before any installation activity." % counts["wreck"]
        )
    pl = counts.get("pipeline", 0) + counts.get("cable", 0) + counts.get("cable_crossing", 0)
    if pl:
        recs.append(
            "%d pipeline / cable contact(s) \u2014 verify crossings manually and confirm separation distances." % pl
        )
    if counts.get("boulder"):
        recs.append(
            "%d boulder contact(s) \u2014 avoid foundation placement nearby; consider clearance." % counts["boulder"]
        )
    if counts.get("debris"):
        recs.append(
            "%d debris contact(s) \u2014 remove or route around prior to cable laying." % counts["debris"]
        )
    if counts.get("rock"):
        recs.append(
            "%d rock contact(s) \u2014 assess foundation suitability for affected areas." % counts["rock"]
        )
    if counts.get("scour"):
        recs.append(
            "%d scour contact(s) \u2014 monitor for pipeline exposure or free-span risk." % counts["scour"]
        )
    if counts.get("obstruction"):
        recs.append(
            "%d obstruction contact(s) \u2014 confirm clearance before route works." % counts["obstruction"]
        )
    if not recs:
        recs.append("No high-priority targets detected; continue routine survey coverage.")
    if hi:
        recs.append(
            "Perform an additional survey pass around the %d high-risk detection(s) (Critical / High)." % hi
        )
    rec_rows = [
        [Paragraph('<font color="%s"><b>%d</b></font>' % ("#" + CYAND.hexval()[2:], i), CELLB), Paragraph(r, BODY)]
        for i, r in enumerate(recs, 1)
    ]
    rect = Table(rec_rows, colWidths=[10 * mm, PW - 10 * mm])
    rect.setStyle(
        TableStyle(
            [
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.3, LINE),
            ]
        )
    )
    story += [rect]

    NumCanvas = _pdf_page_furniture(
        brand, "Multi-frame survey  \u00b7  %d frame(s)" % nf
    )

    out = io.BytesIO()
    frame = Frame(LM, 18 * mm, PW, PAGEH - 18 * mm - 26 * mm, id="f")
    doc = BaseDocTemplate(
        out,
        pagesize=A4,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=26 * mm,
        bottomMargin=18 * mm,
        title="Seabed Object Detection Report - %s" % brand,
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    doc.build(story, canvasmaker=NumCanvas)
    out.seek(0)
    return out.getvalue()

def build_defects_executive_report(payload):
    """Azolla-style seabed defects executive summary PDF from defects_report.html payload."""
    payload = payload or {}
    meta = payload.get("meta") or {}
    rows = payload.get("rows") or []
    repair_rows = payload.get("repairRows") or [r for r in rows if r.get("count")]
    repair_totals = payload.get("repairTotals") or {}
    brand = meta.get("brand") or BRAND

    PAGEW, PAGEH = A4
    LM = RM = 16 * mm
    PW = PAGEW - LM - RM

    styles = getSampleStyleSheet()
    BODY = ParagraphStyle(
        "defBody", parent=styles["Normal"], fontSize=8.2, leading=11, textColor=colors.HexColor("#1a2a33")
    )
    SMALL = ParagraphStyle(
        "defSmall", parent=BODY, fontSize=7.2, leading=9.5, textColor=MUT
    )
    CELL = ParagraphStyle("defCell", parent=BODY, fontSize=7.4, leading=9.5)
    CELLB = ParagraphStyle("defCellB", parent=CELL, fontName="Helvetica-Bold")
    HDR = ParagraphStyle(
        "defHdr", parent=CELL, fontSize=7.5, textColor=colors.white, fontName="Helvetica-Bold"
    )

    story = [
        Paragraph("<b>SEABED DEFECTS SERVICE REPORT</b>", ParagraphStyle(
            "t", parent=BODY, fontSize=14, textColor=NAVY, spaceAfter=4
        )),
        Paragraph(
            "Project: <b>%s</b> &nbsp;|&nbsp; Job: <b>%s</b> &nbsp;|&nbsp; Vessel: %s"
            % (meta.get("project", "—"), meta.get("survey_line", "—"), meta.get("vessel", "—")),
            BODY,
        ),
        Paragraph(
            "Location: %s &nbsp;|&nbsp; Sensor: %s &nbsp;|&nbsp; Date: %s &nbsp;|&nbsp; Frames: %s"
            % (
                meta.get("location", "—"),
                meta.get("sensor", "—"),
                meta.get("date", "—"),
                meta.get("frames", 0),
            ),
            SMALL,
        ),
        Spacer(1, 6 * mm),
        Paragraph("<b>SEABED CONTACT CONDITIONS — EXECUTIVE SUMMARY</b>", ParagraphStyle(
            "s", parent=BODY, fontSize=9.5, textColor=NAVY, spaceAfter=5
        )),
    ]

    head = [
        Paragraph("Contact class", HDR),
        Paragraph("Seabed conditions executive", HDR),
        Paragraph("% contacts", HDR),
        Paragraph("Sev.", HDR),
        Paragraph("Ver.", HDR),
        Paragraph("ROV", HDR),
        Paragraph("Remarks", HDR),
    ]
    data = [head]
    for r in rows:
        pct = ("%.1f%%" % r["pct"]) if r.get("count") else "—"
        data.append([
            Paragraph("<b>%s</b>" % _cap_cls(r.get("label", r.get("cls", ""))), CELLB),
            Paragraph(str(r.get("executive", "")), CELL),
            Paragraph(pct, CELL),
            Paragraph("<b>%s</b>" % r.get("severity", "D"), CELL),
            Paragraph("X" if r.get("verified") else "", CELL),
            Paragraph("X" if r.get("rovRequired") else "", CELL),
            Paragraph(str(r.get("remarks", "")), CELL),
        ])

    colw = [PW * 0.11, PW * 0.46, PW * 0.08, PW * 0.06, PW * 0.05, PW * 0.05, PW * 0.19]
    tbl = Table(data, colWidths=colw, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEB]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [
        tbl,
        Spacer(1, 4 * mm),
        Paragraph(
            "<b>Severity:</b> (A) Light &nbsp; (B) Moderate &nbsp; (C) Heavy &nbsp; (D) Clean / none detected",
            SMALL,
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "Desk-review summary from %s frame(s) and %s contact(s). Percentages are share of total contacts by class."
            % (meta.get("frames", 0), payload.get("total", 0)),
            SMALL,
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("<b>REMEDIATION PLAN</b>", ParagraphStyle(
            "rp", parent=BODY, fontSize=11, textColor=NAVY, spaceAfter=6
        )),
        Paragraph(
            "Repair time, cost, resources, and post-remediation lifespan (re-survey interval) by contact class.",
            SMALL,
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "<b>%s</b> contacts · <b>%s</b> total repair window · <b>%s</b> est. cost · <b>%s</b> active classes"
            % (
                repair_totals.get("contacts", 0),
                repair_totals.get("timeLabel", "—"),
                repair_totals.get("costLabel", "—"),
                repair_totals.get("classes", 0),
            ),
            BODY,
        ),
        Spacer(1, 4 * mm),
    ]

    rhead = [
        Paragraph("Class", HDR),
        Paragraph("Qty", HDR),
        Paragraph("Est. repair time", HDR),
        Paragraph("Est. cost", HDR),
        Paragraph("Resources", HDR),
        Paragraph("Lifespan / re-survey", HDR),
    ]
    rdata = [rhead]
    for r in repair_rows:
        rdata.append([
            Paragraph("<b>%s</b>" % _cap_cls(r.get("label", r.get("cls", ""))), CELLB),
            Paragraph(str(r.get("count", 0)), CELL),
            Paragraph(str(r.get("repairTime", "—")), CELL),
            Paragraph(str(r.get("repairCost", "—")), CELL),
            Paragraph(str(r.get("resources", "—")), CELL),
            Paragraph(str(r.get("lifespan", "—")), CELL),
        ])
    if repair_rows:
        rdata.append([
            Paragraph("<b>Total</b>", CELLB),
            Paragraph("<b>%s</b>" % repair_totals.get("contacts", 0), CELLB),
            Paragraph("<b>%s</b>" % repair_totals.get("timeLabel", "—"), CELLB),
            Paragraph("<b>%s</b>" % repair_totals.get("costLabel", "—"), CELLB),
            Paragraph("Aggregated across active classes", CELL),
            Paragraph("", CELL),
        ])
    else:
        rdata.append([
            Paragraph("No contacts requiring remediation", CELL),
            Paragraph("", CELL), Paragraph("", CELL),
            Paragraph("", CELL), Paragraph("", CELL), Paragraph("", CELL),
        ])

    rcolw = [PW * 0.12, PW * 0.06, PW * 0.16, PW * 0.10, PW * 0.28, PW * 0.28]
    rtbl = Table(rdata, colWidths=rcolw, repeatRows=1)
    rtbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEB]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story += [
        rtbl,
        Spacer(1, 4 * mm),
        Paragraph(
            "Planning estimates for desk-review. Actual remediation depends on site conditions, permits, and contractor rates.",
            SMALL,
        ),
    ]

    photos = payload.get("photos") or []
    if photos:
        story += [PageBreak(), Paragraph("<b>PHOTOGRAPHIC REPORT</b>", ParagraphStyle(
            "p", parent=BODY, fontSize=11, textColor=NAVY, spaceAfter=6
        ))]
        for i in range(0, len(photos), 2):
            row_imgs = []
            for j in range(2):
                if i + j >= len(photos):
                    break
                p = photos[i + j]
                thumb = p.get("thumb") or ""
                cell = []
                if thumb.startswith("data:"):
                    try:
                        raw = base64.b64decode(thumb.split(",", 1)[1])
                        cell.append(Image(io.BytesIO(raw), width=PW * 0.44, height=42 * mm))
                    except Exception:
                        cell.append(Paragraph("Frame %s" % p.get("frame", ""), CELL))
                else:
                    cell.append(Paragraph("Frame %s" % p.get("frame", ""), CELL))
                cell.append(Paragraph(str(p.get("caption", "")), SMALL))
                row_imgs.append(cell)
            if len(row_imgs) == 1:
                row_imgs.append([Paragraph("", CELL)])
            story.append(Table([row_imgs], colWidths=[PW * 0.48, PW * 0.48]))
            story.append(Spacer(1, 3 * mm))

    NumCanvas = _pdf_page_furniture(brand, "Seabed defects executive summary")
    out = io.BytesIO()
    frame = Frame(LM, 18 * mm, PW, PAGEH - 18 * mm - 26 * mm, id="f")
    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=22 * mm, bottomMargin=18 * mm)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    doc.build(story, canvasmaker=NumCanvas)
    out.seek(0)
    return out.getvalue()


def _intelligence_summary_story(meta, PW, BODY, SMALL, CELL, CELLB, CELLH, LINE):
    """Optional flowables for survey intelligence (ROI, fusion, lifecycle)."""
    meta = meta or {}
    roi = meta.get("intelligence_roi") or {}
    fused = meta.get("fused_contacts") or []
    lifecycle = meta.get("lifecycle") or {}
    items = lifecycle.get("items") if isinstance(lifecycle, dict) else {}
    if not roi and not fused and not items:
        return []
    try:
        from survey_intelligence import intelligence_summary

        summ = intelligence_summary(meta)
    except Exception:
        summ = {}
    story = [
        Paragraph(
            "Automated survey intelligence: cross-line fusion, ROV ROI, and contact lifecycle tracking.",
            BODY,
        ),
        Spacer(1, 8),
    ]
    rows = [
        ["Multi-line fused contacts", str(summ.get("multi_line_confirmed", len(fused)))],
        ["Est. ROV dives avoided", str(summ.get("dives_avoided", roi.get("dives_avoided", 0)))],
        ["Est. cost saved (USD)", "$%s" % summ.get("cost_saved_usd", roi.get("cost_saved_usd", 0))],
        ["Est. CO\u2082 saved (kg)", str(summ.get("co2_saved_kg", roi.get("co2_saved_kg", 0)))],
        ["Clearance-ready contacts", str(summ.get("clearance_count", 0))],
    ]
    if meta.get("install_date"):
        rows.append(["Install activity date", str(meta.get("install_date"))[:10]])
    tbl = Table(
        [[Paragraph(k, CELL), Paragraph("<b>%s</b>" % v, CELLB)] for k, v in rows],
        colWidths=[PW * 0.55, PW * 0.45],
    )
    tbl.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.35, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(tbl)
    if items:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Contact lifecycle states", CELLB))
        lc_rows = [["Contact ID", "State"]]
        for cid, rec in list(items.items())[:12]:
            lc_rows.append([str(cid), str((rec or {}).get("state", "ai_detected")).replace("_", " ").title()])
        lct = Table(
            [[Paragraph(c, CELLH if ri == 0 else CELL) for c in row] for ri, row in enumerate(lc_rows)],
            colWidths=[PW * 0.4, PW * 0.6],
        )
        lct.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.35, LINE)]))
        story.append(lct)
    story.append(Spacer(1, 8))
    story.append(Paragraph("See /intelligence console for full feature detail.", SMALL))
    return story


def build_clearance_certificate(meta=None):
    """One-page clearance certificate PDF from intelligence lifecycle data."""
    meta = meta or {}
    brand = meta.get("brand", BRAND)
    project = meta.get("project", "Seabed Survey")
    try:
        from survey_intelligence import clearance_contacts_from_meta, intelligence_summary

        cleared = clearance_contacts_from_meta(meta)
        summ = intelligence_summary(meta)
    except Exception:
        cleared = []
        summ = {}

    PAGEW, PAGEH = A4
    LM = RM = 18 * mm
    PW = PAGEW - LM - RM
    ss = getSampleStyleSheet()

    def stl(nm, **k):
        return ParagraphStyle(nm, parent=ss["Normal"], **k)

    H1 = stl("H1", fontSize=16, textColor=NAVY, fontName="Helvetica-Bold", alignment=TA_CENTER)
    BODY = stl("BODY", fontSize=10.5, leading=15, textColor=colors.HexColor("#1c2b33"))
    SMALL = stl("SM", fontSize=8.5, leading=12, textColor=MUT, alignment=TA_CENTER)
    CELL = stl("CELL", fontSize=9.2, leading=12, textColor=colors.HexColor("#1c2b33"))
    CELLB = stl("CELLB", fontSize=9.2, leading=12, textColor=NAVY, fontName="Helvetica-Bold")
    CELLH = stl("CELLH", fontSize=8.6, leading=11, textColor=colors.white, fontName="Helvetica-Bold")

    story = [
        Spacer(1, 12 * mm),
        Paragraph("SEABED CLEARANCE CERTIFICATE", H1),
        Spacer(1, 4 * mm),
        Paragraph(project, stl("SUB", fontSize=11, alignment=TA_CENTER, textColor=CYAND)),
        Spacer(1, 10 * mm),
        Paragraph(
            "This certificate confirms that desk review and ROV verification workflows have been applied "
            "to seabed contacts identified during the survey. Clearance-ready contacts are listed below.",
            BODY,
        ),
        Spacer(1, 8 * mm),
    ]
    if cleared:
        data = [[Paragraph("Contact ID", CELLH), Paragraph("State", CELLH)]]
        for c in cleared[:20]:
            data.append(
                [
                    Paragraph(str(c.get("contact_id", "—")), CELLB),
                    Paragraph(str(c.get("state", "")).replace("_", " ").title(), CELL),
                ]
            )
        ct = Table(data, colWidths=[PW * 0.45, PW * 0.55])
        ct.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, LINE), ("BACKGROUND", (0, 0), (-1, 0), NAVY)]))
        story.append(ct)
    else:
        story.append(
            Paragraph(
                "No contacts have reached ROV-confirmed or clearance-issued state yet. "
                "Verify contacts in Data Review and advance lifecycle in Survey Intelligence.",
                BODY,
            )
        )
    story += [
        Spacer(1, 10 * mm),
        Paragraph(
            "ROI summary: %d dive(s) avoided · $%s saved · %s kg CO\u2082 saved"
            % (
                summ.get("dives_avoided", 0),
                summ.get("cost_saved_usd", 0),
                summ.get("co2_saved_kg", 0),
            ),
            BODY,
        ),
        Spacer(1, 16 * mm),
        Paragraph("Issued: %s" % datetime.now().strftime("%d %b %Y %H:%M UTC"), BODY),
        Spacer(1, 6 * mm),
        Paragraph("%s · %s" % (brand, WEBSITE), SMALL),
    ]

    NumCanvas = _pdf_page_furniture(brand, "Clearance certificate")
    out = io.BytesIO()
    frame = Frame(LM, 18 * mm, PW, PAGEH - 18 * mm - 26 * mm, id="f")
    doc = BaseDocTemplate(out, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=22 * mm, bottomMargin=18 * mm)
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    doc.build(story, canvasmaker=NumCanvas)
    out.seek(0)
    return out.getvalue()

