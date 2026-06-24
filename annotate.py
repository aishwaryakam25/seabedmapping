"""Draw detection boxes with non-overlapping class labels."""
import cv2

from report import class_hex, _bgr

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PAD = 3


def _label_rect(lx, ly, lw, lh):
    return [lx, ly, lx + lw, ly + lh]


def _overlaps(a, b, gap=2):
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def _fits(rect, placed, gap=2):
    return not any(_overlaps(rect, p, gap) for p in placed)


def annotate_image(img, dets, font_scale=0.45, thickness=1):
    """Return BGR image with boxes and de-collided labels."""
    out = img.copy()
    h, w = out.shape[:2]
    placed = []
    items = []

    for d in dets:
        b = d["box"]
        cls = str(d["cls"])
        conf = d.get("conf")
        if conf is not None:
            text = "%s %d%%" % (cls.upper(), round(float(conf) * 100))
        else:
            text = cls.upper()
        (tw, th), _ = cv2.getTextSize(text, _FONT, font_scale, thickness)
        lw, lh = tw + _PAD * 2, th + _PAD * 2
        bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]

        lx = max(0, min(bx1, w - lw))
        ly = max(0, by1 - lh - 2)
        rect = _label_rect(lx, ly, lw, lh)

        for _ in range(40):
            if _fits(rect, placed):
                break
            ly -= lh + 2
            if ly < 0:
                ly = by2 + 2
            if ly + lh > h:
                ly = min(by1 + 2, h - lh)
            lx = min(lx + lw // 2, w - lw)
            rect = _label_rect(lx, ly, lw, lh)
        else:
            ly = min(by1 + 2, h - lh)
            lx = max(0, min(bx1, w - lw))
            rect = _label_rect(lx, ly, lw, lh)

        placed.append(rect)
        items.append((b, text, lx, ly, lw, lh, tw, th, class_hex(cls)))

    for b, text, lx, ly, lw, lh, tw, th, hexcol in items:
        col = _bgr(hexcol)
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), col, 2)
        cv2.rectangle(out, (lx, ly), (lx + lw, ly + lh), col, -1)
        cv2.putText(
            out, text, (lx + _PAD, ly + lh - _PAD - 1),
            _FONT, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return out
