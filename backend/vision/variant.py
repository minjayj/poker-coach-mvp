"""Conservative ClubGG table-type cue from a single shared application window.

The recordings show a green NLH felt and a blue PLO5 felt. This is a cue,
not card OCR: overlays, themes, or a desktop containing multiple windows must
cause abstention rather than a confident switch.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np


def detect_table_variant(image_data: str) -> tuple[str, float] | None:
    try:
        encoded = image_data.split(",", 1)[1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (IndexError, ValueError):
        return None
    return detect_table_variant_frame(frame)


def detect_table_variant_frame(frame: np.ndarray | None) -> tuple[str, float] | None:
    """Classify an already decoded, single ClubGG table view."""
    if frame is None or min(frame.shape[:2]) < 180:
        return None
    height, width = frame.shape[:2]
    # The middle may be covered by four/five board cards. Require both side
    # felt patches to agree; this also abstains on mixed-color multi-table views.
    votes: list[tuple[str, float]] = []
    for x1, x2 in ((.23, .30), (.68, .75)):
        crop = frame[int(height * .43):int(height * .57), int(width * x1):int(width * x2)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturated = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 35)
        if float(saturated.mean()) < .65:
            return None
        hue = hsv[:, :, 0]
        green = float(((hue >= 40) & (hue < 100) & saturated).sum()) / float(saturated.sum())
        blue = float(((hue >= 100) & (hue < 140) & saturated).sum()) / float(saturated.sum())
        if green >= .80:
            votes.append(("nlh", green))
        elif blue >= .80:
            votes.append(("plo5", blue))
        else:
            return None
    if votes[0][0] == votes[1][0]:
        return votes[0][0], round(min(votes[0][1], votes[1][1]), 3)
    return None
