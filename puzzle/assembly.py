import cv2
import numpy as np


def render_puzzle(placements, piece_images, canvas_size):
    """Render assembled puzzle given piece placements.

    Parameters
    ----------
    placements : list of tuples
        Sequence of (piece_id, x, y, rotation_deg).
    piece_images : dict
        Mapping piece_id -> BGR image for that piece.
    canvas_size : tuple
        (height, width) of the resulting canvas.
    """
    canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    for piece_id, x, y, rot in placements:
        img = piece_images[piece_id]
        if rot % 360 != 0:
            center = (img.shape[1] // 2, img.shape[0] // 2)
            M = cv2.getRotationMatrix2D(center, rot, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        h, w = img.shape[:2]
        x_end = min(canvas.shape[1], x + w)
        y_end = min(canvas.shape[0], y + h)
        canvas[y:y_end, x:x_end] = np.where(
            img[0 : y_end - y, 0 : x_end - x] == 0,
            canvas[y:y_end, x:x_end],
            img[0 : y_end - y, 0 : x_end - x],
        )
    return canvas
