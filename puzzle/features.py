import cv2
import numpy as np


def _edge_strip_mask(mask, pt1, pt2, width):
    """Create a binary mask for a strip along an edge defined by two points."""
    strip = np.zeros_like(mask, dtype=np.uint8)
    cv2.line(strip, tuple(pt1), tuple(pt2), color=1, thickness=width)
    return (strip & mask).astype(np.uint8)


def _color_histogram(img, mask, bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _sift_descriptors(img, mask):
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        return None
    keypoints, desc = sift.detectAndCompute(img, mask)
    if desc is None:
        return None
    return desc.mean(axis=0)


def extract_edge_descriptors(image, mask, corners, edge_width: int = 15, hist_bins: int = 8):
    """Compute descriptors for each edge of a piece.

    Parameters
    ----------
    image : ndarray
        Original BGR image of the puzzle piece.
    mask : ndarray
        Binary mask of the piece (1=piece, 0=background).
    corners : ndarray
        4x2 array of corner coordinates in order.
    edge_width : int
        Thickness of the strip sampled along each edge.
    hist_bins : int
        Number of bins per channel for the color histogram.

    Returns
    -------
    list[dict]
        A list of dictionaries with keys 'hist' and 'sift' for each edge.
    """
    descriptors = []
    for i in range(4):
        pt1 = corners[i]
        pt2 = corners[(i + 1) % 4]
        strip_mask = _edge_strip_mask(mask, pt1, pt2, edge_width)
        if np.count_nonzero(strip_mask) == 0:
            descriptors.append({'hist': None, 'sift': None})
            continue
        hist = _color_histogram(image, strip_mask * 255, bins=hist_bins)
        sift = _sift_descriptors(image, strip_mask * 255)
        descriptors.append({'hist': hist, 'sift': sift})
    return descriptors
