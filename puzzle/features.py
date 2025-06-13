import cv2
import numpy as np


def _edge_strip_mask(mask, pt1, pt2, width):
    """Return a mask for a line segment within a piece.

    Parameters
    ----------
    mask : ndarray
        Binary ``(H, W)`` mask of the piece where non-zero pixels are valid.
    pt1, pt2 : array-like
        End points of the edge segment ``(x, y)``.
    width : int
        Thickness in pixels of the strip to draw.

    Returns
    -------
    ndarray
        Binary mask of the same shape as ``mask`` with only the strip set.
    """
    strip = np.zeros_like(mask, dtype=np.uint8)
    cv2.line(strip, tuple(pt1), tuple(pt2), color=1, thickness=width)
    return (strip & mask).astype(np.uint8)


def _color_histogram(img, mask, bins=8):
    """Calculate a normalized HSV histogram for the masked region.

    Parameters
    ----------
    img : ndarray
        Input BGR image.
    mask : ndarray
        Binary mask with the same height and width as ``img``.
        Non-zero pixels indicate the area to sample.
    bins : int, optional
        Number of bins per HSV channel.

    Returns
    -------
    ndarray
        Flattened histogram vector.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _sift_descriptors(img, mask):
    """Compute the mean SIFT descriptor within a masked region.

    Parameters
    ----------
    img : ndarray
        BGR image from which to extract features.
    mask : ndarray
        Binary ``(H, W)`` mask selecting pixels to consider.

    Returns
    -------
    ndarray or None
        Averaged descriptor vector or ``None`` if SIFT is unavailable or no
        keypoints are detected.
    """
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
