import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class EdgeFeatures:
    """Structure holding measurements for a single edge."""

    edge_type: str
    length: float
    angle: float
    hu_moments: np.ndarray | None
    color_hist: np.ndarray | None
    color_profile: np.ndarray | None


@dataclass
class PieceFeatures:
    """Collection of contour level metrics for an entire piece."""

    contour: np.ndarray
    area: float
    bbox: tuple[int, int, int, int]
    edges: list[EdgeFeatures]


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


def classify_edge_types(contour, corners):
    """Label edges as ``tab``, ``hole`` or ``flat`` using convexity analysis."""

    if contour.ndim == 3:
        contour = contour[:, 0, :]

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    hull_pts = cv2.convexHull(contour, returnPoints=True)[:, 0, :]
    defects = cv2.convexityDefects(contour, hull_idx)

    def _nearest_idx(pt):
        d = np.linalg.norm(contour - pt, axis=1)
        return int(np.argmin(d))

    def _segment(points, i1, i2):
        if i1 <= i2:
            return points[i1 : i2 + 1]
        return np.vstack((points[i1:], points[: i2 + 1]))

    def _arc_length(points, i1, i2):
        seg = _segment(points, i1, i2)
        return cv2.arcLength(seg, False)

    labels = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        idx1 = _nearest_idx(p1)
        idx2 = _nearest_idx(p2)
        seg = _segment(contour, idx1, idx2)
        h1 = np.argmin(np.linalg.norm(hull_pts - p1, axis=1))
        h2 = np.argmin(np.linalg.norm(hull_pts - p2, axis=1))
        hull_order = hull_idx.squeeze()

        if h1 <= h2:
            num_hull_vertices = h2 - h1
        else:
            num_hull_vertices = len(hull_order) - (h1 - h2)

        is_hole = False
        if defects is not None:
            for d in defects[:, 0]:
                start, end = d[0], d[1]
                if idx1 <= idx2:
                    if start >= idx1 and end <= idx2:
                        is_hole = True
                        break
                else:
                    if start >= idx1 or end <= idx2:
                        is_hole = True
                        break

        if is_hole:
            labels.append("hole")
        elif num_hull_vertices > 1:
            labels.append("tab")
        else:
            labels.append("flat")

    return labels


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
        A list of dictionaries describing each edge. Keys include 'hist',
        'sift', 'hu' and 'color_profile'.
    """
    descriptors = []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for i in range(4):
        pt1 = corners[i]
        pt2 = corners[(i + 1) % 4]
        strip_mask = _edge_strip_mask(mask, pt1, pt2, edge_width)
        if np.count_nonzero(strip_mask) == 0:
            descriptors.append({'hist': None, 'sift': None, 'hu': None, 'color_profile': None})
            continue
        hist = _color_histogram(image, strip_mask * 255, bins=hist_bins)
        sift = _sift_descriptors(image, strip_mask * 255)

        moments = cv2.moments(strip_mask)
        hu = cv2.HuMoments(moments).flatten()

        # Average HSV values along the edge strip as a simple color profile
        colors = hsv[strip_mask.astype(bool)]
        profile = colors.mean(axis=0) if colors.size else None

        descriptors.append({'hist': hist, 'sift': sift, 'hu': hu, 'color_profile': profile})
    return descriptors
