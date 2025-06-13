import cv2
import numpy as np
from sklearn.cluster import KMeans


def remove_background(
    piece_img, iter_count: int = 5, rect_margin: int = 1, kernel_size: int = 3
):
    """Segment a puzzle piece from the background using GrabCut.

    After the GrabCut mask is created, the result is cleaned with a
    morphological closing followed by an opening to remove small holes and
    noise. The size of the structuring element controlling these operations is
    configurable.

    Parameters
    ----------
    piece_img : ndarray
        BGR image of a single puzzle piece.
    iter_count : int, optional
        Number of GrabCut iterations to run.
    rect_margin : int, optional
        Margin in pixels around the piece when initializing GrabCut.
    kernel_size : int, optional
        Size of the square structuring element for morphological operations.

    Returns
    -------
    tuple(ndarray, ndarray)
        ``(mask, segmented_image)`` where ``mask`` is the cleaned binary mask
        and ``segmented_image`` is the piece with background removed.
    """
    mask = np.zeros(piece_img.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    h, w = piece_img.shape[:2]
    rect = (rect_margin, rect_margin, w - 2 * rect_margin, h - 2 * rect_margin)
    cv2.grabCut(piece_img, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")

    if kernel_size and kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    segmented = piece_img * mask2[:, :, np.newaxis]
    return mask2, segmented


def detect_piece_corners(mask, max_corners: int = 20, quality_level: float = 0.01, min_distance: int = 10):
    """Detect corner points on the piece boundary using Shi-Tomasi."""
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    corners = cv2.goodFeaturesToTrack(edges, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    if corners is None:
        return np.empty((0, 2), dtype=np.int32)
    return corners.astype(np.int32).reshape(-1, 2)


def select_four_corners(corner_pts):
    """Reduce many detected corners to four cluster centers.

    Parameters
    ----------
    corner_pts : ndarray of shape ``(n, 2)``
        Candidate corner coordinates as ``(x, y)`` pairs.

    Returns
    -------
    ndarray
        Array of four ``(x, y)`` points when ``n >= 4`` otherwise the input
        array is returned.
    """
    if len(corner_pts) < 4:
        return corner_pts
    kmeans = KMeans(n_clusters=4, random_state=42).fit(corner_pts)
    return np.int32(kmeans.cluster_centers_)


def is_edge_straight(mask, corner1, corner2, tolerance: float = 0.98, sample_points: int = 50):
    """Check if the boundary between two corners is mostly straight.

    Parameters
    ----------
    mask : ndarray
        Binary ``(H, W)`` mask of the piece.
    corner1, corner2 : array-like
        End points ``(x, y)`` of the edge to test.
    tolerance : float, optional
        Minimum fraction of sampled points that must lie on ``mask``.
    sample_points : int, optional
        Number of points to sample along the edge.

    Returns
    -------
    bool
        ``True`` if the portion of points on the mask meets ``tolerance``.
    """
    x1, y1 = corner1
    x2, y2 = corner2
    h, w = mask.shape
    x_coords = np.linspace(x1, x2, sample_points).astype(int)
    y_coords = np.linspace(y1, y2, sample_points).astype(int)
    valid = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
    x_coords = x_coords[valid]
    y_coords = y_coords[valid]
    if len(x_coords) == 0:
        return False
    points_on_mask = mask[y_coords, x_coords]
    percentage = np.sum(points_on_mask) / sample_points
    return percentage >= tolerance


def classify_piece_type(mask, corners):
    """Heuristic classification into corner, edge or middle piece."""
    straight_count = 0
    for i in range(4):
        a = tuple(corners[i])
        b = tuple(corners[(i + 1) % 4])
        if is_edge_straight(mask, a, b, tolerance=0.98):
            straight_count += 1
    if straight_count == 2:
        return "corner"
    elif straight_count == 1:
        return "edge"
    return "middle"


def segment_pieces(image, min_area: int = 1000):
    """Segment an image containing multiple puzzle pieces.

    This utility performs a naive foreground extraction by thresholding
    the image and returning a list of cropped piece images. It is not
    perfect but works reasonably well when pieces are placed on a light
    background.

    Parameters
    ----------
    image : ndarray
        BGR image that potentially contains many pieces.
    min_area : int, optional
        Minimum contour area to consider a region a puzzle piece.

    Returns
    -------
    list[numpy.ndarray]
        Cropped BGR images, one for each detected piece.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pieces = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        piece = image[y : y + h, x : x + w]
        pieces.append(piece)

    return pieces
