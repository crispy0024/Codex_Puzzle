import cv2
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass


@dataclass
class PuzzlePiece:
    """Container for a segmented puzzle piece and its metadata."""

    image: np.ndarray
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    angle: float = 0.0


def remove_background(
    piece_img,
    iter_count: int = 5,
    rect_margin: int = 1,
    lower_thresh: int | None = None,
    upper_thresh: int | None = None,
    kernel_size: int | None = None,
):
    """Segment a puzzle piece from the background using the watershed algorithm.


    Parameters
    ----------
    piece_img : ndarray
        BGR image of a single puzzle piece.
    iter_count : int, optional
        Ignored, present for backward compatibility.
    rect_margin : int, optional
        Ignored, present for backward compatibility.
    lower_thresh, upper_thresh : int or None, optional
        Grayscale intensity bounds describing the background. When provided
        they are used to create the initial binary mask prior to watershed.
    kernel_size : int or None, optional
        Size of the morphological kernel to clean up the resulting mask.
        When ``None`` or less than 2 no morphology is applied.
    """

    gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)

    if lower_thresh is not None and upper_thresh is not None:
        mask2 = cv2.inRange(gray, lower_thresh, upper_thresh)
        mask2 = cv2.bitwise_not(mask2)
        mask2 = (mask2 > 0).astype("uint8")
    else:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        _, markers = cv2.connectedComponents(clean)
        markers = markers + 1

        markers = cv2.watershed(piece_img.copy(), markers)
        mask2 = np.where(markers > 1, 1, 0).astype("uint8")

    if kernel_size and kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

    segmented = piece_img * mask2[:, :, np.newaxis]
    return mask2, segmented


def remove_background_canny(
    piece_img,
    canny1: int = 50,
    canny2: int = 150,
    **kwargs,
):
    """Run :func:`remove_background` followed by Canny edge detection."""

    mask, result = remove_background(
        piece_img,
        lower_thresh=kwargs.get("lower_thresh"),
        upper_thresh=kwargs.get("upper_thresh"),
        kernel_size=kwargs.get("kernel_size"),
    )
    edges = cv2.Canny((mask * 255).astype(np.uint8), canny1, canny2)
    return mask, result, edges


def detect_piece_corners(
    mask, max_corners: int = 20, quality_level: float = 0.01, min_distance: int = 10
):
    """Detect corner points on the piece boundary using Shi-Tomasi."""
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    corners = cv2.goodFeaturesToTrack(
        edges,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
    )
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


def is_edge_straight(
    mask, corner1, corner2, tolerance: float = 0.98, sample_points: int = 50
):
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


def detect_orientation(contour):
    """Return the rotation angle that best aligns a contour upright."""
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    if rect[1][0] < rect[1][1]:
        angle += 90
    return angle


def rotate_piece(img, contour, angle):
    """Rotate an image and associated contour by ``angle`` degrees."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    cnt = contour
    if cnt.ndim == 2:
        cnt = cnt[:, np.newaxis, :]
    rotated_cnt = cv2.transform(cnt.astype(np.float32), rot_mat).astype(np.int32)
    return rotated, rotated_cnt


def _validate_piece_orientation(img, contour, piece_type):
    """Rotate by 90 degree increments until straight edges are axis aligned."""
    if piece_type not in {"corner", "edge"}:
        return img, contour, 0.0

    applied = 0.0
    for _ in range(4):
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        corners = select_four_corners(detect_piece_corners(thresh))
        if len(corners) == 4:
            flat_angles = []
            for i in range(4):
                a = corners[i]
                b = corners[(i + 1) % 4]
                if is_edge_straight(thresh, a, b, tolerance=0.98):
                    ang = abs(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))) % 180
                    flat_angles.append(ang)
            required = 2 if piece_type == "corner" else 1
            hv_count = sum(ang < 15 or abs(ang - 90) < 15 for ang in flat_angles)
            if hv_count >= required:
                break
        img, contour = rotate_piece(img, contour, 90)
        applied += 90
    return img, contour, applied % 360


def apply_threshold(image, method: str = "otsu", **params):
    """Return a binary mask using one of several thresholding methods."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    method = method.lower()

    if method == "adaptive":
        block = int(params.get("block_size", 11))
        C = int(params.get("C", 2))
        if block % 2 == 0:
            block += 1
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            C,
        )
        return thresh

    if method == "canny":
        t1 = int(params.get("threshold1", 50))
        t2 = int(params.get("threshold2", 150))
        return cv2.Canny(gray, t1, t2)

    # default to otsu/global threshold
    thresh_val = int(params.get("thresh_val", 0))
    flag = cv2.THRESH_BINARY_INV
    if method == "otsu" or thresh_val == 0:
        ret, thresh = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    else:
        ret, thresh = cv2.threshold(gray, thresh_val, 255, flag)
    return thresh


def segment_pieces(
    image,
    min_area: int = 1000,
    thresh_val: int = 250,
    kernel_size: int = 3,
    method: str = "otsu",
    use_hull: bool = False,
    **params,
):
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
    thresh_val : int, optional
        Grayscale threshold used to separate the pieces from the background.
    kernel_size : int, optional
        Size of the morphological kernel applied to smooth ``thresh``. When
        less than ``2`` the morphology step is skipped.
    use_hull : bool, optional
        When ``True`` compute the convex hull of each contour before
        extracting bounding boxes.

    Returns
    -------
    tuple[list[numpy.ndarray], int]
        Cropped BGR images and the number of detected contours before
        filtering by ``min_area``.
    """
    thresh = apply_threshold(
        image,
        method=method,
        thresh_val=thresh_val,
        **params,
    )

    if kernel_size and kernel_size > 1 and method != "canny":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num_contours = len(contours)
    pieces = []
    for cnt in contours:
        box_cnt = cv2.convexHull(cnt) if use_hull else cnt
        x, y, w, h = cv2.boundingRect(box_cnt)
        if w * h < min_area:
            continue
        piece = image[y : y + h, x : x + w]
        pieces.append(piece)

    return pieces, num_contours


def segment_pieces_by_median(
    image,
    thresh_val: int = 250,
    kernel_size: int = 3,
    method: str = "otsu",
    min_area: int = 0,
    **params,
):
    """Segment pieces and filter by contour area around the median.

    The image is thresholded in the same way as :func:`segment_pieces` to
    obtain contours.  After all contours are detected, the median area is
    computed and only pieces whose area falls within ``75%`` to ``125%`` of
    this median are kept.  For every accepted contour an image is returned
    that contains a blank background with only the contour outline drawn on
    it.

    Parameters
    ----------
    image : ndarray
        BGR image that potentially contains many pieces.
    thresh_val : int, optional
        Threshold value used to separate foreground from background.
    kernel_size : int, optional
        Size of the morphological kernel used to clean up the mask.  When less
        than ``2`` the morphology step is skipped.

    Returns
    -------
    list[numpy.ndarray]
        Cropped images with only the outer contour drawn.
    """

    thresh = apply_threshold(
        image,
        method=method,
        thresh_val=thresh_val,
        **params,
    )

    if kernel_size and kernel_size > 1 and method != "canny":
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    filtered = [
        (c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area
    ]
    if not filtered:
        return []

    areas = [a for _, a in filtered]
    median = float(np.median(areas))
    lower = 0.75 * median
    upper = 1.25 * median

    outputs = []
    for cnt, area in filtered:
        if area < lower or area > upper:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        local = cnt - [x, y]
        cv2.drawContours(blank, [local], -1, (255, 255, 255), 1)
        outputs.append(blank)

    return outputs


def extract_mask_contours(mask, area_ratio: float = 0.25):
    """Return contour crops from a binary mask exceeding ``area_ratio`` of the
    median area.

    Parameters
    ----------
    mask : ndarray
        Binary image where puzzle pieces are white (non-zero).
    area_ratio : float, optional
        Fraction of the median area used as the cutoff for filtering.

    Returns
    -------
    tuple[list[numpy.ndarray], int]
        Cropped contour images and the total number of detected contours.
    """

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype("uint8")

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)
    if not contours:
        return [], num_contours

    areas = [cv2.contourArea(c) for c in contours]
    median = float(np.median(areas))
    threshold = area_ratio * median

    outputs = []
    for cnt in contours:
        if cv2.contourArea(cnt) < threshold:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        local = cnt - [x, y]
        cv2.drawContours(blank, [local], -1, (255, 255, 255), 1)
        outputs.append(blank)

    return outputs, num_contours


def extract_clean_pieces(image, blur: int = 5, kernel_size: int = 3):
    """Simple piece extraction with blur and morphology.

    The input image is blurred, thresholded and then cleaned using an
    erosion followed by dilation to eliminate small particles. Each
    resulting contour crop is returned.

    Parameters
    ----------
    image : ndarray
        BGR image containing one or more puzzle pieces.
    blur : int, optional
        Size of the Gaussian blur kernel. Must be odd.
    kernel_size : int, optional
        Size of the morphological kernel for the open/close operations.

    Returns
    -------
    list[numpy.ndarray]
        Cropped piece images.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur and blur % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    if kernel_size and kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pieces = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        piece_img = image[y : y + h, x : x + w]
        pieces.append(piece_img)

    return pieces


def segment_pieces_metadata(
    image, min_area: int = 1000, margin: int = 5, normalize: bool = True, use_hull: bool = False
):
    """Segment puzzle pieces and return metadata for each piece.

    Parameters
    ----------
    image : ndarray
        BGR image containing puzzle pieces.
    min_area : int, optional
        Minimum contour area to consider a region a puzzle piece.
    margin : int, optional
        Extra pixels to include around the detected bounding box.
    normalize : bool, optional
        When ``True`` rotate each cropped piece using the angle from
        ``cv2.minAreaRect`` so that it is axis aligned. The applied rotation
        angle is stored in the returned :class:`PuzzlePiece` objects.
    use_hull : bool, optional
        Compute the convex hull before extracting the bounding box when
        ``True``.

    Returns
    -------
    list[PuzzlePiece]
        Cropped pieces along with contour and bounding box metadata.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = image.shape[:2]
    pieces = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        angle = detect_orientation(cnt)

        box_cnt = cv2.convexHull(cnt) if use_hull else cnt
        x, y, w, h = cv2.boundingRect(box_cnt)
        x0 = max(x - margin, 0)
        y0 = max(y - margin, 0)
        x1 = min(x + w + margin, w_img)
        y1 = min(y + h + margin, h_img)

        crop = image[y0:y1, x0:x1].copy()
        local_cnt = cnt - [x0, y0]
        applied_angle = 0.0
        if normalize:
            crop, local_cnt = rotate_piece(crop, local_cnt, angle)
            mask = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, m = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            corners = select_four_corners(detect_piece_corners(m))
            if len(corners) == 4:
                piece_type = classify_piece_type(m, corners)
                crop, local_cnt, extra = _validate_piece_orientation(
                    crop, local_cnt, piece_type
                )
                angle += extra
            applied_angle = angle

        piece = PuzzlePiece(
            image=crop,
            contour=local_cnt if normalize else cnt,
            bbox=(x0, y0, x1 - x0, y1 - y0),
            angle=applied_angle if normalize else 0.0,
        )
        pieces.append(piece)

    return pieces
