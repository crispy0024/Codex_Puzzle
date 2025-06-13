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
    """Segment a puzzle piece from the background using GrabCut.


    Parameters
    ----------
    piece_img : ndarray
        BGR image of a single puzzle piece.
    iter_count : int, optional

        Number of GrabCut iterations.
    rect_margin : int, optional
        Margin for the rectangle initialization when no thresholds are
        provided.
    lower_thresh, upper_thresh : int or None, optional
        Grayscale intensity bounds describing the background. When provided
        an initial mask is created with pixels inside this range marked as
        background and GrabCut is initialized with ``cv2.GC_INIT_WITH_MASK``.
        If either value is ``None`` the rectangle based initialization is
        used instead.
    kernel_size : int or None, optional
        Size of the morphological kernel to clean up the resulting mask.
        When ``None`` or less than 2 no morphology is applied.
    """


    mask = np.zeros(piece_img.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    use_rect = True
    if lower_thresh is not None and upper_thresh is not None:
        gray = cv2.cvtColor(piece_img, cv2.COLOR_BGR2GRAY)
        thresh_mask = cv2.inRange(gray, lower_thresh, upper_thresh)
        has_bgd = np.any(thresh_mask == 255)
        has_obj = np.any(thresh_mask == 0)
        if has_bgd and has_obj:
            mask.fill(cv2.GC_PR_FGD)
            mask[thresh_mask == 255] = cv2.GC_BGD
            cv2.grabCut(
                piece_img,
                mask,
                None,
                bgd_model,
                fgd_model,
                iter_count,
                cv2.GC_INIT_WITH_MASK,
            )
            use_rect = False

    if use_rect:
        h, w = piece_img.shape[:2]
        rect = (rect_margin, rect_margin, w - 2 * rect_margin, h - 2 * rect_margin)
        cv2.grabCut(
            piece_img,
            mask,
            rect,
            bgd_model,
            fgd_model,
            iter_count,
            cv2.GC_INIT_WITH_RECT,
        )

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


def segment_pieces(
    image,
    min_area: int = 1000,
    thresh_val: int = 250,
    kernel_size: int = 3,
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

    Returns
    -------
    list[numpy.ndarray]
        Cropped BGR images, one for each detected piece.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

    if kernel_size and kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    pieces = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        piece = image[y : y + h, x : x + w]
        pieces.append(piece)

    return pieces


def segment_pieces_metadata(image, min_area: int = 1000, margin: int = 5, normalize: bool = True):
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

        x, y, w, h = cv2.boundingRect(cnt)
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
                crop, local_cnt, extra = _validate_piece_orientation(crop, local_cnt, piece_type)
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
