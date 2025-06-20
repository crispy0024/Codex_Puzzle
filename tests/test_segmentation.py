import numpy as np
import cv2
import pytest
from puzzle.segmentation import (
    remove_background,
    remove_background_canny,
    select_four_corners,
    apply_threshold,
    segment_pieces,
    segment_pieces_by_median,
    segment_pieces_metadata,
    detect_orientation,
    extract_mask_contours,
    PuzzlePiece,
)
from puzzle.watershed import watershed_steps


def test_select_four_corners_returns_four():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [8, 8]])
    corners = select_four_corners(pts)
    assert corners.shape == (4, 2)


def test_segment_pieces_detects_regions():
    img = np.full((20, 40, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (8, 18), (0, 0, 0), -1)
    cv2.rectangle(img, (22, 2), (38, 18), (0, 0, 0), -1)
    pieces, num = segment_pieces(img, min_area=10)
    assert num == 2
    assert len(pieces) == 2


def test_remove_background_with_threshold():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (14, 14), (0, 0, 0), -1)
    mask, _ = remove_background(img, lower_thresh=240, upper_thresh=255, iter_count=1)
    assert mask.max() == 1 and mask.min() == 0


def test_remove_background_custom_kernel_closes_holes():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (4, 4), (15, 15), (0, 0, 0), -1)
    cv2.rectangle(img, (8, 8), (11, 11), (255, 255, 255), -1)
    mask, _ = remove_background(
        img,
        lower_thresh=240,
        upper_thresh=255,
        kernel_size=5,
        iter_count=1,
    )
    n_labels, _ = cv2.connectedComponents(mask)
    assert n_labels == 2 and mask[9, 9] == 1


def test_remove_background_canny_returns_edges():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (14, 14), (0, 0, 0), -1)
    mask, result, edges = remove_background_canny(
        img, lower_thresh=240, upper_thresh=255
    )
    assert edges.shape == mask.shape and edges.dtype == np.uint8


def test_segment_pieces_preserves_separation_with_morphology():
    img = np.full((20, 50, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (18, 18), (0, 0, 0), -1)
    cv2.rectangle(img, (32, 2), (48, 18), (0, 0, 0), -1)
    pieces, num = segment_pieces(img, min_area=10, thresh_val=240, kernel_size=5)
    assert num == 2
    assert len(pieces) == 2


def test_segment_pieces_use_hull():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    contour = np.array([[2, 2], [18, 2], [18, 10], [10, 10], [10, 18], [2, 18]])
    cv2.drawContours(img, [contour], -1, (0, 0, 0), -1)
    pieces1, _ = segment_pieces(img, min_area=10, use_hull=False)
    pieces2, _ = segment_pieces(img, min_area=10, use_hull=True)
    assert len(pieces1) == len(pieces2) == 1


def test_segment_pieces_metadata_use_hull():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    contour = np.array([[2, 2], [18, 2], [18, 10], [10, 10], [10, 18], [2, 18]])
    cv2.drawContours(img, [contour], -1, (0, 0, 0), -1)
    pieces = segment_pieces_metadata(img, min_area=10, normalize=False, use_hull=True)
    assert len(pieces) == 1 and isinstance(pieces[0], PuzzlePiece)


def test_segment_pieces_metadata_returns_objects():
    img = np.full((20, 40, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (18, 18), (0, 0, 0), -1)
    pieces = segment_pieces_metadata(img, min_area=10)
    assert all(isinstance(p, PuzzlePiece) for p in pieces)


def test_segment_pieces_metadata_bbox_matches_contour():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (3, 3), (16, 16), (0, 0, 0), -1)
    pieces = segment_pieces_metadata(img, min_area=10, margin=0, normalize=False)
    assert len(pieces) == 1
    piece = pieces[0]
    bx, by, bw, bh = cv2.boundingRect(piece.contour)
    assert piece.bbox == (bx, by, bw, bh)


def test_segment_pieces_metadata_angle_zero_without_normalization():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (4, 2), (18, 16), (0, 0, 0), -1)
    pieces = segment_pieces_metadata(img, min_area=10, normalize=False)
    assert pieces[0].angle == 0.0


def _rotated_mock_piece(angle):
    base = np.full((30, 30, 3), 255, dtype=np.uint8)
    cv2.rectangle(base, (6, 10), (24, 24), (0, 0, 0), -1)
    M = cv2.getRotationMatrix2D((15, 15), angle, 1.0)
    return cv2.warpAffine(base, M, (30, 30), borderValue=(255, 255, 255))


def test_rotated_piece_normalized_angle_correct():
    img = _rotated_mock_piece(45)
    raw = segment_pieces_metadata(img, min_area=10, normalize=False)[0]
    expected = detect_orientation(raw.contour)
    pieces = segment_pieces_metadata(img, min_area=10)
    assert len(pieces) == 1
    piece = pieces[0]
    assert pytest.approx(piece.angle, abs=1) == pytest.approx(expected, abs=1)
    mask = cv2.cvtColor(piece.image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(cnts[0])
    ang = rect[2]
    if rect[1][0] < rect[1][1]:
        ang += 90
    assert pytest.approx(ang % 90, abs=1) == 0


def test_segment_pieces_by_median_filters_by_area():
    img = np.full((40, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (11, 11), (0, 0, 0), -1)
    cv2.rectangle(img, (22, 2), (31, 11), (0, 0, 0), -1)
    cv2.rectangle(img, (42, 2), (61, 21), (0, 0, 0), -1)
    pieces = segment_pieces_by_median(img, thresh_val=240)
    assert len(pieces) == 2


def test_segment_pieces_by_median_returns_contour_only():
    img = np.full((30, 30, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (14, 14), (0, 0, 0), -1)
    pieces = segment_pieces_by_median(img, thresh_val=240)
    assert len(pieces) == 1
    piece = pieces[0]
    gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    assert np.all(gray[1 : h - 1, 1 : w - 1] == 0)


def test_apply_threshold_otsu():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (15, 15), (0, 0, 0), -1)
    out = apply_threshold(img, method="otsu")
    assert out.shape == img.shape[:2]
    assert out.max() == 255 and out.min() == 0


def test_apply_threshold_adaptive():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (15, 15), (0, 0, 0), -1)
    out = apply_threshold(img, method="adaptive", block_size=3, C=1)
    assert out.shape == img.shape[:2]
    assert out.dtype == np.uint8


def test_apply_threshold_canny():
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (15, 15), (0, 0, 0), -1)
    out = apply_threshold(img, method="canny", threshold1=50, threshold2=100)
    assert out.shape == img.shape[:2]
    assert out.dtype == np.uint8


def test_extract_mask_contours_filters_by_area():
    mask = np.zeros((40, 80), dtype=np.uint8)
    cv2.rectangle(mask, (2, 2), (11, 11), 255, -1)
    cv2.rectangle(mask, (22, 2), (31, 11), 255, -1)
    cv2.rectangle(mask, (42, 2), (45, 5), 255, -1)
    pieces, num = extract_mask_contours(mask)
    assert num == 3
    assert len(pieces) == 2


def test_watershed_steps_returns_result():
    img = np.full((30, 60, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (5, 5), (25, 25), (0, 0, 0), -1)
    cv2.rectangle(img, (35, 5), (55, 25), (0, 0, 0), -1)
    outputs = watershed_steps(img)
    assert "result" in outputs and outputs["result"].shape == img.shape
