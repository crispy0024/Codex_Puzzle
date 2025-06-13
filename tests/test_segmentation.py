import numpy as np
import cv2
from puzzle.segmentation import (
    remove_background,
    select_four_corners,
    segment_pieces,
    segment_pieces_metadata,
    PuzzlePiece,
)


def test_select_four_corners_returns_four():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [8, 8]])
    corners = select_four_corners(pts)
    assert corners.shape == (4, 2)


def test_segment_pieces_detects_regions():
    img = np.full((20, 40, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (8, 18), (0, 0, 0), -1)
    cv2.rectangle(img, (22, 2), (38, 18), (0, 0, 0), -1)
    pieces = segment_pieces(img, min_area=10)
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


def test_segment_pieces_preserves_separation_with_morphology():
    img = np.full((20, 50, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (18, 18), (0, 0, 0), -1)
    cv2.rectangle(img, (32, 2), (48, 18), (0, 0, 0), -1)
    pieces = segment_pieces(img, min_area=10, thresh_val=240, kernel_size=5)
    assert len(pieces) == 2


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
