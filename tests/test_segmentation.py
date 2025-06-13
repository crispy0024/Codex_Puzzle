import numpy as np
import cv2
from puzzle.segmentation import remove_background, select_four_corners, segment_pieces


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
