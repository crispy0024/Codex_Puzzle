import numpy as np
from puzzle.segmentation import select_four_corners


def test_select_four_corners_returns_four():
    pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [8, 8]])
    corners = select_four_corners(pts)
    assert corners.shape == (4, 2)
