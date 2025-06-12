import numpy as np
import cv2
from puzzle.features import extract_edge_descriptors


def test_extract_edge_descriptors_simple_square():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    mask = np.ones((10, 10), dtype=np.uint8)
    corners = np.array([[0, 0], [9, 0], [9, 9], [0, 9]])
    desc = extract_edge_descriptors(img, mask, corners, edge_width=2)
    assert len(desc) == 4
    for d in desc:
        assert 'hist' in d and 'sift' in d
