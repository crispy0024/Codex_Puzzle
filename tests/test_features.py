import numpy as np
import cv2
from puzzle.features import (
    extract_edge_descriptors,
    classify_edge_types,
    EdgeFeatures,
    PieceFeatures,
)


def test_extract_edge_descriptors_simple_square():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    mask = np.ones((10, 10), dtype=np.uint8)
    corners = np.array([[0, 0], [9, 0], [9, 9], [0, 9]])
    desc = extract_edge_descriptors(img, mask, corners, edge_width=2)
    assert len(desc) == 4
    for d in desc:
        assert 'hist' in d and 'sift' in d
        assert 'hu' in d and 'color_profile' in d
        if d['hu'] is not None:
            assert len(d['hu']) == 7
        if d['color_profile'] is not None:
            assert len(d['color_profile']) == 3


def test_classify_edge_types_tab_and_hole():
    contour_tab = np.array([[0, 0], [10, 0], [12, 5], [10, 10], [0, 10]], dtype=np.int32)
    corners_tab = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
    labels_tab = classify_edge_types(contour_tab, corners_tab)
    assert labels_tab == ["flat", "tab", "flat", "flat"]

    contour_hole = np.array(
        [[0, 0], [10, 0], [10, 10], [6, 10], [6, 8], [4, 8], [4, 10], [0, 10]], dtype=np.int32
    )
    corners_hole = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32)
    labels_hole = classify_edge_types(contour_hole, corners_hole)
    assert labels_hole == ["flat", "flat", "hole", "flat"]


def test_edge_and_piece_dataclasses():
    e = EdgeFeatures(edge_type="flat", length=10.0, angle=0.0, hu_moments=None, color_hist=None, color_profile=None)
    p = PieceFeatures(contour=np.zeros((1, 2), dtype=np.int32), area=0.0, bbox=(0, 0, 1, 1), edges=[e])
    assert p.edges[0].edge_type == "flat"
