import numpy as np
from puzzle.group import PieceGroup, merge_groups, split_group
from puzzle.features import EdgeFeatures, PieceFeatures


def _dummy_features(size):
    h, w = size
    contour = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    edges = [
        EdgeFeatures(edge_type="flat", length=float(w), angle=0.0, hu_moments=None, color_hist=None, color_profile=None)
        for _ in range(4)
    ]
    return PieceFeatures(contour=contour, area=float(h * w), bbox=(0, 0, w, h), edges=edges)


def test_merge_and_split_groups():
    mask = np.ones((10, 10), dtype=np.uint8)
    pf1 = _dummy_features((10, 10))
    pf2 = _dummy_features((10, 10))
    g1 = PieceGroup([1], {1: (0, 0)}, mask, pf1, {1: mask}, {1: pf1})
    g2 = PieceGroup([2], {2: (0, 0)}, mask, pf2, {2: mask}, {2: pf2})

    merged = merge_groups(g1, g2, 1, 3)
    assert merged.mask.shape == (10, 20)
    assert merged.piece_ids == [1, 2]

    remaining, removed = split_group(merged, 1)
    assert remaining.piece_ids == [2]
    assert remaining.mask.shape == (10, 10)
    assert removed.piece_ids == [1]
    assert removed.mask.shape == (10, 10)
    assert removed.features.area == pf1.area
