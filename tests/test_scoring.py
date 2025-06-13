import numpy as np
from puzzle.scoring import shape_similarity, color_similarity, compatibility_score
from puzzle.features import EdgeFeatures


def _edge(edge_type, hu, color):
    return EdgeFeatures(
        edge_type=edge_type,
        length=1.0,
        angle=0.0,
        hu_moments=np.array(hu, dtype=np.float32),
        color_hist=None,
        color_profile=np.array(color, dtype=np.float32),
    )


def test_identical_edges_score_lower():
    tab = _edge("tab", [0.5] * 7, [10, 20, 30])
    hole_same = _edge("hole", [0.5] * 7, [10, 20, 30])
    hole_diff = _edge("hole", [2.0] * 7, [0, 0, 0])

    good = compatibility_score(tab, hole_same)
    bad = compatibility_score(tab, hole_diff)

    assert good < bad


def test_incompatible_edge_types():
    tab1 = _edge("tab", [0.5] * 7, [10, 20, 30])
    tab2 = _edge("tab", [0.5] * 7, [10, 20, 30])
    hole = _edge("hole", [0.5] * 7, [10, 20, 30])

    assert compatibility_score(tab1, hole) < float("inf")
    assert compatibility_score(tab1, tab2) == float("inf")
