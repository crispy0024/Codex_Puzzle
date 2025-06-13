import numpy as np
from puzzle.scoring import shape_similarity, color_similarity, compatibility_score, top_n_matches
from puzzle.features import EdgeFeatures, PieceFeatures


def _edge(edge_type, hu, color):
    return EdgeFeatures(
        edge_type=edge_type,
        length=1.0,
        angle=0.0,
        hu_moments=np.array(hu, dtype=np.float32),
        color_hist=None,
        color_profile=np.array(color, dtype=np.float32),
    )


def _piece(edges):
    return PieceFeatures(
        contour=np.zeros((1, 2), dtype=np.int32),
        area=0.0,
        bbox=(0, 0, 1, 1),
        edges=edges,
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


def test_top_n_matches_finds_best_partner():
    tab_good = _edge("tab", [0.1] * 7, [1, 2, 3])
    tab_other = _edge("tab", [2.0] * 7, [5, 5, 5])
    piece_a = _piece([tab_good, _edge("flat", [0] * 7, [0, 0, 0]), tab_other, _edge("flat", [0] * 7, [0, 0, 0])])

    hole_match = _edge("hole", [0.1] * 7, [1, 2, 3])
    hole_other = _edge("hole", [3.0] * 7, [10, 10, 10])
    piece_b = _piece([hole_match, hole_other, _edge("tab", [0] * 7, [0, 0, 0]), _edge("flat", [0] * 7, [0, 0, 0])])

    results = top_n_matches([piece_a, piece_b], n=1)

    assert (0, 0) in results
    match = results[(0, 0)][0]
    assert match[0] == 1 and match[1] == 0
