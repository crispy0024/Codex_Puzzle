import cv2
import numpy as np

from .features import EdgeFeatures, PieceFeatures


def shape_similarity(edge_a: EdgeFeatures, edge_b: EdgeFeatures) -> float:
    """Return a distance between two edges based on Hu moments."""
    if edge_a.hu_moments is None or edge_b.hu_moments is None:
        return float("inf")
    # Ensure arrays are float32 for cv2.norm
    a = edge_a.hu_moments.astype(np.float32)
    b = edge_b.hu_moments.astype(np.float32)
    return float(cv2.norm(a, b))


def color_similarity(edge_a: EdgeFeatures, edge_b: EdgeFeatures) -> float:
    """Return a distance between edge colors using histograms or HSV profiles."""
    if edge_a.color_hist is not None and edge_b.color_hist is not None:
        h1 = edge_a.color_hist.astype(np.float32)
        h2 = edge_b.color_hist.astype(np.float32)
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

    if edge_a.color_profile is not None and edge_b.color_profile is not None:
        p1 = edge_a.color_profile.astype(np.float32)
        p2 = edge_b.color_profile.astype(np.float32)
        return float(cv2.norm(p1, p2))

    return float("inf")


_VALID_PAIRS = {
    ("tab", "hole"),
    ("hole", "tab"),
    ("flat", "flat"),
}


def compatibility_score(
    edge_a: EdgeFeatures,
    edge_b: EdgeFeatures,
    weight_shape: float = 2.0,
    weight_color: float = 1.0,
) -> float:
    """Combine shape and color metrics for a matching score."""
    if (edge_a.edge_type, edge_b.edge_type) not in _VALID_PAIRS:
        return float("inf")

    s_shape = shape_similarity(edge_a, edge_b)
    s_color = color_similarity(edge_a, edge_b)

    if np.isinf(s_shape) or np.isinf(s_color):
        return float("inf")

    return weight_shape * s_shape + weight_color * s_color


def top_n_matches(pieces: list[PieceFeatures], n: int = 5):
    """Return the best matching edges between all pieces.

    Parameters
    ----------
    pieces : list[PieceFeatures]
        Pieces whose edges will be compared.
    n : int, optional
        Number of matches to return per edge.

    Returns
    -------
    dict
        Mapping ``(piece_id, edge_index)`` to a list of
        ``(other_piece_id, other_edge_index, score)`` tuples sorted by score.
    """

    results: dict[tuple[int, int], list[tuple[int, int, float]]] = {}

    for pid, piece in enumerate(pieces):
        for ei, edge in enumerate(piece.edges):
            if edge.edge_type == "flat":
                continue

            matches: list[tuple[int, int, float]] = []

            for opid, other in enumerate(pieces):
                if opid == pid:
                    continue
                for oei, other_edge in enumerate(other.edges):
                    if (edge.edge_type, other_edge.edge_type) not in _VALID_PAIRS:
                        continue

                    score = compatibility_score(edge, other_edge)
                    if np.isinf(score):
                        continue
                    matches.append((opid, oei, score))

            matches.sort(key=lambda x: x[2])
            results[(pid, ei)] = matches[:n]

    return results
