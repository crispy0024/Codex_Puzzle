from .segmentation import (
    remove_background,
    detect_piece_corners,
    select_four_corners,
    is_edge_straight,
    classify_piece_type,
    detect_orientation,
    rotate_piece,
    segment_pieces,
    segment_pieces_metadata,
    PuzzlePiece,
)
from .features import (
    extract_edge_descriptors,
    classify_edge_types,
    EdgeFeatures,
    PieceFeatures,
)
from .assembly import render_puzzle
from .scoring import (
    shape_similarity,
    color_similarity,
    compatibility_score,
)

__all__ = [
    "remove_background",
    "detect_piece_corners",
    "select_four_corners",
    "is_edge_straight",
    "classify_piece_type",
    "detect_orientation",
    "rotate_piece",
    "segment_pieces",
    "segment_pieces_metadata",
    "PuzzlePiece",
    "extract_edge_descriptors",
    "classify_edge_types",
    "EdgeFeatures",
    "PieceFeatures",
    "render_puzzle",
    "shape_similarity",
    "color_similarity",
    "compatibility_score",
]
