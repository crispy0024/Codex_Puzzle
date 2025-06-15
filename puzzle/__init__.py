from .segmentation import (
    remove_background,
    detect_piece_corners,
    select_four_corners,
    is_edge_straight,
    classify_piece_type,
    detect_orientation,
    rotate_piece,
    segment_pieces,
    segment_pieces_by_median,
    segment_pieces_metadata,
    extract_mask_contours,
    PuzzlePiece,
)
from .watershed import watershed_steps
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
    top_n_matches,
)
from .group import PieceGroup, merge_groups, split_group

__all__ = [
    "remove_background",
    "detect_piece_corners",
    "select_four_corners",
    "is_edge_straight",
    "classify_piece_type",
    "detect_orientation",
    "rotate_piece",
    "segment_pieces",
    "segment_pieces_by_median",
    "segment_pieces_metadata",
    "extract_mask_contours",
    "PuzzlePiece",
    "watershed_steps",
    "extract_edge_descriptors",
    "classify_edge_types",
    "EdgeFeatures",
    "PieceFeatures",
    "render_puzzle",
    "shape_similarity",
    "color_similarity",
    "compatibility_score",
    "top_n_matches",
    "PieceGroup",
    "merge_groups",
    "split_group",
]
