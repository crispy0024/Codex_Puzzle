import cv2
import numpy as np
from dataclasses import dataclass, field

from .features import EdgeFeatures, PieceFeatures, extract_edge_descriptors, classify_edge_types
from .segmentation import detect_piece_corners, select_four_corners


def _compute_features(img: np.ndarray, mask: np.ndarray) -> PieceFeatures:
    conts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        contour = np.zeros((0, 2), dtype=np.int32)
        area = 0.0
        bbox = (0, 0, 0, 0)
        edges: list[EdgeFeatures] = []
        return PieceFeatures(contour=contour, area=area, bbox=bbox, edges=edges)
    contour = max(conts, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    bbox = cv2.boundingRect(contour)
    corners = select_four_corners(detect_piece_corners(mask))
    if len(corners) < 4:
        h, w = mask.shape
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    descs = extract_edge_descriptors(img, mask, corners)
    labels = classify_edge_types(contour, corners)
    edges: list[EdgeFeatures] = []
    for i, d in enumerate(descs):
        pt1 = corners[i]
        pt2 = corners[(i + 1) % 4]
        length = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
        angle = float(np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])))
        edges.append(
            EdgeFeatures(
                edge_type=labels[i] if i < len(labels) else "flat",
                length=length,
                angle=angle,
                hu_moments=np.array(d["hu"]) if d["hu"] is not None else None,
                color_hist=np.array(d["hist"]) if d["hist"] is not None else None,
                color_profile=np.array(d["color_profile"]) if d["color_profile"] is not None else None,
            )
        )
    return PieceFeatures(contour=contour.squeeze(), area=area, bbox=bbox, edges=edges)


@dataclass
class PieceGroup:
    """Collection of pieces treated as a single unit."""

    piece_ids: list[int]
    transforms: dict[int, tuple[int, int]]
    mask: np.ndarray
    features: PieceFeatures
    piece_masks: dict[int, np.ndarray] = field(default_factory=dict)
    original_features: dict[int, PieceFeatures] = field(default_factory=dict)


def _place_mask(base: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    h, w = mask.shape
    base[y : y + h, x : x + w] = np.maximum(base[y : y + h, x : x + w], mask)


def merge_groups(group_a: PieceGroup, group_b: PieceGroup, edge_a: int, edge_b: int) -> PieceGroup:
    """Return new group composed of ``group_a`` and ``group_b``."""

    h1, w1 = group_a.mask.shape
    h2, w2 = group_b.mask.shape
    tx, ty = 0, 0
    if edge_a == 1 and edge_b == 3:
        tx = w1
    elif edge_a == 3 and edge_b == 1:
        tx = -w2
    elif edge_a == 2 and edge_b == 0:
        ty = h1
    elif edge_a == 0 and edge_b == 2:
        ty = -h2

    off_x1 = max(0, -tx)
    off_y1 = max(0, -ty)
    off_x2 = off_x1 + tx
    off_y2 = off_y1 + ty
    new_w = max(off_x1 + w1, off_x2 + w2)
    new_h = max(off_y1 + h1, off_y2 + h2)
    new_mask = np.zeros((new_h, new_w), dtype=np.uint8)
    _place_mask(new_mask, group_a.mask, off_x1, off_y1)
    _place_mask(new_mask, group_b.mask, off_x2, off_y2)
    img = np.dstack([new_mask * 255] * 3)
    features = _compute_features(img, new_mask)

    piece_ids = group_a.piece_ids + group_b.piece_ids
    transforms: dict[int, tuple[int, int]] = {}
    for pid in group_a.piece_ids:
        x, y = group_a.transforms.get(pid, (0, 0))
        transforms[pid] = (x + off_x1, y + off_y1)
    for pid in group_b.piece_ids:
        x, y = group_b.transforms.get(pid, (0, 0))
        transforms[pid] = (x + off_x2, y + off_y2)

    piece_masks = {**group_a.piece_masks, **group_b.piece_masks}
    original_features = {**group_a.original_features, **group_b.original_features}

    return PieceGroup(
        piece_ids=piece_ids,
        transforms=transforms,
        mask=new_mask,
        features=features,
        piece_masks=piece_masks,
        original_features=original_features,
    )


def split_group(group: PieceGroup, piece_id: int) -> tuple[PieceGroup, PieceGroup]:
    """Remove ``piece_id`` and return (remaining_group, removed_group)."""

    if piece_id not in group.piece_ids:
        raise ValueError("piece_id not in group")

    remain_ids = [pid for pid in group.piece_ids if pid != piece_id]
    remain_transforms = {pid: t for pid, t in group.transforms.items() if pid != piece_id}
    remain_masks = {pid: m for pid, m in group.piece_masks.items() if pid != piece_id}
    remain_feats = {pid: f for pid, f in group.original_features.items() if pid != piece_id}

    if remain_ids:
        xs = [t[0] for t in remain_transforms.values()]
        ys = [t[1] for t in remain_transforms.values()]
        min_x, min_y = min(xs), min(ys)
        shift_x = -min_x
        shift_y = -min_y
        w = 0
        h = 0
        for pid in remain_ids:
            mask = remain_masks[pid]
            x, y = remain_transforms[pid]
            w = max(w, x + shift_x + mask.shape[1])
            h = max(h, y + shift_y + mask.shape[0])
        new_mask = np.zeros((h, w), dtype=np.uint8)
        for pid in remain_ids:
            mask = remain_masks[pid]
            x, y = remain_transforms[pid]
            _place_mask(new_mask, mask, x + shift_x, y + shift_y)
            remain_transforms[pid] = (x + shift_x, y + shift_y)
        img = np.dstack([new_mask * 255] * 3)
        features = _compute_features(img, new_mask)
    else:
        new_mask = np.zeros((1, 1), dtype=np.uint8)
        features = _compute_features(np.zeros((1, 1, 3), dtype=np.uint8), new_mask)

    remaining = PieceGroup(
        piece_ids=remain_ids,
        transforms=remain_transforms,
        mask=new_mask,
        features=features,
        piece_masks=remain_masks,
        original_features=remain_feats,
    )

    removed_mask = group.piece_masks[piece_id]
    removed_feat = group.original_features[piece_id]
    removed = PieceGroup(
        piece_ids=[piece_id],
        transforms={piece_id: (0, 0)},
        mask=removed_mask,
        features=removed_feat,
        piece_masks={piece_id: removed_mask},
        original_features={piece_id: removed_feat},
    )

    return remaining, removed
