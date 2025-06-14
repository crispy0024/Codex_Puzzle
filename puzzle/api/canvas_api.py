import base64
import json
import logging
import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..group import merge_groups
from .state import canvas_items, merge_history
from ..scoring import top_n_matches
from .rl_api import rank_with_policy

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/merge_pieces")
async def merge_pieces_endpoint(request: Request):
    data = await request.json()
    piece_ids = data.get("piece_ids")
    if not isinstance(piece_ids, list) or len(piece_ids) < 2:
        return JSONResponse(status_code=400, content={"error": "piece_ids must be list of at least 2"})
    items = [it for it in canvas_items if it["id"] in piece_ids]
    if len(items) != len(piece_ids):
        return JSONResponse(status_code=400, content={"error": "invalid piece id"})
    merged_group = items[0]["group"]
    for it in items[1:]:
        merged_group = merge_groups(merged_group, it["group"], 1, 3)
    new_id = max([it["id"] for it in canvas_items] + [0]) + 1
    new_item = {"id": new_id, "group": merged_group, "x": 0, "y": 0, "type": "group"}
    remaining = [it for it in canvas_items if it["id"] not in piece_ids]
    canvas_items.clear()
    canvas_items.extend(remaining + [new_item])
    merge_history.append((new_item, items))
    img = np.dstack([merged_group.mask * 255] * 3)
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode("utf-8")
    return {"id": new_id, "image": b64, "x": 0, "y": 0, "piece_ids": merged_group.piece_ids}


@router.post("/undo_merge")
async def undo_merge_endpoint():
    if not merge_history:
        items = []
        for it in canvas_items:
            img = np.dstack([it["group"].mask * 255] * 3)
            _, buf = cv2.imencode(".png", img)
            b64 = base64.b64encode(buf).decode("utf-8")
            items.append({"id": it["id"], "image": b64, "x": it["x"], "y": it["y"], "type": it.get("type", "group")})
        return {"items": items}
    new_item, old_items = merge_history.pop()
    canvas_items.remove(new_item)
    canvas_items.extend(old_items)
    items = []
    for it in canvas_items:
        img = np.dstack([it["group"].mask * 255] * 3)
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf).decode("utf-8")
        items.append({"id": it["id"], "image": b64, "x": it["x"], "y": it["y"], "type": it.get("type", "group")})
    return {"items": items}


@router.post("/suggest_match")
async def suggest_match_endpoint(request: Request):
    data = await request.json()
    try:
        piece_id = int(data.get("piece_id"))
        edge_index = int(data.get("edge_index"))
    except (TypeError, ValueError):
        return JSONResponse(status_code=400, content={"error": "invalid parameters"})
    id_to_idx = {it["id"]: i for i, it in enumerate(canvas_items)}
    if piece_id not in id_to_idx:
        return JSONResponse(status_code=400, content={"error": "invalid piece_id"})
    idx = id_to_idx[piece_id]
    pieces = [it["group"].features for it in canvas_items]
    if edge_index < 0 or edge_index >= len(pieces[idx].edges):
        return JSONResponse(status_code=400, content={"error": "edge_index out of range"})
    n = data.get("n", 5)
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 5
    matches = top_n_matches(pieces, n=n)
    results = []
    for op_idx, oe_idx, score in matches.get((idx, edge_index), []):
        results.append({"piece_id": canvas_items[op_idx]["id"], "edge_index": oe_idx, "score": score})
    state = {"piece_id": piece_id, "edge_index": edge_index}
    results = rank_with_policy(state, results)
    return {"matches": results}


@router.post("/submit_feedback")
async def submit_feedback_endpoint(request: Request):
    data = await request.json()
    if not all(k in data for k in ("state", "action", "reward")):
        return JSONResponse(status_code=400, content={"error": "missing fields"})
    entry = {"state": data["state"], "action": data["action"], "reward": data["reward"]}
    with open("feedback.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "ok"}
