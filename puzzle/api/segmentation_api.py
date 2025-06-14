import base64
import json
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

from ..segmentation import (
    remove_background,
    detect_piece_corners,
    select_four_corners,
    classify_piece_type,
    segment_pieces,
    segment_pieces_by_median,
    segment_pieces_metadata,
    PuzzlePiece,
)
from ..features import (
    extract_edge_descriptors,
    classify_edge_types,
    EdgeFeatures,
    PieceFeatures,
)
from ..scoring import compatibility_score

router = APIRouter()


@router.post("/remove_background")
async def remove_background_endpoint(
    image: UploadFile = File(...),
    threshold_low: int | None = Form(None),
    threshold_high: int | None = Form(None),
    kernel_size: int | None = Form(None),
):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    mask, result = remove_background(
        img,
        lower_thresh=threshold_low,
        upper_thresh=threshold_high,
        kernel_size=kernel_size,
    )
    _, buf = cv2.imencode(".png", result)
    result_b64 = base64.b64encode(buf).decode("utf-8")
    _, mask_buf = cv2.imencode(".png", mask * 255)
    mask_b64 = base64.b64encode(mask_buf).decode("utf-8")
    return {"image": result_b64, "mask": mask_b64}


@router.post("/segment_pieces")
async def segment_pieces_endpoint(
    image: UploadFile = File(...),
    threshold: int = Form(250),
    kernel_size: int = Form(3),
):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    pieces = segment_pieces(
        img,
        min_area=100,
        thresh_val=threshold,
        kernel_size=kernel_size,
    )
    outputs = []
    for p in pieces:
        _, buf = cv2.imencode(".png", p)
        outputs.append(base64.b64encode(buf).decode("utf-8"))
    return {"pieces": outputs}


@router.post("/extract_filtered_pieces")
async def extract_filtered_pieces_endpoint(
    image: UploadFile = File(...),
    threshold: int = Form(250),
    kernel_size: int = Form(3),
):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    pieces = segment_pieces_by_median(img, thresh_val=threshold, kernel_size=kernel_size)
    outputs = []
    for p in pieces:
        _, buf = cv2.imencode(".png", p)
        outputs.append(base64.b64encode(buf).decode("utf-8"))
    return {"pieces": outputs}


@router.post("/adjust_image")
async def adjust_image_endpoint(
    image: UploadFile = File(...),
    threshold: int = Form(128),
    blur: int = Form(0),
    color: str = Form("red"),
):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    COLOR_MAP = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}
    color_val = COLOR_MAP.get(color.lower(), (0, 0, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur > 0 and blur % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    color_layer = np.zeros_like(img)
    color_layer[thresh == 255] = color_val
    output = cv2.addWeighted(img, 0.7, color_layer, 0.3, 0)
    _, buf = cv2.imencode(".png", output)
    b64 = base64.b64encode(buf).decode("utf-8")
    return {"image": b64}


@router.post("/compare_edges")
async def compare_edges_endpoint(request: Request):
    form = await request.form()
    desc1_json = form.get("desc1")
    desc2_json = form.get("desc2")
    if desc1_json and desc2_json:
        try:
            d1 = json.loads(desc1_json)
            d2 = json.loads(desc2_json)
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid descriptors"})

        def _from_desc(d):
            return EdgeFeatures(
                edge_type=d.get("edge_type", "flat"),
                length=0.0,
                angle=0.0,
                hu_moments=np.array(d.get("hu")) if d.get("hu") is not None else None,
                color_hist=np.array(d.get("hist")) if d.get("hist") is not None else None,
                color_profile=np.array(d.get("color_profile")) if d.get("color_profile") is not None else None,
            )

        edge1 = _from_desc(d1)
        edge2 = _from_desc(d2)
        score = compatibility_score(edge1, edge2)
        return {"score": score}

    if "image1" not in form or "image2" not in form:
        return JSONResponse(status_code=400, content={"error": "Two images required"})

    file1: UploadFile = form["image1"]
    file2: UploadFile = form["image2"]
    img1 = cv2.imdecode(np.frombuffer(await file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(await file2.read(), np.uint8), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    def _pint(name, default):
        try:
            return int(form.get(name, default))
        except (TypeError, ValueError):
            return default

    idx1 = _pint("edge1", 0)
    idx2 = _pint("edge2", 0)

    mask1, _ = remove_background(img1)
    mask2, _ = remove_background(img2)
    corners1 = select_four_corners(detect_piece_corners(mask1))
    corners2 = select_four_corners(detect_piece_corners(mask2))

    conts1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts1 or not conts2:
        return JSONResponse(status_code=400, content={"error": "Unable to find contours"})
    cont1 = max(conts1, key=cv2.contourArea)
    cont2 = max(conts2, key=cv2.contourArea)

    labels1 = classify_edge_types(cont1, corners1)
    labels2 = classify_edge_types(cont2, corners2)
    descs1 = extract_edge_descriptors(img1, mask1, corners1)
    descs2 = extract_edge_descriptors(img2, mask2, corners2)

    if idx1 >= len(descs1) or idx2 >= len(descs2):
        return JSONResponse(status_code=400, content={"error": "Edge index out of range"})

    def _make_edge(desc, label, pt1, pt2):
        length = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
        angle = float(np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])))
        return EdgeFeatures(
            edge_type=label,
            length=length,
            angle=angle,
            hu_moments=np.array(desc["hu"]) if desc["hu"] is not None else None,
            color_hist=np.array(desc["hist"]) if desc["hist"] is not None else None,
            color_profile=np.array(desc["color_profile"]) if desc["color_profile"] is not None else None,
        )

    e1 = _make_edge(descs1[idx1], labels1[idx1], corners1[idx1], corners1[(idx1 + 1) % 4])
    e2 = _make_edge(descs2[idx2], labels2[idx2], corners2[idx2], corners2[(idx2 + 1) % 4])
    score = compatibility_score(e1, e2)
    return {"score": score}
