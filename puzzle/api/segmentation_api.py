import base64
import json
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse

from ..segmentation import (
    remove_background,
    remove_background_canny,
    detect_piece_corners,
    select_four_corners,
    classify_piece_type,
    segment_pieces,
    segment_pieces_by_median,
    segment_pieces_metadata,
    extract_mask_contours,
    extract_clean_pieces,
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
    inv = cv2.bitwise_not(mask * 255)
    _, mask_buf = cv2.imencode(".png", inv)
    mask_b64 = base64.b64encode(mask_buf).decode("utf-8")
    return {"image": result_b64, "mask": mask_b64}


@router.post("/background_canny")
async def background_canny_endpoint(
    image: UploadFile = File(...),
    threshold_low: int | None = Form(None),
    threshold_high: int | None = Form(None),
    kernel_size: int | None = Form(None),
):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    mask, result, edges = remove_background_canny(
        img,
        lower_thresh=threshold_low,
        upper_thresh=threshold_high,
        kernel_size=kernel_size,
    )
    _, rbuf = cv2.imencode(".png", result)
    inv = cv2.bitwise_not(mask * 255)
    _, mbuf = cv2.imencode(".png", inv)
    _, ebuf = cv2.imencode(".png", edges)
    return {
        "image": base64.b64encode(rbuf).decode("utf-8"),
        "mask": base64.b64encode(mbuf).decode("utf-8"),
        "edges": base64.b64encode(ebuf).decode("utf-8"),
    }


@router.post("/segment_pieces")
async def segment_pieces_endpoint(
    image: UploadFile = File(...),
    threshold: int = Form(250),
    kernel_size: int = Form(3),
    method: str = Form("otsu"),
    min_area: int = Form(0),
    block_size: int | None = Form(None),
    C: int | None = Form(None),
    threshold1: int | None = Form(None),
    threshold2: int | None = Form(None),
    use_hull: bool = Form(False),
):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    pieces, num_contours = segment_pieces(
        img,
        min_area=min_area,
        thresh_val=threshold,
        kernel_size=kernel_size,
        method=method,
        use_hull=use_hull,
        block_size=block_size,
        C=C,
        threshold1=threshold1,
        threshold2=threshold2,
    )
    outputs = []
    for p in pieces:
        _, buf = cv2.imencode(".png", p)
        outputs.append(base64.b64encode(buf).decode("utf-8"))
    return {"pieces": outputs, "num_contours": num_contours}


@router.post("/extract_filtered_pieces")
async def extract_filtered_pieces_endpoint(
    image: UploadFile = File(...),
    threshold: int = Form(250),
    kernel_size: int = Form(3),
    min_area: int = Form(0),
):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    pieces = segment_pieces_by_median(
        img,
        thresh_val=threshold,
        kernel_size=kernel_size,
        min_area=min_area,
    )
    outputs = []
    for p in pieces:
        _, buf = cv2.imencode(".png", p)
        outputs.append(base64.b64encode(buf).decode("utf-8"))
    return {"pieces": outputs}


@router.post("/extract_then_remove")
async def extract_then_remove_endpoint(
    image: UploadFile = File(...),
    blur: int = Form(5),
    kernel_size: int = Form(3),
):
    """Extract pieces, clean them and run remove_background."""
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    pieces = extract_clean_pieces(img, blur=blur, kernel_size=kernel_size)
    outputs = []
    for p in pieces:
        mask, result = remove_background(p)
        _, rbuf = cv2.imencode(".png", result)
        inv = cv2.bitwise_not(mask * 255)
        _, mbuf = cv2.imencode(".png", inv)
        outputs.append(
            {
                "image": base64.b64encode(rbuf).decode("utf-8"),
                "mask": base64.b64encode(mbuf).decode("utf-8"),
            }
        )
    return {"pieces": outputs}


@router.post("/mask_contours")
async def mask_contours_endpoint(image: UploadFile = File(...)):
    """Return contour crops from a mask and the number of detected contours."""
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    pieces, num = extract_mask_contours(img)
    outputs = []
    for p in pieces:
        _, buf = cv2.imencode(".png", p)
        outputs.append(base64.b64encode(buf).decode("utf-8"))
    return {"contours": outputs, "num_contours": num}


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


@router.post("/detect_corners")
async def detect_corners_endpoint(image: UploadFile = File(...)):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    mask, _ = remove_background(img)
    corners = select_four_corners(detect_piece_corners(mask))
    return {"corners": corners.tolist()}


@router.post("/classify_piece")
async def classify_piece_endpoint(image: UploadFile = File(...)):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    mask, _ = remove_background(img)
    corners = select_four_corners(detect_piece_corners(mask))
    if len(corners) < 4:
        return {"type": "unknown"}
    piece_type = classify_piece_type(mask, corners)
    return {"type": piece_type}


@router.post("/edge_descriptors")
async def edge_descriptors_endpoint(image: UploadFile = File(...)):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})
    mask, _ = remove_background(img)
    corners = select_four_corners(detect_piece_corners(mask))
    descs = extract_edge_descriptors(img, mask, corners)
    metrics = []
    for d in descs:
        metrics.append(
            {
                "hist": d["hist"].tolist() if d["hist"] is not None else None,
                "hu": d["hu"].tolist() if d["hu"] is not None else None,
                "color_profile": (
                    d["color_profile"].tolist()
                    if d.get("color_profile") is not None
                    else None
                ),
            }
        )
    return {"metrics": metrics}


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
            return JSONResponse(
                status_code=400, content={"error": "Invalid descriptors"}
            )

        def _from_desc(d):
            return EdgeFeatures(
                edge_type=d.get("edge_type", "flat"),
                length=0.0,
                angle=0.0,
                hu_moments=np.array(d.get("hu")) if d.get("hu") is not None else None,
                color_hist=(
                    np.array(d.get("hist")) if d.get("hist") is not None else None
                ),
                color_profile=(
                    np.array(d.get("color_profile"))
                    if d.get("color_profile") is not None
                    else None
                ),
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
    if len(corners1) != 4:
        x, y, w, h = cv2.boundingRect(conts1[0])
        corners1 = np.array([[x, y + h - 1], [x, y], [x + w - 1, y + h - 1], [x + w - 1, y]], dtype=np.int32)
    if len(corners2) != 4:
        x, y, w, h = cv2.boundingRect(conts2[0])
        corners2 = np.array([[x, y + h - 1], [x, y], [x + w - 1, y + h - 1], [x + w - 1, y]], dtype=np.int32)

    conts1, _ = cv2.findContours(
        mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    conts2, _ = cv2.findContours(
        mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not conts1 or not conts2:
        return JSONResponse(
            status_code=400, content={"error": "Unable to find contours"}
        )
    cont1 = max(conts1, key=cv2.contourArea)
    cont2 = max(conts2, key=cv2.contourArea)

    labels1 = classify_edge_types(cont1, corners1)
    labels2 = classify_edge_types(cont2, corners2)
    descs1 = extract_edge_descriptors(img1, mask1, corners1)
    descs2 = extract_edge_descriptors(img2, mask2, corners2)

    if idx1 >= len(descs1) or idx2 >= len(descs2):
        return JSONResponse(
            status_code=400, content={"error": "Edge index out of range"}
        )

    def _make_edge(desc, label, pt1, pt2):
        length = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
        angle = float(np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])))
        return EdgeFeatures(
            edge_type=label,
            length=length,
            angle=angle,
            hu_moments=np.array(desc["hu"]) if desc["hu"] is not None else None,
            color_hist=np.array(desc["hist"]) if desc["hist"] is not None else None,
            color_profile=(
                np.array(desc["color_profile"])
                if desc["color_profile"] is not None
                else None
            ),
        )

    e1 = _make_edge(
        descs1[idx1], labels1[idx1], corners1[idx1], corners1[(idx1 + 1) % 4]
    )
    e2 = _make_edge(
        descs2[idx2], labels2[idx2], corners2[idx2], corners2[(idx2 + 1) % 4]
    )
    score = compatibility_score(e1, e2)
    return {"score": score}


@router.post("/save_pieces")
async def save_pieces_endpoint(request: Request):
    """Persist cropped pieces and optional labels to disk."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})
    pieces = payload.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        return JSONResponse(status_code=400, content={"error": "pieces must be a list"})

    import os
    import uuid
    from pathlib import Path

    save_dir = Path(os.environ.get("PIECE_SAVE_DIR", "saved_pieces"))
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_path = save_dir / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
        except Exception:
            metadata = []
    else:
        metadata = []

    saved = []
    for entry in pieces:
        if not isinstance(entry, dict) or "image" not in entry:
            continue
        b64_str = entry["image"]
        if b64_str.startswith("data:"):
            b64_str = b64_str.split(",", 1)[-1]
        try:
            img_bytes = base64.b64decode(b64_str)
        except Exception:
            continue
        fname = f"{uuid.uuid4().hex}.png"
        with open(save_dir / fname, "wb") as f:
            f.write(img_bytes)
        metadata.append({"file": fname, "label": entry.get("label")})
        saved.append(fname)

    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    return {"saved": saved}
