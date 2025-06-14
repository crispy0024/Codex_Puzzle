import numpy as np
import cv2
import io
import json
import base64

import pathlib
import puzzle.api as server
from fastapi.testclient import TestClient


def test_remove_background_endpoint():
    app = server.app
    client = TestClient(app)
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    response = client.post(
        "/remove_background",
        data={
            "image": (io.BytesIO(buf.tobytes()), "test.png"),
            "threshold_low": "240",
            "threshold_high": "255",
            "kernel_size": "3",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "image" in data and "mask" in data
    assert len(data["image"]) > 0 and len(data["mask"]) > 0


def test_segment_pieces_endpoint():
    app = server.app
    client = TestClient(app)
    # Create simple image with two black squares on white background
    img = np.full((20, 40, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (8, 18), (0, 0, 0), -1)
    cv2.rectangle(img, (22, 2), (38, 18), (0, 0, 0), -1)
    _, buf = cv2.imencode(".png", img)
    response = client.post(
        "/segment_pieces", data={"image": (io.BytesIO(buf.tobytes()), "test.png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "pieces" in data and "num_contours" in data
    assert data["num_contours"] == 2
    assert len(data["pieces"]) == 2
    # verify min_area filtering
    response = client.post(
        "/segment_pieces",
        data={"image": (io.BytesIO(buf.tobytes()), "test.png"), "min_area": "400"},
    )
    data = response.json()
    assert len(data["pieces"]) == 0 and data["num_contours"] == 2
    # ensure use_hull parameter is accepted
    response = client.post(
        "/segment_pieces",
        data={"image": (io.BytesIO(buf.tobytes()), "test.png"), "use_hull": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["pieces"]) == 2


def test_adjust_image_endpoint():
    app = server.app
    client = TestClient(app)
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (8, 8), (0, 0, 0), -1)
    _, buf = cv2.imencode(".png", img)
    response = client.post(
        "/adjust_image",
        data={
            "image": (io.BytesIO(buf.tobytes()), "test.png"),
            "threshold": "100",
            "blur": "3",
            "color": "green",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "image" in data and len(data["image"]) > 0


def test_compare_edges_endpoint():
    app = server.app
    client = TestClient(app)

    img1 = np.full((50, 50, 3), 255, dtype=np.uint8)
    cv2.rectangle(img1, (5, 5), (45, 45), (0, 0, 0), -1)
    img2 = np.full((50, 50, 3), 255, dtype=np.uint8)
    cv2.rectangle(img2, (5, 5), (45, 45), (0, 0, 0), -1)
    _, buf1 = cv2.imencode(".png", img1)
    _, buf2 = cv2.imencode(".png", img2)

    response = client.post(
        "/compare_edges",
        data={
            "image1": (io.BytesIO(buf1.tobytes()), "p1.png"),
            "image2": (io.BytesIO(buf2.tobytes()), "p2.png"),
            "edge1": "0",
            "edge2": "0",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert isinstance(data["score"], (int, float))


def test_extract_filtered_pieces_endpoint():
    app = server.app
    client = TestClient(app)
    img = np.full((40, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (11, 11), (0, 0, 0), -1)
    cv2.rectangle(img, (22, 2), (31, 11), (0, 0, 0), -1)
    cv2.rectangle(img, (42, 2), (61, 21), (0, 0, 0), -1)
    _, buf = cv2.imencode(".png", img)
    response = client.post(
        "/extract_filtered_pieces", data={"image": (io.BytesIO(buf.tobytes()), "t.png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "pieces" in data and len(data["pieces"]) == 2
    response = client.post(
        "/extract_filtered_pieces",
        data={"image": (io.BytesIO(buf.tobytes()), "t.png"), "min_area": "400"},
    )
    data = response.json()
    assert "pieces" in data and len(data["pieces"]) == 0


def _dummy_features(size):
    h, w = size
    contour = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    edges = [
        server.EdgeFeatures(
            edge_type="flat",
            length=float(w),
            angle=0.0,
            hu_moments=None,
            color_hist=None,
            color_profile=None,
        )
        for _ in range(4)
    ]
    return server.PieceFeatures(
        contour=contour, area=float(h * w), bbox=(0, 0, w, h), edges=edges
    )


def test_merge_and_undo_endpoints():
    app = server.app
    client = TestClient(app)

    server.canvas_items.clear()
    server.merge_history.clear()

    mask = np.ones((10, 10), dtype=np.uint8)
    pf1 = _dummy_features((10, 10))
    pf2 = _dummy_features((10, 10))
    g1 = server.PieceGroup([1], {1: (0, 0)}, mask, pf1, {1: mask}, {1: pf1})
    g2 = server.PieceGroup([2], {2: (0, 0)}, mask, pf2, {2: mask}, {2: pf2})

    server.canvas_items.extend(
        [
            {"id": 1, "group": g1, "x": 0, "y": 0, "type": "piece"},
            {"id": 2, "group": g2, "x": 10, "y": 0, "type": "piece"},
        ]
    )

    resp = client.post("/merge_pieces", json={"piece_ids": [1, 2]})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data and "image" in data
    assert len(server.canvas_items) == 1
    assert server.canvas_items[0]["group"].piece_ids == [1, 2]

    resp2 = client.post("/undo_merge")
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert "items" in data2 and len(data2["items"]) == 2
    ids = sorted([it["id"] for it in server.canvas_items])
    assert ids == [1, 2]


def _edge(edge_type, hu, color):
    return server.EdgeFeatures(
        edge_type=edge_type,
        length=1.0,
        angle=0.0,
        hu_moments=np.array(hu, dtype=np.float32),
        color_hist=None,
        color_profile=np.array(color, dtype=np.float32),
    )


def _piece(edges):
    return server.PieceFeatures(
        contour=np.zeros((1, 2), dtype=np.int32),
        area=0.0,
        bbox=(0, 0, 1, 1),
        edges=edges,
    )


def test_suggest_match_endpoint():
    app = server.app
    client = TestClient(app)

    server.canvas_items.clear()
    server.merge_history.clear()

    mask = np.ones((5, 5), dtype=np.uint8)

    tab_good = _edge("tab", [0.1] * 7, [1, 2, 3])
    tab_other = _edge("tab", [2.0] * 7, [5, 5, 5])
    piece_a = _piece(
        [
            tab_good,
            _edge("flat", [0] * 7, [0, 0, 0]),
            tab_other,
            _edge("flat", [0] * 7, [0, 0, 0]),
        ]
    )

    hole_match = _edge("hole", [0.1] * 7, [1, 2, 3])
    hole_other = _edge("hole", [3.0] * 7, [10, 10, 10])
    piece_b = _piece(
        [
            hole_match,
            hole_other,
            _edge("tab", [0] * 7, [0, 0, 0]),
            _edge("flat", [0] * 7, [0, 0, 0]),
        ]
    )

    g1 = server.PieceGroup([1], {1: (0, 0)}, mask, piece_a, {1: mask}, {1: piece_a})
    g2 = server.PieceGroup([2], {2: (0, 0)}, mask, piece_b, {2: mask}, {2: piece_b})

    server.canvas_items.extend(
        [
            {"id": 1, "group": g1, "x": 0, "y": 0, "type": "piece"},
            {"id": 2, "group": g2, "x": 10, "y": 0, "type": "piece"},
        ]
    )

    resp = client.post("/suggest_match", json={"piece_id": 1, "edge_index": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert "matches" in data and len(data["matches"]) > 0
    m = data["matches"][0]
    assert m["piece_id"] == 2 and m["edge_index"] == 0


def test_submit_feedback_endpoint():
    app = server.app
    client = TestClient(app)

    if pathlib.Path("feedback.jsonl").exists():
        pathlib.Path("feedback.jsonl").unlink()

    resp = client.post(
        "/submit_feedback",
        json={"state": {"a": 1}, "action": "test", "reward": 1},
    )
    assert resp.status_code == 200

    # Ensure feedback appended
    with open("feedback.jsonl") as f:
        lines = f.readlines()
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    assert rec["action"] == "test" and rec["reward"] == 1


def test_additional_segmentation_endpoints():
    app = server.app
    client = TestClient(app)

    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (18, 18), (0, 0, 0), -1)
    _, buf = cv2.imencode(".png", img)

    resp = client.post(
        "/detect_corners", data={"image": (io.BytesIO(buf.tobytes()), "p.png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "corners" in data and len(data["corners"]) == 4

    resp = client.post(
        "/classify_piece", data={"image": (io.BytesIO(buf.tobytes()), "p.png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "type" in data

    resp = client.post(
        "/edge_descriptors", data={"image": (io.BytesIO(buf.tobytes()), "p.png")}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data and len(data["metrics"]) == 4


def test_save_pieces_endpoint(tmp_path, monkeypatch):
    app = server.app
    client = TestClient(app)

    monkeypatch.setenv("PIECE_SAVE_DIR", str(tmp_path))
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (8, 8), (0, 0, 0), -1)
    _, buf = cv2.imencode(".png", img)
    b64 = base64.b64encode(buf).decode("utf-8")

    response = client.post(
        "/save_pieces",
        json={"pieces": [{"image": b64, "label": "test"}]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "saved" in data and len(data["saved"]) == 1

    saved_file = tmp_path / data["saved"][0]
    assert saved_file.exists()
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta[0]["label"] == "test"
