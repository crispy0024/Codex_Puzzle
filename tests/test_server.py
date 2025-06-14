import numpy as np
import cv2
import io
import json

import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("server", pathlib.Path(__file__).resolve().parents[1] / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_remove_background_endpoint():
    app = server.app
    client = app.test_client()
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)
    response = client.post(
        '/remove_background',
        data={
            'image': (io.BytesIO(buf.tobytes()), 'test.png'),
            'threshold_low': '240',
            'threshold_high': '255',
            'kernel_size': '3',
        },
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'image' in data and 'mask' in data
    assert len(data['image']) > 0 and len(data['mask']) > 0


def test_segment_pieces_endpoint():
    app = server.app
    client = app.test_client()
    # Create simple image with two black squares on white background
    img = np.full((20, 40, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (8, 18), (0, 0, 0), -1)
    cv2.rectangle(img, (22, 2), (38, 18), (0, 0, 0), -1)
    _, buf = cv2.imencode('.png', img)
    response = client.post('/segment_pieces', data={'image': (io.BytesIO(buf.tobytes()), 'test.png')})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'pieces' in data
    assert len(data['pieces']) == 2


def test_adjust_image_endpoint():
    app = server.app
    client = app.test_client()
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (8, 8), (0, 0, 0), -1)
    _, buf = cv2.imencode('.png', img)
    response = client.post(
        '/adjust_image',
        data={
            'image': (io.BytesIO(buf.tobytes()), 'test.png'),
            'threshold': '100',
            'blur': '3',
            'color': 'green',
        },
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'image' in data and len(data['image']) > 0


def test_compare_edges_endpoint():
    app = server.app
    client = app.test_client()

    img1 = np.full((50, 50, 3), 255, dtype=np.uint8)
    cv2.rectangle(img1, (5, 5), (45, 45), (0, 0, 0), -1)
    img2 = np.full((50, 50, 3), 255, dtype=np.uint8)
    cv2.rectangle(img2, (5, 5), (45, 45), (0, 0, 0), -1)
    _, buf1 = cv2.imencode('.png', img1)
    _, buf2 = cv2.imencode('.png', img2)

    response = client.post(
        '/compare_edges',
        data={
            'image1': (io.BytesIO(buf1.tobytes()), 'p1.png'),
            'image2': (io.BytesIO(buf2.tobytes()), 'p2.png'),
            'edge1': '0',
            'edge2': '0',
        },
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'score' in data
    assert isinstance(data['score'], (int, float))


def test_extract_filtered_pieces_endpoint():
    app = server.app
    client = app.test_client()
    img = np.full((40, 80, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (11, 11), (0, 0, 0), -1)
    cv2.rectangle(img, (22, 2), (31, 11), (0, 0, 0), -1)
    cv2.rectangle(img, (42, 2), (61, 21), (0, 0, 0), -1)
    _, buf = cv2.imencode('.png', img)
    response = client.post('/extract_filtered_pieces', data={'image': (io.BytesIO(buf.tobytes()), 't.png')})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'pieces' in data and len(data['pieces']) == 2


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
    return server.PieceFeatures(contour=contour, area=float(h * w), bbox=(0, 0, w, h), edges=edges)


def test_merge_and_undo_endpoints():
    app = server.app
    client = app.test_client()

    server.canvas_items.clear()
    server.merge_history.clear()

    mask = np.ones((10, 10), dtype=np.uint8)
    pf1 = _dummy_features((10, 10))
    pf2 = _dummy_features((10, 10))
    g1 = server.PieceGroup([1], {1: (0, 0)}, mask, pf1, {1: mask}, {1: pf1})
    g2 = server.PieceGroup([2], {2: (0, 0)}, mask, pf2, {2: mask}, {2: pf2})

    server.canvas_items.extend([
        {'id': 1, 'group': g1, 'x': 0, 'y': 0, 'type': 'piece'},
        {'id': 2, 'group': g2, 'x': 10, 'y': 0, 'type': 'piece'},
    ])

    resp = client.post('/merge_pieces', json={'piece_ids': [1, 2]})
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert 'id' in data and 'image' in data
    assert len(server.canvas_items) == 1
    assert server.canvas_items[0]['group'].piece_ids == [1, 2]

    resp2 = client.post('/undo_merge')
    assert resp2.status_code == 200
    data2 = json.loads(resp2.data)
    assert 'items' in data2 and len(data2['items']) == 2
    ids = sorted([it['id'] for it in server.canvas_items])
    assert ids == [1, 2]
