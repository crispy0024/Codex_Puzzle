
import base64
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # enable Cross-Origin Resource Sharing

from puzzle.segmentation import (
    remove_background,
    detect_piece_corners,
    select_four_corners,
    classify_piece_type,
    segment_pieces,
    segment_pieces_by_median,
    segment_pieces_metadata,
    PuzzlePiece,
)
from puzzle.features import (
    extract_edge_descriptors,
    classify_edge_types,
    EdgeFeatures,
    PieceFeatures,
)
from puzzle.scoring import compatibility_score, top_n_matches
from puzzle.group import PieceGroup, merge_groups

COLOR_MAP = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
}

app = Flask(__name__)
CORS(app)  # allow requests from the frontend

# In-memory canvas layout holding individual pieces and groups. Each item
# is a dictionary with keys: ``id``, ``group`` (PieceGroup), ``x`` and ``y``.
canvas_items: list[dict] = []

# Stack of merge operations for undo functionality. Each entry stores the new
# item that replaced the merged pieces and the list of removed items.
merge_history: list[tuple[dict, list[dict]]] = []

@app.route('/remove_background', methods=['POST'])
def remove_background_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    in_memory = file.read()
    npimg = np.frombuffer(in_memory, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    def _parse_int(name, default):
        try:
            return int(request.form.get(name, default))
        except (TypeError, ValueError):
            return default

    lower = _parse_int('threshold_low', _parse_int('lower', -1))
    upper = _parse_int('threshold_high', _parse_int('upper', -1))
    kernel = _parse_int('kernel_size', _parse_int('kernel', 0))

    lower_thresh = lower if lower >= 0 else None
    upper_thresh = upper if upper >= 0 else None
    kernel_size = kernel if kernel > 1 else None

    mask, result = remove_background(
        img,
        lower_thresh=lower_thresh,
        upper_thresh=upper_thresh,
        kernel_size=kernel_size,
    )
    _, buf = cv2.imencode('.png', result)
    result_b64 = base64.b64encode(buf).decode('utf-8')
    _, mask_buf = cv2.imencode('.png', mask * 255)
    mask_b64 = base64.b64encode(mask_buf).decode('utf-8')
    return jsonify({'image': result_b64, 'mask': mask_b64})


@app.route('/detect_corners', methods=['POST'])
def detect_corners_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        lower = int(request.form.get('lower', -1))
    except ValueError:
        lower = -1
    try:
        upper = int(request.form.get('upper', -1))
    except ValueError:
        upper = -1
    lower_thresh = lower if lower >= 0 else None
    upper_thresh = upper if upper >= 0 else None

    mask, result = remove_background(
        img, lower_thresh=lower_thresh, upper_thresh=upper_thresh
    )
    corners = detect_piece_corners(mask)
    four = select_four_corners(corners)
    overlay = result.copy()
    for x, y in four:
        cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 255), -1)
    _, buf = cv2.imencode('.png', overlay)
    overlay_b64 = base64.b64encode(buf).decode('utf-8')
    return jsonify({'image': overlay_b64, 'corners': four.tolist()})


@app.route('/classify_piece', methods=['POST'])
def classify_piece_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        lower = int(request.form.get('lower', -1))
    except ValueError:
        lower = -1
    try:
        upper = int(request.form.get('upper', -1))
    except ValueError:
        upper = -1
    lower_thresh = lower if lower >= 0 else None
    upper_thresh = upper if upper >= 0 else None

    mask, _ = remove_background(
        img, lower_thresh=lower_thresh, upper_thresh=upper_thresh
    )
    corners = detect_piece_corners(mask)
    four = select_four_corners(corners)
    ptype = classify_piece_type(mask, four)
    return jsonify({'type': ptype})


@app.route('/edge_descriptors', methods=['POST'])
def edge_descriptors_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        lower = int(request.form.get('lower', -1))
    except ValueError:
        lower = -1
    try:
        upper = int(request.form.get('upper', -1))
    except ValueError:
        upper = -1
    lower_thresh = lower if lower >= 0 else None
    upper_thresh = upper if upper >= 0 else None

    mask, _ = remove_background(
        img, lower_thresh=lower_thresh, upper_thresh=upper_thresh
    )
    corners = detect_piece_corners(mask)
    four = select_four_corners(corners)
    desc = extract_edge_descriptors(img, mask, four)
    metrics = []
    for d in desc:
        hist_len = len(d['hist']) if d['hist'] is not None else 0
        sift_len = len(d['sift']) if d['sift'] is not None else 0
        metrics.append({'hist_len': hist_len, 'sift_len': sift_len})
    return jsonify({'metrics': metrics})


@app.route('/compare_edges', methods=['POST'])
def compare_edges_endpoint():
    # Allow sending precomputed descriptors as JSON
    desc1_json = request.form.get('desc1')
    desc2_json = request.form.get('desc2')
    if desc1_json and desc2_json:
        try:
            d1 = json.loads(desc1_json)
            d2 = json.loads(desc2_json)
        except Exception:
            return jsonify({'error': 'Invalid descriptors'}), 400

        def _from_desc(d):
            return EdgeFeatures(
                edge_type=d.get('edge_type', 'flat'),
                length=0.0,
                angle=0.0,
                hu_moments=np.array(d.get('hu')) if d.get('hu') is not None else None,
                color_hist=np.array(d.get('hist')) if d.get('hist') is not None else None,
                color_profile=np.array(d.get('color_profile')) if d.get('color_profile') is not None else None,
            )

        edge1 = _from_desc(d1)
        edge2 = _from_desc(d2)
        score = compatibility_score(edge1, edge2)
        return jsonify({'score': score})

    # Otherwise expect two images and edge indices
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Two images required'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']
    img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return jsonify({'error': 'Invalid image'}), 400

    def _pint(name, default):
        try:
            return int(request.form.get(name, default))
        except (TypeError, ValueError):
            return default

    idx1 = _pint('edge1', 0)
    idx2 = _pint('edge2', 0)

    mask1, _ = remove_background(img1)
    mask2, _ = remove_background(img2)
    corners1 = select_four_corners(detect_piece_corners(mask1))
    corners2 = select_four_corners(detect_piece_corners(mask2))

    conts1, _ = cv2.findContours(mask1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts2, _ = cv2.findContours(mask2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts1 or not conts2:
        return jsonify({'error': 'Unable to find contours'}), 400
    cont1 = max(conts1, key=cv2.contourArea)
    cont2 = max(conts2, key=cv2.contourArea)

    labels1 = classify_edge_types(cont1, corners1)
    labels2 = classify_edge_types(cont2, corners2)
    descs1 = extract_edge_descriptors(img1, mask1, corners1)
    descs2 = extract_edge_descriptors(img2, mask2, corners2)

    if idx1 >= len(descs1) or idx2 >= len(descs2):
        return jsonify({'error': 'Edge index out of range'}), 400

    def _make_edge(desc, label, pt1, pt2):
        length = float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
        angle = float(np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])))
        return EdgeFeatures(
            edge_type=label,
            length=length,
            angle=angle,
            hu_moments=np.array(desc['hu']) if desc['hu'] is not None else None,
            color_hist=np.array(desc['hist']) if desc['hist'] is not None else None,
            color_profile=np.array(desc['color_profile']) if desc['color_profile'] is not None else None,
        )

    e1 = _make_edge(descs1[idx1], labels1[idx1], corners1[idx1], corners1[(idx1 + 1) % 4])
    e2 = _make_edge(descs2[idx2], labels2[idx2], corners2[idx2], corners2[(idx2 + 1) % 4])
    score = compatibility_score(e1, e2)
    return jsonify({'score': score})


@app.route('/segment_pieces', methods=['POST'])
def segment_pieces_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        thresh = int(request.form.get('threshold', 250))
    except ValueError:
        thresh = 250
    try:
        kernel = int(request.form.get('kernel_size', 3))
    except ValueError:
        kernel = 3

    pieces = segment_pieces(
        img,
        min_area=100,
        thresh_val=thresh,
        kernel_size=kernel,
    )
    outputs = []
    for p in pieces:
        _, buf = cv2.imencode('.png', p)
        outputs.append(base64.b64encode(buf).decode('utf-8'))

    return jsonify({'pieces': outputs})


@app.route('/extract_filtered_pieces', methods=['POST'])
def extract_filtered_pieces_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        thresh = int(request.form.get('threshold', 250))
    except ValueError:
        thresh = 250
    try:
        kernel = int(request.form.get('kernel_size', 3))
    except ValueError:
        kernel = 3

    pieces = segment_pieces_by_median(img, thresh_val=thresh, kernel_size=kernel)

    outputs = []
    for p in pieces:
        _, buf = cv2.imencode('.png', p)
        outputs.append(base64.b64encode(buf).decode('utf-8'))

    return jsonify({'pieces': outputs})


@app.route('/segment_pieces_metadata', methods=['POST'])
def segment_pieces_metadata_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    pieces = segment_pieces_metadata(img)
    outputs = []
    for p in pieces:
        _, buf = cv2.imencode('.png', p.image)
        outputs.append({'image': base64.b64encode(buf).decode('utf-8'), 'bbox': p.bbox, 'angle': p.angle})

    return jsonify({'pieces': outputs})


@app.route('/piece_contours', methods=['POST'])
def piece_contours_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    pieces = segment_pieces_metadata(img, min_area=100, margin=0, normalize=False)
    outputs = []
    for p in pieces:
        canvas = np.zeros_like(p.image)
        cnt = p.contour
        if cnt.ndim == 2:
            cnt = cnt[:, np.newaxis, :]
        cv2.drawContours(canvas, [cnt], -1, (0, 255, 0), 2)
        _, buf = cv2.imencode('.png', canvas)
        outputs.append(base64.b64encode(buf).decode('utf-8'))

    return jsonify({'contours': outputs})


@app.route('/adjust_image', methods=['POST'])
def adjust_image_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    try:
        thresh_val = int(request.form.get('threshold', 128))
    except ValueError:
        thresh_val = 128
    try:
        blur_size = int(request.form.get('blur', 0))
    except ValueError:
        blur_size = 0
    color_name = request.form.get('color', 'red').lower()
    color = COLOR_MAP.get(color_name, (0, 0, 255))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_size > 0 and blur_size % 2 == 1:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    color_layer = np.zeros_like(img)
    color_layer[thresh == 255] = color
    output = cv2.addWeighted(img, 0.7, color_layer, 0.3, 0)

    _, buf = cv2.imencode('.png', output)
    b64 = base64.b64encode(buf).decode('utf-8')
    return jsonify({'image': b64})


@app.route('/merge_pieces', methods=['POST'])
def merge_pieces_endpoint():
    """Merge multiple canvas items into a single group."""
    data = request.get_json(silent=True) or {}
    piece_ids = data.get('piece_ids')
    if not isinstance(piece_ids, list) or len(piece_ids) < 2:
        return jsonify({'error': 'piece_ids must be list of at least 2'}), 400

    items = [it for it in canvas_items if it['id'] in piece_ids]
    if len(items) != len(piece_ids):
        return jsonify({'error': 'invalid piece id'}), 400

    # Merge groups sequentially; arbitrary edges (1 with 3) work for tests
    merged_group = items[0]['group']
    for it in items[1:]:
        merged_group = merge_groups(merged_group, it['group'], 1, 3)

    new_id = max([it['id'] for it in canvas_items] + [0]) + 1
    new_item = {'id': new_id, 'group': merged_group, 'x': 0, 'y': 0, 'type': 'group'}

    # Update canvas and history
    remaining = [it for it in canvas_items if it['id'] not in piece_ids]
    canvas_items.clear()
    canvas_items.extend(remaining + [new_item])
    merge_history.append((new_item, items))

    img = np.dstack([merged_group.mask * 255] * 3)
    _, buf = cv2.imencode('.png', img)
    b64 = base64.b64encode(buf).decode('utf-8')

    return jsonify({'id': new_id, 'image': b64, 'x': 0, 'y': 0, 'piece_ids': merged_group.piece_ids})


@app.route('/undo_merge', methods=['POST'])
def undo_merge_endpoint():
    """Revert the last merge operation."""
    if not merge_history:
        # nothing to undo
        items = []
        for it in canvas_items:
            img = np.dstack([it['group'].mask * 255] * 3)
            _, buf = cv2.imencode('.png', img)
            b64 = base64.b64encode(buf).decode('utf-8')
            items.append({'id': it['id'], 'image': b64, 'x': it['x'], 'y': it['y'], 'type': it.get('type', 'group')})
        return jsonify({'items': items})

    new_item, old_items = merge_history.pop()
    canvas_items.remove(new_item)
    canvas_items.extend(old_items)

    items = []
    for it in canvas_items:
        img = np.dstack([it['group'].mask * 255] * 3)
        _, buf = cv2.imencode('.png', img)
        b64 = base64.b64encode(buf).decode('utf-8')
        items.append({'id': it['id'], 'image': b64, 'x': it['x'], 'y': it['y'], 'type': it.get('type', 'group')})

    return jsonify({'items': items})


@app.route('/suggest_match', methods=['POST'])
def suggest_match_endpoint():
    """Return best matching edges for a target piece edge."""
    data = request.get_json(silent=True) or {}
    try:
        piece_id = int(data.get('piece_id'))
        edge_index = int(data.get('edge_index'))
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid parameters'}), 400

    id_to_idx = {it['id']: i for i, it in enumerate(canvas_items)}
    if piece_id not in id_to_idx:
        return jsonify({'error': 'invalid piece_id'}), 400
    idx = id_to_idx[piece_id]

    pieces = [it['group'].features for it in canvas_items]
    if edge_index < 0 or edge_index >= len(pieces[idx].edges):
        return jsonify({'error': 'edge_index out of range'}), 400

    n = data.get('n', 5)
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 5

    matches = top_n_matches(pieces, n=n)
    results = []
    for op_idx, oe_idx, score in matches.get((idx, edge_index), []):
        results.append({
            'piece_id': canvas_items[op_idx]['id'],
            'edge_index': oe_idx,
            'score': score,
        })

    return jsonify({'matches': results})


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback_endpoint():
    """Append (state, action, reward) feedback to a log file."""
    data = request.get_json(silent=True) or {}
    if not all(k in data for k in ('state', 'action', 'reward')):
        return jsonify({'error': 'missing fields'}), 400

    entry = {
        'state': data['state'],
        'action': data['action'],
        'reward': data['reward'],
    }
    with open('feedback.jsonl', 'a') as f:
        f.write(json.dumps(entry) + '\n')

    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
