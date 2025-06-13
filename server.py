
import base64
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
)
from puzzle.features import extract_edge_descriptors

COLOR_MAP = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
}

app = Flask(__name__)
CORS(app)  # allow requests from the frontend

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
    kernel = _parse_int('kernel', 0)

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


@app.route('/segment_pieces', methods=['POST'])
def segment_pieces_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400

    pieces = segment_pieces(img, min_area=100)
    outputs = []
    for p in pieces:
        _, buf = cv2.imencode('.png', p)
        outputs.append(base64.b64encode(buf).decode('utf-8'))

    return jsonify({'pieces': outputs})


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
