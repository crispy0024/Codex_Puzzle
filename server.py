
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
    mask, result = remove_background(img)
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
    mask, result = remove_background(img)
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
    mask, _ = remove_background(img)
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
    mask, _ = remove_background(img)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
