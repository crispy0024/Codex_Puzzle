import io
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify

from puzzle.segmentation import remove_background

app = Flask(__name__)

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
    _mask, result = remove_background(img)
    _, buf = cv2.imencode('.png', result)
    b64 = base64.b64encode(buf).decode('utf-8')
    return jsonify({'image': b64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
