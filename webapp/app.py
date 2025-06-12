import base64
import cv2
import numpy as np
from flask import Flask, request, render_template
from puzzle import (
    remove_background,
    detect_piece_corners,
    select_four_corners,
    classify_piece_type,
)

app = Flask(__name__)


def _read_image(file_storage):
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_image(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_data = None
    if request.method == 'POST' and 'piece' in request.files:
        file = request.files['piece']
        if file.filename:
            img = _read_image(file)
            mask, segmented = remove_background(img)
            corners = detect_piece_corners(mask)
            corners = select_four_corners(corners)
            result = classify_piece_type(mask, corners)
            img_data = _encode_image(segmented)
    return render_template('index.html', result=result, img_data=img_data)


if __name__ == '__main__':
    app.run(debug=True)
