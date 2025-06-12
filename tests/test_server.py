import numpy as np
import cv2
import io
import json

import server


def test_remove_background_endpoint():
    app = server.app
    client = app.test_client()
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    _, buf = cv2.imencode('.png', img)
    response = client.post('/remove_background', data={'image': (io.BytesIO(buf.tobytes()), 'test.png')})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'image' in data
    assert len(data['image']) > 0
