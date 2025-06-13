from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from puzzle.segmentation import remove_background

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/segment")
async def segment(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        data = await file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        mask, piece = remove_background(img)
        _, buf = cv2.imencode(".png", piece)
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        results.append({"name": file.filename, "pieces": [b64]})
    return {"results": results}
