# FastAPI "cloud" server: receives an image, does simple CV, returns boxes + timing.
# Run with:  uvicorn server.cloud_server:app --reload

import time
from typing import List, Dict
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Cloud Inference Server (Minimal)")

def simple_defect_detector(img_bgr: np.ndarray) -> List[List[int]]:
    """
    Very simple 'defect' finder:
    - convert to gray
    - blur + Otsu threshold
    - find external contours
    - return bounding boxes [x, y, w, h]
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # adaptive/otsu threshold for robustness
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[List[int]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # filter tiny noise
        if w*h >= 50:
            boxes.append([int(x), int(y), int(w), int(h)])
    return boxes

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/infer"], "docs": "/docs"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # Read and decode the uploaded image
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    t0 = time.perf_counter()
    boxes = simple_defect_detector(img)
    server_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "num_boxes": len(boxes),
        "boxes": boxes,           # list of [x, y, w, h]
        "server_ms": round(server_ms, 3)
    }

