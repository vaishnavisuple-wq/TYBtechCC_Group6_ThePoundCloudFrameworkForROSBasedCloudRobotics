# "Robot" client: JPEG-encodes image, simulates net latency, POSTs to the cloud,
# and draws server results. Run with:  python client/client.py --image data/sample.jpg

import os, io, time, argparse
import requests
import numpy as np
import cv2

def simulate_network_latency(payload_bytes: int,
                             uplink_mbps: float = 10.0,
                             downlink_mbps: float = 20.0,
                             rtt_ms: float = 15.0):
    """
    Simulated latency budget:
    - uplink transfer: payload / uplink_rate + RTT
    - downlink transfer: payload / downlink_rate + RTT
    Returns (uplink_s, downlink_s, total_net_s)
    """
    up = (payload_bytes * 8) / (uplink_mbps * 1e6) + (rtt_ms / 1000.0)
    down = (payload_bytes * 8) / (downlink_mbps * 1e6) + (rtt_ms / 1000.0)
    return up, down, up + down

def make_synthetic(path: str):
    """Create a simple synthetic 'defect' image if user has no data."""
    img = np.full((360, 480, 3), 220, np.uint8)
    cv2.circle(img, (120, 180), 35, (0, 0, 0), -1)
    cv2.rectangle(img, (260, 120), (330, 210), (0, 0, 0), -1)
    cv2.putText(img, "SYNTHETIC", (160, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (40,40,40), 2, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def draw_boxes(img, boxes, color=(0,0,255)):
    for (x,y,w,h) in boxes:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000/infer", help="Cloud inference endpoint")
    parser.add_argument("--image", default="data/sample.jpg", help="Input image")
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPEG quality (lower = smaller payload)")
    parser.add_argument("--uplink_mbps", type=float, default=10.0)
    parser.add_argument("--downlink_mbps", type=float, default=20.0)
    parser.add_argument("--rtt_ms", type=float, default=15.0)
    args = parser.parse_args()

    # Prepare an image
    if not os.path.exists(args.image):
        print("No image found; generating a synthetic sample at", args.image)
        make_synthetic(args.image)

    img = cv2.imread(args.image)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    payload = enc.tobytes()
    payload_kb = len(payload) / 1024.0

    up_s, down_s, net_s = simulate_network_latency(len(payload),
                                                   uplink_mbps=args.uplink_mbps,
                                                   downlink_mbps=args.downlink_mbps,
                                                   rtt_ms=args.rtt_ms)

    # Send to cloud
    t0 = time.perf_counter()
    resp = requests.post(args.server, files={"file": ("frame.jpg", payload, "image/jpeg")}, timeout=60)
    roundtrip_s = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    boxes = data.get("boxes", [])
    server_ms = data.get("server_ms", None)

    # Save visualization
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)
    vis = draw_boxes(img.copy(), boxes)
    out_path = os.path.join(out_dir, os.path.basename(args.image).rsplit(".",1)[0] + "_out.jpg")
    cv2.imwrite(out_path, vis)

    print("\n=== Cloud Offloading Report ===")
    print(f"Server endpoint      : {args.server}")
    print(f"Input file           : {args.image}")
    print(f"JPEG quality         : {args.jpeg_quality}")
    print(f"Payload size         : {payload_kb:.1f} KB")
    print(f"Sim uplink           : {up_s*1000:.1f} ms")
    print(f"Server compute (ms)  : {server_ms}")
    print(f"Sim downlink         : {down_s*1000:.1f} ms")
    print(f"Sim net total        : {net_s*1000:.1f} ms")
    print(f"Measured round-trip  : {roundtrip_s*1000:.1f} ms (HTTP request/response)")
    print(f"Detections           : {len(boxes)} boxes")
    print(f"Saved visualization  : {out_path}\n")

if __name__ == "__main__":
    main()

