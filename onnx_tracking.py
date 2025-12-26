from tools.infer_onnx import ONNX_Predict
from tools.tracker import KalmanSortTracker
import cv2
import time
import collections

# --------------------------------------------------
# Model & inference setup
# --------------------------------------------------
onnx_model = r"runs\export\1\model_decoded.onnx"  #<--- Path to your exported onnx model

providers = ["CPUExecutionProvider"]

predict = ONNX_Predict(
    onnx_model,
    providers=providers,
    use_letterbox=True   # Match training if resize was used
)

img_size = 640
conf = 0.55
iou = 0.30
max_det = 300

# --------------------------------------------------
# Class names & colors (example)
# --------------------------------------------------
CLASS_NAMES = {
    0: "ClassA",
    1: "ClassB",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
}

# --------------------------------------------------
# Video input
# --------------------------------------------------
cap = cv2.VideoCapture(r"test_video_path.mp4")

# Optional display scaling
max_w = 800

# --------------------------------------------------
# Tracker
# --------------------------------------------------
tracker = KalmanSortTracker(
    max_age=60,
    min_hits=5
)

# FPS smoothing
fps_history = collections.deque(maxlen=30)

# --------------------------------------------------
# Main loop
# --------------------------------------------------
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # ---------------------------
    # Inference (timed)
    # ---------------------------
    t0 = time.time()
    boxes, scores, classes = predict.infer_image(
        frame,
        img_size=img_size,
        conf=conf,
        iou=iou,
        max_det=max_det
    )
    infer_time = (time.time() - t0) * 1000

    # ---------------------------
    # Tracking
    # ---------------------------
    tracks = tracker.update(boxes, scores, classes)

    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        track_id = t["track_id"]
        cls_id = t["cls"]
        score = t["score"]

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"ID {track_id} | {CLASS_NAMES.get(cls_id, cls_id)} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    # ---------------------------
    # FPS display
    # ---------------------------
    dt = time.time() - start_time
    fps = 1.0 / dt if dt > 0 else 0.0
    fps_history.append(fps)
    fps_avg = sum(fps_history) / len(fps_history)

    cv2.putText(
        frame,
        f"{img_size}x{img_size} CPU  FPS: {fps_avg:5.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("ONNX Inference + Tracking", frame)
    print(f"FPS: {fps_avg:5.1f} | Inference: {infer_time:5.1f} ms")

    if cv2.waitKey(1) == 13:  # Enter
        break

cap.release()
cv2.destroyAllWindows()
