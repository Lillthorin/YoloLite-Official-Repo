import numpy as np
from dataclasses import dataclass

# -------------------------
# Hjälpfunktioner för bbox
# -------------------------

def xyxy_to_cxsysr(bbox):
    """
    bbox: [x1, y1, x2, y2]
    return: [cx, cy, s, r]
        cx, cy: center
        s: area (w*h)
        r: aspect ratio (w/h)
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    s = w * h
    r = w / (h + 1e-6)
    return np.array([cx, cy, s, r], dtype=np.float32)


def cxsysr_to_xyxy(x):
    """
    x: [cx, cy, s, r] eller state-vektor där x[0:4] är cx,cy,s,r
    return: [x1, y1, x2, y2]
    """
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    w = np.sqrt(s * r)
    h = s / (w + 1e-6)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou_xyxy(boxA, boxB):
    """
    boxA: [N,4], boxB: [M,4]
    return: IoU matrix [N,M]
    """
    if boxA.size == 0 or boxB.size == 0:
        return np.zeros((boxA.shape[0], boxB.shape[0]), dtype=np.float32)

    A = boxA.shape[0]
    B = boxB.shape[0]

    a = boxA[:, None, :]  # [A,1,4]
    b = boxB[None, :, :]  # [1,B,4]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    union = area_a + area_b - inter_area
    iou = np.where(union > 0.0, inter_area / union, 0.0)
    return iou.astype(np.float32)


# -------------------------
# Enkel Kalman-filter
# -------------------------

class KalmanFilter:
    """
    Minimal Kalman-filter för SORT-style tracking.
    State: [cx, cy, s, r, vx, vy, vs]^T   (7D)
    Measurement: [cx, cy, s, r]^T         (4D)
    """

    def __init__(self):
        self.dim_x = 7
        self.dim_z = 4

        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0

        # State transition F
        self.F = np.eye(self.dim_x, dtype=np.float32)
        # position += velocity
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # s  += vs

        # Process noise
        self.Q = np.eye(self.dim_x, dtype=np.float32) * 0.01

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy
        self.H[2, 2] = 1.0  # s
        self.H[3, 3] = 1.0  # r

        # Measurement noise
        self.R = np.eye(self.dim_z, dtype=np.float32) * 1.0

        self.I = np.eye(self.dim_x, dtype=np.float32)

    def initiate(self, measurement):
        """
        measurement: [cx, cy, s, r]
        """
        self.x[:4, 0] = measurement
        self.x[4:, 0] = 0.0  # init velocity = 0
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        measurement: [cx, cy, s, r]
        """
        z = measurement.reshape((self.dim_z, 1)).astype(np.float32)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P

    def get_state_bbox(self):
        """
        return predicted bbox [x1, y1, x2, y2]
        """
        return cxsysr_to_xyxy(self.x[:4, 0])


@dataclass
class KalmanTrack:
    kf: KalmanFilter
    track_id: int
    cls: int
    score: float
    hits: int = 0
    age: int = 0
    time_since_update: int = 0


# -------------------------
# SORT tracker with Kalman
# -------------------------

class KalmanSortTracker:
    """
    SORT-lik multi-object-tracker med Kalman-filter.

    Usage:
        tracker = KalmanSortTracker()
        ...
        boxes, scores, classes = predict.infer_image(...)
        tracks = tracker.update(boxes, scores, classes)


        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["track_id"]
            cls_id = t["cls"]
            score = t["score"]
            
    -Only draw tacker boxes!
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 15,
        min_hits: int = 2,
        match_by_class: bool = True,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age      # max frames utan uppdatering
        self.min_hits = min_hits    # min antal träffar innan track "räknas"
        self.match_by_class = match_by_class

        self.tracks = []            # list[KalmanTrack]
        self._next_id = 1

    def reset(self):
        self.tracks = []
        self._next_id = 1

    def _create_track(self, bbox, score, cls_id):
        kf = KalmanFilter()
        kf.initiate(xyxy_to_cxsysr(bbox))
        tr = KalmanTrack(
            kf=kf,
            track_id=self._next_id,
            cls=int(cls_id),
            score=float(score),
            hits=1,
            age=1,
            time_since_update=0,
        )
        self._next_id += 1
        self.tracks.append(tr)

    def update(self, boxes, scores, classes):
        """
        boxes   : [N,4] i xyxy
        scores  : [N]
        classes : [N]
        return: list av dicts:
            {
                "track_id": int,
                "bbox": np.ndarray [4] (float32, xyxy),
                "cls": int,
                "score": float,
            }
        """

        # Säkerställ np.arrays
        if boxes is None or len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            boxes = np.asarray(boxes, dtype=np.float32)

        if scores is None or len(scores) == 0:
            scores = np.zeros((boxes.shape[0],), dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32)

        if classes is None or len(classes) == 0:
            classes = np.zeros((boxes.shape[0],), dtype=np.int32)
        else:
            classes = np.asarray(classes, dtype=np.int32)

        # 1) Prediktera alla befintliga tracks
        for tr in self.tracks:
            tr.kf.predict()
            tr.age += 1
            tr.time_since_update += 1

        # Om inga detectioner: rensa bort gamla tracks och returnera tomt
        if boxes.shape[0] == 0:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return []

        # 2) Bygg lista av predikterade bboxar
        if len(self.tracks) > 0:
            track_boxes = np.array(
                [tr.kf.get_state_bbox() for tr in self.tracks],
                dtype=np.float32
            )
            det_boxes = boxes
            iou_mat = iou_xyxy(track_boxes, det_boxes)
        else:
            iou_mat = np.zeros((0, boxes.shape[0]), dtype=np.float32)

        # 3) Matchning track <-> detection via greedy IoU
        matched_tracks = set()
        matched_dets = set()
        matches = []

        if len(self.tracks) > 0 and boxes.shape[0] > 0:
            T, D = iou_mat.shape
            # Om vi vill matcha per klass: nolla IoU när klasser skiljer sig
            if self.match_by_class:
                track_cls = np.array([tr.cls for tr in self.tracks])[:, None]   # [T,1]
                det_cls = classes[None, :]                                     # [1,D]
                same = (track_cls == det_cls).astype(np.float32)
                iou_mat = iou_mat * same

            flat = iou_mat.reshape(-1)
            order = np.argsort(-flat)  # fallande IoU

            for idx in order:
                i = idx // D
                j = idx % D
                if iou_mat[i, j] < self.iou_threshold:
                    break
                if i in matched_tracks or j in matched_dets:
                    continue
                matched_tracks.add(i)
                matched_dets.add(j)
                matches.append((i, j))

        # 4) Uppdatera matchade tracks med nya measurement
        for ti, dj in matches:
            tr = self.tracks[ti]
            z = xyxy_to_cxsysr(boxes[dj])
            tr.kf.update(z)
            tr.score = max(tr.score, float(scores[dj]))  # kan tweakas
            # valfritt: uppdatera klass endast om densamma
            if not self.match_by_class:
                tr.cls = int(classes[dj])

            tr.hits += 1
            tr.time_since_update = 0

        # 5) Skapa nya tracks för omatchade detectioner
        unmatched_dets = [j for j in range(boxes.shape[0]) if j not in matched_dets]
        for j in unmatched_dets:
            self._create_track(boxes[j], scores[j], classes[j])

        # 6) Ta bort gamla tracks som inte uppdaterats på max_age frames
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 7) Bygg return-lista: endast tracks som uppdaterats i denna frame och är "stabila"
        outputs = []
        for tr in self.tracks:
            if tr.time_since_update == 0 and tr.hits >= self.min_hits:
                bbox = tr.kf.get_state_bbox()
                outputs.append(
                    {
                        "track_id": tr.track_id,
                        "bbox": bbox,
                        "cls": tr.cls,
                        "score": tr.score,
                    }
                )

        return outputs
