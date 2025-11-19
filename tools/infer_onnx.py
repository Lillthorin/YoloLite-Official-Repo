# onnx_infer_decoded_min.py

import numpy as np
import cv2
import onnxruntime as ort
import torch

# --------- Preprocess (letterbox + ImageNet norm) ----------
def letterbox(img_bgr, new_size=640, color=(114,114,114)):
    h, w = img_bgr.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im_resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return im_padded, scale, (left, top)

def preprocess_bgr_letterbox(img_bgr, img_size):
    lb, scale, (padx, pady) = letterbox(img_bgr, img_size)
    img = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))[None]  # [1,3,H,W]
    return img, scale, padx, pady

def preprocess_bgr_resize(img_bgr, img_size):
    """
    Ren resize till (img_size, img_size) utan padding.
    Återanvänder samma ImageNet-normalisering.
    """
    resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))[None]  # [1,3,H,W]
    return img  # ingen scale/pad behövs här

# --------- Postprocess ----------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def nms_per_class(boxes, scores, iou_th=0.5, topk=None):
    # boxes: Nx4 (xyxy), scores: N
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if topk and len(keep) >= topk:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h

        # ✅ korrekt union-beräkning
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        order = order[1:][iou <= iou_th]

    return np.array(keep, dtype=np.int64)


def postprocess(decoded_outs, conf_th=0.25, nms_iou=0.5, max_det=300):
    boxes = decoded_outs["boxes_xyxy"][0]               # [N,4]
    obj_log = decoded_outs["obj_logits"][0].reshape(-1) # [N]
    cls_log = decoded_outs["cls_logits"][0]             # [N,C]

    obj = 1.0 / (1.0 + np.exp(-obj_log))                # [N]
    C = cls_log.shape[-1] if cls_log.ndim == 2 else 0

    if C > 1:
        cls_sig = 1.0 / (1.0 + np.exp(-cls_log))        # [N,C]
        cls_ids = cls_sig.argmax(axis=1)                # [N]
        cls_scores = cls_sig.max(axis=1)                # [N]
        scores = obj * cls_scores
    elif C == 1:
        # matcha infer_onnx.py: använd bara obj vid 1-klass
        cls_ids = np.zeros_like(obj, dtype=np.int64)
        scores = obj
    else:
        # saknar klasslogits (ovanligt) – använd obj
        cls_ids = np.zeros_like(obj, dtype=np.int64)
        scores = obj

    m = scores > conf_th
    if not np.any(m):
        return (np.zeros((0,4),np.float32),
                np.zeros((0,),np.float32),
                np.zeros((0,),np.int64))

    boxes = boxes[m]
    scores = scores[m]
    cls_ids = cls_ids[m]

    # per-klass NMS
    final_b, final_s, final_c = [], [], []
    for c in np.unique(cls_ids):
        mc = (cls_ids == c)
        keep = nms_per_class(boxes[mc], scores[mc], iou_th=nms_iou)
        if keep.size:
            final_b.append(boxes[mc][keep])
            final_s.append(scores[mc][keep])
            final_c.append(np.full((keep.size,), int(c), dtype=np.int64))

    if not final_b:
        return (np.zeros((0,4),np.float32),
                np.zeros((0,),np.float32),
                np.zeros((0,),np.int64))

    boxes = np.concatenate(final_b, 0).astype(np.float32)
    scores = np.concatenate(final_s, 0).astype(np.float32)
    classes = np.concatenate(final_c, 0).astype(np.int64)

    if boxes.shape[0] > max_det:
        top = scores.argsort()[::-1][:max_det]
        boxes, scores, classes = boxes[top], scores[top], classes[top]
    return boxes, scores, classes


# --------- Core API ----------

class ONNX_Predict:
    """
    Minimal infer for ONNX-decoded export:
    outputs = ["boxes_xyxy", "obj_logits", "cls_logits"]

    use_letterbox:
        True  -> letterbox + padding (classic YOLO-scaling)
        False -> pure resize (img_size, img_size)
    """
    def __init__(self, onnx_path: str, providers=None, use_letterbox: bool = True):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        # map outputs by name to avoid order issues
        self.output_map = {o.name: o.name for o in self.session.get_outputs()}
        self.use_letterbox_default = use_letterbox

    def infer_image(
        self,
        img_bgr,
        img_size: int = 640,
        conf: float = 0.25,
        iou: float = 0.50,
        max_det: int = 300,
        use_letterbox: bool | None = None,
    ):
        """
        img_bgr: OpenCV BGR-bild (H,W,3)
        use_letterbox:
            None  -> använd default från __init__
            True  -> letterbox
            False -> ren resize
        """
        if use_letterbox is None:
            use_letterbox = self.use_letterbox_default

        orig_h, orig_w = img_bgr.shape[:2]

        # ----- Preprocess -----
        if use_letterbox:
            inp, scale, padx, pady = preprocess_bgr_letterbox(img_bgr, img_size)
        else:
            inp = preprocess_bgr_resize(img_bgr, img_size)
            scale = None
            padx = 0
            pady = 0

        # ----- ONNX run -----
        outs = self.session.run(
            [
                self.output_map.get("boxes_xyxy"),
                self.output_map.get("obj_logits"),
                self.output_map.get("cls_logits"),
            ],
            {self.input_name: inp}
        )
        boxes_net, obj_logits, cls_logits = outs

        # ----- Postprocess (NMS etc i nätverkskoordinater) -----
        boxes, scores, classes = postprocess(
            {
                "boxes_xyxy": boxes_net,
                "obj_logits": obj_logits,
                "cls_logits": cls_logits,
            },
            conf_th=conf,
            nms_iou=iou,
            max_det=max_det,
        )

        # ----- Back-map till originalbild -----
        if boxes.shape[0]:
            if use_letterbox:
                # samma som i benchmark-scriptet
                boxes[:, [0, 2]] -= padx
                boxes[:, [1, 3]] -= pady
                boxes /= max(scale, 1e-6)
            else:
                # ren resize: img0 -> nät-inp via warp till (img_size, img_size)
                # x_net = x_orig * (img_size / orig_w)  =>  x_orig = x_net * (orig_w / img_size)
                # y_net = y_orig * (img_size / orig_h)  =>  y_orig = y_net * (orig_h / img_size)
                sx = orig_w / float(img_size)
                sy = orig_h / float(img_size)
                boxes[:, [0, 2]] *= sx
                boxes[:, [1, 3]] *= sy

            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_w - 1)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_h - 1)

        return boxes, scores, classes


