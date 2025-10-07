import torch
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms

def _xywh_to_xyxy_t(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh.unbind(-1)
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)

def _make_grid(S, device):
    gy, gx = torch.meshgrid(
        torch.arange(S, device=device),
        torch.arange(S, device=device),
        indexing="ij"
    )
    return gx.view(1, S, S), gy.view(1, S, S)




@torch.no_grad()
def decode_preds_anchorfree(
    preds_levels,            # [P3,P4,P5,...], varje [B,A,S,S,5+C] ELLER [B,S,S,5+C]
    img_size: int,
    center_mode: str = "v8",     # "v8" | "simple"  (samma som i save_val_debug_anchorfree)
    wh_mode: str = "softplus",   # "v8" | "softplus" | "exp"
):
    """
    Anchor-FREE decode (utan mask/NMS).
    Returnerar en dict med 3 tensorer:
      box: [B, N, 4]  xyxy i pixlar (float)
      cls: [B, N, C]  CLASS-LOGITS (inte sigmoid)
      obj: [B, N, 1]  OBJ-LOGITS   (inte sigmoid)

    Där N = sum_{levels}(A*S*S).
    Senare steg (t.ex. COCOeval/rit-funktion) kan göra sigmoid, score, tröskel och NMS.
    """
    import torch
    import torch.nn.functional as F

    levels = preds_levels if isinstance(preds_levels, (list, tuple)) else [preds_levels]
    device = levels[0].device
    B = levels[0].shape[0]
    C = int(levels[0].shape[-1]) - 5

    def _make_grid(S, dev):
        gy, gx = torch.meshgrid(
            torch.arange(S, device=dev),
            torch.arange(S, device=dev),
            indexing="ij"
        )
        # [1,1,S,S] för broadcast
        return gx.view(1,1,S,S), gy.view(1,1,S,S)

    # per-batch ackumulatorer
    boxes_all = []
    cls_all   = []
    obj_all   = []

    for pred in levels:
        # Normalisera shape till [B,A,S,S,D]
        if pred.dim() == 4:  # [B,S,S,D] => A=1
            pred = pred.unsqueeze(1)               # [B,1,S,S,D]
        B_, A, S, _, D = pred.shape
        assert B_ == B, "Batch mismatch mellan nivåer"

        stride = img_size / float(S)
        gx, gy = _make_grid(S, device)

        # Split
        tx = pred[..., 0]                          # [B,A,S,S]
        ty = pred[..., 1]
        tw = pred[..., 2]
        th = pred[..., 3]
        tobj = pred[..., 4]                        # obj-logits [B,A,S,S]
        tcls = pred[..., 5:] if C > 0 else pred.new_zeros((*pred.shape[:4], 0))  # [B,A,S,S,C]

        # --- CENTER (som i din rit-funktion) ---
        if center_mode == "v8":
            px = ((torch.sigmoid(tx) * 2.0 - 0.5) + gx) * stride
            py = ((torch.sigmoid(ty) * 2.0 - 0.5) + gy) * stride
        else:  # "simple"
            px = (torch.sigmoid(tx) + gx) * stride
            py = (torch.sigmoid(ty) + gy) * stride

        # --- SIZE (som i din rit-funktion) ---
        if   wh_mode == "v8":
            pw = (torch.sigmoid(tw) * 2.0).pow(2.0) * stride
            ph = (torch.sigmoid(th) * 2.0).pow(2.0) * stride
        elif wh_mode == "softplus":
            pw = F.softplus(tw) * stride
            ph = F.softplus(th) * stride
        else:  # "exp"
            pw = tw.clamp(-4, 4).exp() * stride
            ph = th.clamp(-4, 4).exp() * stride

        # Bygg xyxy
        x1 = (px - pw * 0.5).clamp(0, img_size - 1)
        y1 = (py - ph * 0.5).clamp(0, img_size - 1)
        x2 = (px + pw * 0.5).clamp(0, img_size - 1)
        y2 = (py + ph * 0.5).clamp(0, img_size - 1)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)       # [B,A,S,S,4]

        # Platta ut A,S,S
        Nl = A * S * S
        boxes = boxes.view(B, Nl, 4)                        # [B,Nl,4]
        obj   = tobj.view(B, Nl, 1)                         # [B,Nl,1]
        cls   = tcls.view(B, Nl, C) if C > 0 else tcls.view(B, Nl, 0)  # [B,Nl,C]

        boxes_all.append(boxes)
        obj_all.append(obj)
        cls_all.append(cls)

    # Konkatenerea nivåer längs N
    boxes_cat = torch.cat(boxes_all, dim=1)                 # [B,N,4]
    obj_cat   = torch.cat(obj_all,   dim=1)                 # [B,N,1]
    cls_cat   = torch.cat(cls_all,   dim=1)                 # [B,N,C]

    return {"box": boxes_cat, "obj": obj_cat, "cls": cls_cat}


