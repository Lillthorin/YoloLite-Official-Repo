# loss.py
import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Hjälp-funktioner ---------------------------

def _targets_to_xyxy_px(tgt: dict, W: int, H: int, device: torch.device):
    """
    Accepts tgt["boxes"] as:
      - xywhn (0..1), xyxyn (0..1)
      - xywh (pixels), xyxy (pixels)
    Returns [N,4] xyxy in pixels (on device).
    """
    # --- pick first present key without boolean-evaluating tensors ---
    boxes = None
    for k in ("boxes", "bboxes", "xyxy"):
        if k in tgt and tgt[k] is not None:
            boxes = tgt[k]
            break

    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)

    # to tensor
    if isinstance(boxes, torch.Tensor):
        b = boxes.detach().to(device=device, dtype=torch.float32)
    else:
        b = torch.as_tensor(boxes, dtype=torch.float32, device=device)

    if b.numel() == 0:
        return b.view(0, 4)

    # heuristics: normalized vs pixels; xyxy vs xywh
    b_min = float(b.min().item())
    b_max = float(b.max().item())

    def xywh_px_to_xyxy_px(bt: torch.Tensor) -> torch.Tensor:
        x1 = bt[:, 0] - bt[:, 2] * 0.5
        y1 = bt[:, 1] - bt[:, 3] * 0.5
        x2 = bt[:, 0] + bt[:, 2] * 0.5
        y2 = bt[:, 1] + bt[:, 3] * 0.5
        return torch.stack([x1, y1, x2, y2], dim=1)

    # normalized (0..1)
    if -1e-3 <= b_min <= 1.01 and -1e-3 <= b_max <= 1.01:
        mean_wh = float((b[:, 2] + b[:, 3]).mean().item())
        if mean_wh <= 2.01:  # xywhn
            cx = b[:, 0] * W
            cy = b[:, 1] * H
            ww = b[:, 2] * W
            hh = b[:, 3] * H
            return xywh_px_to_xyxy_px(torch.stack([cx, cy, ww, hh], dim=1))
        else:  # xyxyn
            x1 = b[:, 0] * W
            y1 = b[:, 1] * H
            x2 = b[:, 2] * W
            y2 = b[:, 3] * H
            return torch.stack([x1, y1, x2, y2], dim=1)

    # pixels: decide xyxy vs xywh
    likely_xyxy = ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).float().mean().item() > 0.8
    return b if likely_xyxy else xywh_px_to_xyxy_px(b)


def xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = xywh.unbind(-1)
    x1 = cx - w * 0.5; y1 = cy - h * 0.5
    x2 = cx + w * 0.5; y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_ciou(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5
    center_dist = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw ** 2 + ch ** 2 + eps

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)

    ciou = iou - (center_dist / c2) - alpha * v
    return ciou


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    N = box1.size(0); M = box2.size(0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    union = area1[:, None] + area2 - inter + eps
    return (inter / union).clamp(min=0.0, max=1.0)


# ---------------------------stil AF-loss ---------------------------

class LossAF(nn.Module):
    """
    Anchor-free,k matchning:
      - Level-gating (ATSS-lik) baserat på GT-storlek vs nivåns stride.
      - Center-prior (radie ~ 2.5*stride) + liten GT-storleksandel.
      - Cost = IoU-driven SimOTA-hybrid + (liten) size/AR-prior.
      - Dynamic-K (IoU-summering) + unik matchning.
      - Dynamiskt antal nivåer (P3..P5 eller P3..P6), strides hämtas per nivå.
    Decode följer din modell: center_mode {'simple','v8'}, wh_mode {'v8','softplus','exp'}.
    """

    def __init__(self,
                 num_classes: int,
                 img_size: int,
                 lambda_box: float = 5.0,
                 lambda_obj: float = 1.0,
                 lambda_cls: float = 0.5,
                 assign_cls_weight: float = 0.5,
                 center_mode: str = "v8",   # "simple" eller "v8"
                 wh_mode: str = "softplus",           # "v8" | "softplus" | "exp"
                 center_radius_cells: float = 2.0,
                 topk_limit: int = 20,
                 focal: bool = False,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 cls_smoothing: float = 0.05,
                 # ATSS-lik area-gating i "cells" (px_area / stride^2)
                 area_cells_min: float = 4.0,
                 area_cells_max: float = 256.0,
                 area_tol: float = 1.25,
                 # små priorvikter
                 size_prior_w: float = 0.20,
                 ar_prior_w: float = 0.10,
                 iou_cost_w: float = 3.0,
                 center_cost_w: float = 0.5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_cls = float(lambda_cls)
        self.assign_cls_weight = float(assign_cls_weight)

        self.center_mode = str(center_mode)
        self.wh_mode = str(wh_mode)
        self.center_radius_cells = float(center_radius_cells)
        self.topk_limit = int(topk_limit)

        self.focal = bool(focal)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.cls_smoothing = float(cls_smoothing)

        # ATSS-lik gating i cell-ytor (invarianta över nivå)
        self.area_cells_min = float(area_cells_min) / float(area_tol)
        self.area_cells_max = float(area_cells_max) * float(area_tol)

        # cost-vikter
        self.size_prior_w = float(size_prior_w)
        self.ar_prior_w = float(ar_prior_w)
        self.iou_cost_w = float(iou_cost_w)
        self.center_cost_w = float(center_cost_w)

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.register_buffer("obj_balance", torch.ones(8))  # vägs per nivå (expanderas)
        self._levels_seen = 0

    # ---------------- Decode-stöd ----------------

    def _decode_wh(self, tw, th, stride):
        s = float(stride)
        if self.wh_mode == "v8":
            # OBS: detta cappar till 4*s – byt gärna till "softplus" om du vill ha obegränsat.
            pw = (torch.sigmoid(tw) * 2).pow(2) * s
            ph = (torch.sigmoid(th) * 2).pow(2) * s
        elif self.wh_mode == "softplus":
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=False):
                pw = F.softplus(tw.float()) * s
                ph = F.softplus(th.float()) * s
            pw = pw.to(tw.dtype); ph = ph.to(th.dtype)
        else:  # exp (stabil)
            with torch.cuda.amp.autocast(enabled=False):
                pw = torch.exp(tw.float().clamp(-10, 8)) * s
                ph = torch.exp(th.float().clamp(-10, 8)) * s
            pw = pw.to(tw.dtype); ph = ph.to(th.dtype)
        return pw, ph

    # ---------------- Hjälp för focal/label smoothing ----------------

    @staticmethod
    def _smooth_onehot(indices: torch.Tensor, n_classes: int, eps: float = 0.0) -> torch.Tensor:
        device = indices.device; N = indices.numel()
        if n_classes == 1:
            return torch.ones((N, 1), device=device)
        if eps <= 0:
            out = torch.zeros((N, n_classes), device=device)
            out.scatter_(1, indices.view(-1, 1).clamp_(0, n_classes - 1), 1.0)
            return out
        off = eps / (n_classes - 1)
        out = torch.full((N, n_classes), off, device=device)
        out.scatter_(1, indices.view(-1, 1).clamp_(0, n_classes - 1), 1.0 - eps)
        return out

    def _focal_weight(self, logits, targets):
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return alpha_t * (1 - pt).pow(self.gamma)

    def _ensure_balance_len(self, li: int):
        if li >= self.obj_balance.numel():
            new_len = max(li + 1, self.obj_balance.numel() * 2)
            new_buf = torch.ones(new_len, device=self.obj_balance.device, dtype=self.obj_balance.dtype)
            new_buf[:self.obj_balance.numel()] = self.obj_balance
            self.obj_balance = nn.Parameter(new_buf, requires_grad=False)

    # ---------------- Huvud: forward ----------------

    def forward(self, preds: List[torch.Tensor], targets: List[dict], anchors=None):
        device = preds[0].device
        B = preds[0].shape[0]; C = self.num_classes

        # init i samma dtype/device som pred
        loss_box = preds[0].new_tensor(0.0)
        loss_obj = preds[0].new_tensor(0.0)
        loss_cls = preds[0].new_tensor(0.0)
        total_pos = 0

        for li, p in enumerate(preds):
            self._ensure_balance_len(li)
            B_, A, S, S2, D = p.shape
            assert S == S2, "Grid måste vara kvadratisk"
            stride = self.img_size / S  # fungerar för olika antal nivåer

            # --- Decode (anchor-free)
            tx, ty, tw, th = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
            tobj = p[..., 4]
            tcls_logit = p[..., 5:]

            # grid
            gy, gx = torch.meshgrid(
                torch.arange(S, device=device),
                torch.arange(S, device=device),
                indexing="ij"
            )
            gx = gx.view(1, 1, S, S)
            gy = gy.view(1, 1, S, S)

            if self.center_mode == "v8":
                px = ((torch.sigmoid(tx) * 2.0 - 0.5) + gx) * stride
                py = ((torch.sigmoid(ty) * 2.0 - 0.5) + gy) * stride
            else:
                px = (torch.sigmoid(tx) + gx) * stride
                py = (torch.sigmoid(ty) + gy) * stride

            pw, ph = self._decode_wh(tw, th, stride)

            pred_xywh = torch.stack([px, py, pw, ph], dim=-1)   # [B,A,S,S,4]
            pred_xyxy = xywh_to_xyxy(pred_xywh).view(B, -1, 4)  # [B,Np,4]
            obj_logits = tobj.view(B, -1)                       # [B,Np]
            cls_logits = tcls_logit.view(B, -1, C)              # [B,Np,C]
            Np = pred_xyxy.shape[1]

            for b in range(B):
                # Hämta GT för denna bild
                gt = _targets_to_xyxy_px(targets[b], W=self.img_size, H=self.img_size, device=device)  # [N,4]

                if gt.numel() == 0:
                    # negativa bilder: hård negativ mining på obj
                    obj_t = obj_logits[b].new_zeros(obj_logits[b].shape)  # rätt dtype/device
                    neg_scores = self.bce(obj_logits[b], obj_t)
                    K = min(max(64, 3), neg_scores.numel())
                    neg_obj = neg_scores.topk(K).values.mean() if K > 0 else obj_t.new_tensor(0.0)
                    if self.focal and neg_scores.numel():
                        fw = self._focal_weight(obj_logits[b], obj_t)
                        neg_obj = (neg_scores * fw).mean()
                    with torch.no_grad():
                        self.obj_balance[li] = 0.98 * self.obj_balance[li] + 0.02 * 1.0
                    balance = (self.obj_balance * (int(li + 1) / self.obj_balance[:li + 1].sum())).detach()
                    loss_obj = loss_obj + self.lambda_obj * neg_obj * balance[li]
                    continue

                gtl = targets[b]["labels"].to(device).long()  # [N]
                N = gt.shape[0]
                if N == 0:
                    continue

                # Förbered cost-termer (beräkna i fp32 för stabilitet)
                ious = bbox_iou(pred_xyxy[b], gt)  # [Np,N], fp32
                gt1h = torch.zeros(N, C, device=device, dtype=torch.float32)
                gt1h[torch.arange(N, device=device), gtl] = 1.0

                cls_cost = (1.0 - torch.sigmoid(cls_logits[b].float()) @ gt1h.T)  # [Np,N], fp32
                obj_cost = -torch.sigmoid(obj_logits[b].float()).unsqueeze(1)     # [Np,1] broadcast, fp32

                # Center/size info
                px_flat = px[b].reshape(-1).float()          # [Np]
                py_flat = py[b].reshape(-1).float()          # [Np]
                pw_flat = pw[b].reshape(-1).clamp_min(1.0).float()
                ph_flat = ph[b].reshape(-1).clamp_min(1.0).float()

                gt_cx = ((gt[:, 0] + gt[:, 2]) * 0.5).float()
                gt_cy = ((gt[:, 1] + gt[:, 3]) * 0.5).float()
                gt_w  = (gt[:, 2] - gt[:, 0]).clamp_min(1.0).float()
                gt_h  = (gt[:, 3] - gt[:, 1]).clamp_min(1.0).float()

                dx = px_flat.unsqueeze(1) - gt_cx.unsqueeze(0)  # [Np,N], fp32
                dy = py_flat.unsqueeze(1) - gt_cy.unsqueeze(0)  # [Np,N], fp32

                # ---------------- ATSS-lik level-gating ----------------
                area_cells = (gt_w * gt_h) / (stride * stride)  # [N], fp32
                in_level = (area_cells >= self.area_cells_min) & (area_cells <= self.area_cells_max)  # [N], bool

                # Center-prior
                r_pix = (self.center_radius_cells * stride) + 0.10 * torch.maximum(gt_w, gt_h)  # [N]
                center_mask = (dx * dx + dy * dy) <= (r_pix.unsqueeze(0) ** 2)                  # [Np,N]

                # Kombinerad mask
                valid_mask = center_mask.clone()
                if (~in_level).any():
                    valid_mask[:, ~in_level] = False

                # ---------------- Kostnad (SimOTA-hybrid) ----------------
                center_norm = (dx * dx + dy * dy) / ((gt_w * gt_w + gt_h * gt_h).unsqueeze(0) + 1e-6)

                pred_area = (pw_flat.unsqueeze(1) * ph_flat.unsqueeze(1)).clamp_min(1.0)
                gt_area   = (gt_w * gt_h).clamp_min(1.0)
                size_cost = (pred_area.log() - gt_area.log().unsqueeze(0)).abs()
                size_cost = size_cost / (1.0 + size_cost)

                ar_pred = (pw_flat / ph_flat).unsqueeze(1).clamp(1e-6, 1e6).log()
                ar_gt   = (gt_w / gt_h).clamp(1e-6, 1e6).log().unsqueeze(0)
                ar_cost = (ar_pred - ar_gt).abs()
                ar_cost = ar_cost / (1.0 + ar_cost)

                cost = self.iou_cost_w * (1.0 - ious) \
                    + self.assign_cls_weight * cls_cost \
                    + obj_cost \
                    + self.center_cost_w * center_norm \
                    + self.size_prior_w * size_cost \
                    + self.ar_prior_w * ar_cost

                # maska bort ogiltiga kandidater
                cost = cost.masked_fill(~valid_mask, 1e9)

                # ---------------- Dynamic-K (IoU-summering) ----------------
                cand_per_gt = valid_mask.sum(dim=0)  # [N]
                ious_masked = ious.masked_fill(~valid_mask, 0.0)

                base_cap = 10
                kmin, kmax = 5, self.topk_limit
                k_cap_vec = (base_cap + torch.sqrt(area_cells)).to(torch.int64)
                k_cap_vec = torch.clamp(k_cap_vec, min=kmin, max=kmax)
                k_cap_vec = torch.minimum(k_cap_vec, cand_per_gt.clamp(min=1))

                match_matrix = torch.zeros_like(cost, dtype=torch.bool)  # [Np,N]

                for gi in range(N):
                    k_cap_j = int(k_cap_vec[gi].item())
                    if k_cap_j <= 0:
                        continue
                    topk_vals, _ = torch.topk(ious_masked[:, gi], k=k_cap_j, dim=0)
                    dyn_k = int(torch.clamp(topk_vals.sum().round(), min=1, max=k_cap_j).item())
                    sel = torch.topk(cost[:, gi], k=dyn_k, largest=False).indices
                    match_matrix[sel, gi] = True

                # gör matcherna unika per pred
                if match_matrix.any():
                    multiple = match_matrix.sum(dim=1) > 1
                    if multiple.any():
                        mm = multiple.nonzero(as_tuple=False).squeeze(1)
                        best_gt = cost[mm].argmin(dim=1)
                        match_matrix[mm] = False
                        match_matrix[mm, best_gt] = True

                pos_idx = match_matrix.any(dim=1).nonzero(as_tuple=False).squeeze(1)  # [P]
                if pos_idx.numel() == 0:
                    # negativa på denna nivå
                    obj_t = obj_logits[b].new_zeros(obj_logits[b].shape)
                    neg_scores = self.bce(obj_logits[b], obj_t)
                    K = min(max(64, 3), neg_scores.numel())
                    neg_obj = neg_scores.topk(K).values.mean() if K > 0 else obj_t.new_tensor(0.0)
                    if self.focal and neg_scores.numel():
                        fw = self._focal_weight(obj_logits[b], obj_t)
                        neg_obj = (neg_scores * fw).mean()
                    with torch.no_grad():
                        self.obj_balance[li] = 0.98 * self.obj_balance[li] + 0.02 * 1.0
                    balance = (self.obj_balance * (int(li + 1) / self.obj_balance[:li + 1].sum())).detach()
                    loss_obj = loss_obj + self.lambda_obj * neg_obj * balance[li]
                    continue

                total_pos += int(pos_idx.numel())
                assigned_gt_idx = match_matrix[pos_idx].float().argmax(dim=1)  # [P]
                assigned_gt = gt[assigned_gt_idx]                              # [P,4]
                assigned_cls = gtl[assigned_gt_idx]                            # [P]

                # Box loss (CIoU) – kör i pred/gt's dtype (fp16 funkar), men ciou själv blir fp32 i vår implementation; casta tillbaka
                pred_pos = pred_xyxy[b, pos_idx]
                ciou = bbox_ciou(pred_pos, assigned_gt).clamp(0, 1)
                # säker typ vid ackumulerad loss
                loss_box = loss_box + self.lambda_box * (1.0 - ciou).to(loss_box.dtype).mean()

                # Obj loss: pos styrs av IoU-target, neg via HNM
                obj_t_full = obj_logits[b].new_zeros(obj_logits[b].shape)   # [Np], rätt dtype/device
                iou_pos = ious[pos_idx, assigned_gt_idx].detach().clamp_(0, 1)
                # matcha dtype
                if iou_pos.dtype != obj_t_full.dtype:
                    iou_pos = iou_pos.to(obj_t_full.dtype)

                # robust skrivning (undvik in-place advanced indexing dtype-problem)
                obj_t_full.index_copy_(0, pos_idx, iou_pos)

                pos_obj = self.bce(obj_logits[b][pos_idx], obj_t_full[pos_idx]).mean()

                neg_mask = torch.ones(Np, dtype=torch.bool, device=device)
                neg_mask[pos_idx] = False
                neg_scores = self.bce(obj_logits[b][neg_mask], obj_t_full[neg_mask])
                K = min(max(64, 3 * int(pos_idx.numel())), neg_scores.numel())
                neg_obj = neg_scores.topk(K).values.mean() if K > 0 else obj_t_full.new_tensor(0.0)

                if self.focal:
                    pos_obj *= self._focal_weight(obj_logits[b][pos_idx], obj_t_full[pos_idx]).mean()
                    if neg_scores.numel():
                        fw = self._focal_weight(obj_logits[b][neg_mask], obj_t_full[neg_mask])
                        neg_obj = (neg_scores * fw).topk(K).values.mean()

                level_obj = pos_obj + neg_obj
                with torch.no_grad():
                    inv = 1.0 / float(max(int(pos_idx.numel()), 1))
                    self.obj_balance[li] = 0.98 * self.obj_balance[li] + 0.02 * inv
                balance = (self.obj_balance * (int(li + 1) / self.obj_balance[:li + 1].sum())).detach()
                loss_obj = loss_obj + self.lambda_obj * level_obj * balance[li]

                # Cls loss (endast positives) – targets i logits' dtype
                # Cls loss (endast positives) – CE behöver INTE one-hot
                logits_pos = cls_logits[b][pos_idx]            # [P, C], float
                targets_pos = assigned_cls                     # [P], long
                cls_l = F.cross_entropy(
                    logits_pos.float(),
                    targets_pos.long(),
                    reduction="mean",
                    label_smoothing=self.cls_smoothing if self.cls_smoothing > 0 else 0.0,
                )

                loss_cls = loss_cls + self.lambda_cls * cls_l.to(loss_cls.dtype)
                
                

        total = loss_box + loss_obj + loss_cls
        return total, {
            "box": float(loss_box),
            "obj": float(loss_obj),
            "cls": float(loss_cls),
            "pos": total_pos / max(B, 1),
        }


