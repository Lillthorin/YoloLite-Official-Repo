import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Hjälp-funktioner ---------------------------

def bbox_iou_matrix(box1, box2, eps=1e-7):
    """
    Beräknar IoU mellan alla N_anchors och alla N_gt i samma bild.
    box1: [N_a, 4]
    box2: [N_gt, 4]
    Return: [N_a, N_gt]
    """
    b1 = box1.unsqueeze(1) # [N_a, 1, 4]
    b2 = box2.unsqueeze(0) # [1, N_gt, 4]

    # Intersection
    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    # Union
    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + eps

    return inter / union

def bbox_ciou_flat(pred_xyxy, target_xyxy, eps=1e-7):
    # Standard CIoU implementation för 1-till-1 matchade par
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=eps); ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps); th = (ty2 - ty1).clamp(min=eps)

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    pcx = (px1 + px2) * 0.5; pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5; tcy = (ty1 + ty2) * 0.5
    center_dist = (pcx - tcx)**2 + (pcy - tcy)**2

    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw**2 + ch**2 + eps

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return iou - (center_dist / c2) - alpha * v

def _targets_to_xyxy_px(tgt: dict, W: int, H: int, device: torch.device):
    # Din original-funktion (säkerställer att datat hanteras exakt lika)
    boxes = None
    for k in ("boxes", "bboxes", "xyxy"):
        if k in tgt and tgt[k] is not None:
            boxes = tgt[k]; break
    if boxes is None: return torch.zeros((0, 4), dtype=torch.float32, device=device)
    
    if isinstance(boxes, torch.Tensor): b = boxes.detach().to(device=device, dtype=torch.float32)
    else: b = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    
    if b.numel() == 0: return b.view(0, 4)

    # Heuristik (som i ditt original)
    b_min, b_max = float(b.min().item()), float(b.max().item())
    
    def xywh_px_to_xyxy_px(bt):
        x1 = bt[:, 0] - bt[:, 2]*0.5; y1 = bt[:, 1] - bt[:, 3]*0.5
        x2 = bt[:, 0] + bt[:, 2]*0.5; y2 = bt[:, 1] + bt[:, 3]*0.5
        return torch.stack([x1, y1, x2, y2], dim=1)

    # Normalized
    if -1e-3 <= b_min <= 1.01 and -1e-3 <= b_max <= 1.01:
        mean_wh = float((b[:, 2] + b[:, 3]).mean().item())
        if mean_wh <= 2.01: # xywhn
            cx = b[:, 0]*W; cy = b[:, 1]*H; ww = b[:, 2]*W; hh = b[:, 3]*H
            return xywh_px_to_xyxy_px(torch.stack([cx, cy, ww, hh], dim=1))
        else: # xyxyn
            x1 = b[:, 0]*W; y1 = b[:, 1]*H; x2 = b[:, 2]*W; y2 = b[:, 3]*H
            return torch.stack([x1, y1, x2, y2], dim=1)
            
    likely_xyxy = ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).float().mean().item() > 0.8
    return b if likely_xyxy else xywh_px_to_xyxy_px(b)

# --------------------------- LossAF (Hybrid) ---------------------------

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Hjälp-funktioner ---------------------------

def bbox_iou_matrix(box1, box2, eps=1e-7):
    """
    Beräknar IoU mellan alla N_anchors och alla N_gt i samma bild.
    box1: [N_a, 4], box2: [N_gt, 4] -> [N_a, N_gt]
    """
    b1 = box1.unsqueeze(1)
    b2 = box2.unsqueeze(0)

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + eps

    return inter / union

def bbox_ciou_flat(pred_xyxy, target_xyxy, eps=1e-7):
    # Standard CIoU för 1-till-1 par
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = target_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=eps); ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps); th = (ty2 - ty1).clamp(min=eps)

    inter_w = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    inter_h = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    iou = inter / union

    pcx = (px1 + px2) * 0.5; pcy = (py1 + py2) * 0.5
    tcx = (tx1 + tx2) * 0.5; tcy = (ty1 + ty2) * 0.5
    center_dist = (pcx - tcx)**2 + (pcy - tcy)**2

    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw**2 + ch**2 + eps

    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return iou - (center_dist / c2) - alpha * v

def _targets_to_xyxy_px(tgt: dict, W: int, H: int, device: torch.device):
    boxes = None
    for k in ("boxes", "bboxes", "xyxy"):
        if k in tgt and tgt[k] is not None:
            boxes = tgt[k]; break
    if boxes is None: return torch.zeros((0, 4), dtype=torch.float32, device=device)
    
    if isinstance(boxes, torch.Tensor): b = boxes.detach().to(device=device, dtype=torch.float32)
    else: b = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    
    if b.numel() == 0: return b.view(0, 4)

    b_min, b_max = float(b.min().item()), float(b.max().item())
    
    def xywh_px_to_xyxy_px(bt):
        x1 = bt[:, 0] - bt[:, 2]*0.5; y1 = bt[:, 1] - bt[:, 3]*0.5
        x2 = bt[:, 0] + bt[:, 2]*0.5; y2 = bt[:, 1] + bt[:, 3]*0.5
        return torch.stack([x1, y1, x2, y2], dim=1)

    if -1e-3 <= b_min <= 1.01 and -1e-3 <= b_max <= 1.01:
        mean_wh = float((b[:, 2] + b[:, 3]).mean().item())
        if mean_wh <= 2.01: # xywhn
            cx = b[:, 0]*W; cy = b[:, 1]*H; ww = b[:, 2]*W; hh = b[:, 3]*H
            return xywh_px_to_xyxy_px(torch.stack([cx, cy, ww, hh], dim=1))
        else: # xyxyn
            x1 = b[:, 0]*W; y1 = b[:, 1]*H; x2 = b[:, 2]*W; y2 = b[:, 3]*H
            return torch.stack([x1, y1, x2, y2], dim=1)
            
    likely_xyxy = ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])).float().mean().item() > 0.8
    return b if likely_xyxy else xywh_px_to_xyxy_px(b)

# --------------------------- LossAF (Hybrid + Rescue) ---------------------------

class LossAF(nn.Module):
    def __init__(self,
                 num_classes: int,
                 img_size: int,
                 lambda_box: float = 5.0,
                 lambda_obj: float = 1.0,
                 lambda_cls: float = 0.5,
                 assign_cls_weight: float = 0.5,
                 center_mode: str = "v8",
                 wh_mode: str = "softplus",
                 center_radius_cells: float = 2.0,
                 topk_limit: int = 20,
                 focal: bool = False,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 cls_smoothing: float = 0.05,
                 area_cells_min: float = 4.0,
                 area_cells_max: float = 256.0,
                 area_tol: float = 1.25,
                 size_prior_w: float = 0.20,
                 ar_prior_w: float = 0.10,
                 iou_cost_w: float = 3.0,
                 center_cost_w: float = 0.5):
        super().__init__()
        self.nc = int(num_classes)
        self.img_size = int(img_size)
        self.lambda_box = float(lambda_box)
        self.lambda_obj = float(lambda_obj)
        self.lambda_cls = float(lambda_cls)
        self.assign_cls_weight = float(assign_cls_weight)
        self.center_mode = center_mode
        self.wh_mode = wh_mode
        self.center_radius_cells = float(center_radius_cells)
        self.topk_limit = int(topk_limit)
        self.focal = focal
        self.gamma = gamma
        self.alpha = alpha
        
        # Tolerate small objects better
        self.area_cells_min = float(area_cells_min) / float(area_tol)
        self.area_cells_max = float(area_cells_max) * float(area_tol)
        
        self.size_prior_w = size_prior_w
        self.ar_prior_w = ar_prior_w
        self.iou_cost_w = iou_cost_w
        self.center_cost_w = center_cost_w

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_cls = nn.CrossEntropyLoss(reduction="none", label_smoothing=cls_smoothing)
        self.register_buffer("obj_balance", torch.ones(8)) 
        
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)

    def _make_anchors(self, preds: List[torch.Tensor], device: torch.device):
        shapes = [p.shape[2:4] for p in preds]
        total = sum(s[0]*s[1] for s in shapes)
        if self.anchors.shape[0] != total or self.anchors.device != device:
            al, sl = [], []
            for p in preds:
                _, _, h, w, _ = p.shape
                stride = self.img_size / max(h, w)
                sy, sx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
                al.append(torch.stack([sx, sy], dim=-1).float().view(-1, 2))
                sl.append(torch.full((h * w,), stride, device=device))
            self.anchors = torch.cat(al, dim=0)
            self.strides = torch.cat(sl, dim=0)

    def _decode(self, preds_flat):
        strides = self.strides.view(1, -1, 1)
        anchors = self.anchors.view(1, -1, 2)
        tx, ty, tw, th = preds_flat[..., 0], preds_flat[..., 1], preds_flat[..., 2], preds_flat[..., 3]
        
        if self.center_mode == "v8":
            xy = (torch.sigmoid(torch.stack([tx, ty], -1)) * 2.0 - 0.5 + anchors) * strides
        else:
            xy = (torch.sigmoid(torch.stack([tx, ty], -1)) + anchors) * strides
            
        if self.wh_mode == "v8":
            wh = (torch.sigmoid(torch.stack([tw, th], -1)) * 2).pow(2) * strides
        elif self.wh_mode == "softplus":
             wh = F.softplus(torch.stack([tw, th], -1)) * strides
        else:
             wh = torch.exp(torch.stack([tw, th], -1).clamp(-10, 8)) * strides
             
        x1y1 = xy - 0.5 * wh; x2y2 = xy + 0.5 * wh
        return torch.cat([x1y1, x2y2], dim=-1), xy, wh, strides.squeeze(-1)

    def _focal_weight(self, logits, targets):
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return alpha_t * (1 - pt).pow(self.gamma)

    def forward(self, preds: List[torch.Tensor], targets: List[dict]):
        device = preds[0].device
        if self.obj_balance.device != device: self.obj_balance = self.obj_balance.to(device)
            
        self._make_anchors(preds, device)
        preds_flat = torch.cat([p.flatten(1, 3) for p in preds], dim=1) # [B, N_tot, C]
        B, N_total, _ = preds_flat.shape
        pred_xyxy, pred_ctr, pred_wh, _ = self._decode(preds_flat)
        
        pred_obj = preds_flat[..., 4]
        pred_cls = preds_flat[..., 5:]
        
        loss_box = torch.zeros(1, device=device)
        loss_obj = torch.zeros(1, device=device)
        loss_cls = torch.zeros(1, device=device)
        total_pos = 0.0

        anchor_levels = (torch.log2(self.strides) - torch.log2(self.strides[0])).long().clamp(min=0, max=7)

        # Loop per bild (Bevarar statistik för små/svåra objekt)
        for b in range(B):
            tgt_xyxy = _targets_to_xyxy_px(targets[b], self.img_size, self.img_size, device)
            tgt_labels = targets[b]["labels"].to(device).long()
            N_gt = tgt_xyxy.shape[0]

            if N_gt == 0:
                obj_t = torch.zeros_like(pred_obj[b])
                neg_scores = self.bce(pred_obj[b], obj_t)
                K = min(max(64, 3), neg_scores.numel())
                neg_obj = neg_scores.topk(K).values.mean()
                loss_obj += self.lambda_obj * neg_obj
                continue

            # --- Pre-calc ---
            iou_matrix = bbox_iou_matrix(pred_xyxy[b], tgt_xyxy)
            gt_ctr = (tgt_xyxy[:, :2] + tgt_xyxy[:, 2:]) * 0.5 
            gt_wh = (tgt_xyxy[:, 2:] - tgt_xyxy[:, :2]).clamp(min=1.0)
            
            dist_sq = (pred_ctr[b].unsqueeze(1) - gt_ctr.unsqueeze(0)).pow(2).sum(-1)
            s_col = self.strides.unsqueeze(1)
            
            # --- Tweak 1: Min-Radius Guard (15px) ---
            # Garanterar att även mikroskopiska bollar får en chans att träffa anchors
            raw_r = (self.center_radius_cells * s_col) + 0.10 * gt_wh.max(dim=1).values.unsqueeze(0)
            r_pix = torch.clamp(raw_r, min=15.0) 
            
            center_mask = dist_sq <= r_pix.pow(2)
            
            # Level gating
            gt_area = gt_wh.prod(dim=1).unsqueeze(0)
            area_cells = gt_area / (s_col.pow(2))
            level_mask = (area_cells >= self.area_cells_min) & (area_cells <= self.area_cells_max)
            
            valid_mask = center_mask & level_mask

            # --- Tweak 2: Orphan Rescue (Kritiskt för bollarna) ---
            # Om en boll missade alla anchors, tvinga närmaste att matcha
            gt_hits = valid_mask.sum(dim=0)
            orphans = gt_hits == 0
            if orphans.any():
                orphan_indices = orphans.nonzero(as_tuple=True)[0]
                nearest_anchor_idx = dist_sq[:, orphan_indices].argmin(dim=0)
                valid_mask[nearest_anchor_idx, orphan_indices] = True

            # --- Cost ---
            pred_cls_prob = torch.sigmoid(pred_cls[b])
            class_probs = pred_cls_prob[:, tgt_labels]
            cls_cost = 1.0 - class_probs
            obj_cost = -torch.sigmoid(pred_obj[b]).unsqueeze(1)
            
            p_area = pred_wh[b].prod(dim=1).unsqueeze(1)
            size_cost = (p_area.log() - gt_area.log()).abs() / (1.0 + (p_area.log() - gt_area.log()).abs())
            
            p_ar = (pred_wh[b, :, 0] / pred_wh[b, :, 1]).unsqueeze(1).log()
            g_ar = (gt_wh[:, 0] / gt_wh[:, 1]).unsqueeze(0).log()
            ar_cost = (p_ar - g_ar).abs() / (1.0 + (p_ar - g_ar).abs())
            
            center_norm = dist_sq / (gt_wh[:, 0]**2 + gt_wh[:, 1]**2 + 1e-6).unsqueeze(0)

            cost = (
                self.iou_cost_w * (1.0 - iou_matrix) + 
                self.assign_cls_weight * cls_cost + 
                obj_cost +
                self.center_cost_w * center_norm +
                self.size_prior_w * size_cost +
                self.ar_prior_w * ar_cost
            )
            cost[~valid_mask] = 1e9

            # --- SimOTA ---
            iou_masked = iou_matrix.masked_fill(~valid_mask, 0.0)
            topk_ious, _ = torch.topk(iou_masked, k=min(self.topk_limit, N_total), dim=0)
            dynamic_ks = topk_ious.sum(dim=0).int().clamp(min=1)
            
            _, indices = torch.topk(cost, k=min(self.topk_limit, N_total), dim=0, largest=False)
            
            match_matrix = torch.zeros_like(cost, dtype=torch.bool)
            for gt_i in range(N_gt):
                k = dynamic_ks[gt_i].item()
                match_matrix[indices[:k, gt_i], gt_i] = True
                
            if match_matrix.sum(dim=1).max() > 1:
                multiple_mask = match_matrix.sum(dim=1) > 1
                c_multi = cost[multiple_mask].clone()
                m_multi = match_matrix[multiple_mask]
                c_multi[~m_multi] = 1e9
                best_gt = c_multi.argmin(dim=1)
                match_matrix[multiple_mask] = False
                match_matrix[multiple_mask, best_gt] = True
            
            # --- Loss ---
            pos_mask = match_matrix.any(dim=1)
            pos_inds = pos_mask.nonzero(as_tuple=False).squeeze(1)
            
            if pos_inds.numel() == 0:
                obj_t = torch.zeros_like(pred_obj[b])
                neg_scores = self.bce(pred_obj[b], obj_t)
                K = min(max(64, 3), neg_scores.numel())
                neg_obj = neg_scores.topk(K).values.mean()
                loss_obj += self.lambda_obj * neg_obj
                continue
            
            total_pos += 1.0
            matched_gt_idx = match_matrix[pos_inds].long().argmax(dim=1)
            
            # Mean per image (Crucial!)
            target_box = tgt_xyxy[matched_gt_idx]
            pred_box = pred_xyxy[b, pos_inds]
            iou_final = bbox_ciou_flat(pred_box, target_box)
            loss_box += self.lambda_box * (1.0 - iou_final).mean()
            
            target_label = tgt_labels[matched_gt_idx]
            pred_logit_cls = pred_cls[b, pos_inds]
            loss_cls += self.lambda_cls * self.bce_cls(pred_logit_cls, target_label).mean()
            
            obj_target_scores = iou_matrix[pos_inds, matched_gt_idx].detach().clamp(0, 1).to(pred_obj.dtype)
            obj_t_full = torch.zeros_like(pred_obj[b])
            obj_t_full[pos_inds] = obj_target_scores
            
            pos_obj = self.bce(pred_obj[b][pos_inds], obj_t_full[pos_inds]).mean()
            
            neg_mask = ~pos_mask
            neg_scores = self.bce(pred_obj[b][neg_mask], obj_t_full[neg_mask])
            K = min(max(64, 3 * int(pos_inds.numel())), neg_scores.numel())
            neg_obj = neg_scores.topk(K).values.mean() if K > 0 else torch.tensor(0.0, device=device)
            
            loss_obj += self.lambda_obj * (pos_obj + neg_obj)

        return loss_box + loss_obj + loss_cls, {
            "box": float(loss_box),
            "obj": float(loss_obj),
            "cls": float(loss_cls),
            "pos": total_pos / max(B, 1),
        }