# tools/export_onnx.py
import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= sys.path & imports =========
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Importera dina modelklasser (samma som i train/infer)
from scripts.model.model_v2 import YOLOLiteMS, YOLOLiteMS_CPU  # ändra vid behov


# ========= util: prints =========
def log(msg: str, verbose: bool):
    print(msg, flush=True) if verbose else None

def must(msg: str):
    print(msg, flush=True)


# ========= run-dir =========
def next_run_dir(base: str) -> str:
    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        cand = root / str(n)
        try:
            cand.mkdir(parents=False, exist_ok=False)
            return str(cand.resolve())
        except FileExistsError:
            n += 1


# ========= build model from meta/config =========
def build_model_from_meta(meta: dict) -> nn.Module:
    cfg  = meta.get("config", {}) or {}
    mcfg = cfg.get("model", {}) or {}
    tcfg = cfg.get("training", {}) or {}
    print(cfg)
    arch        = (meta.get("arch") or mcfg.get("arch") or "YOLOLiteMS").lower()
    backbone    = (meta.get("backbone") or mcfg.get("backbone") or "resnet18")
    num_classes = int(meta.get("num_classes") or mcfg.get("num_classes") or 80)
    num_anchors_per_level = tuple(meta.get("num_anchors_per_level") or (1,1,1))
    fpn_channels   = int(mcfg.get("fpn_channels", 128))
    depth_multiple = float(mcfg.get("depth_multiple", 1.0))
    width_multiple = float(mcfg.get("width_multiple", 1.0))
    head_depth     = int(mcfg.get("head_depth", 1))

    img_size = int(tcfg.get("img_size", meta.get("img_size", 640)))
    use_p6 = cfg["training"]["use_p6"]
    use_p2 = cfg["training"]["use_p2"]
    if arch == "yololitems":
        model = YOLOLiteMS(
            backbone=backbone,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple,
            head_depth=head_depth,
            num_anchors_per_level=num_anchors_per_level,
            use_p6=use_p6,
            use_p2=use_p2
        )
    elif arch == "yololitems_cpu":
        model = YOLOLiteMS_CPU(
            backbone=backbone,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple,
            head_depth=head_depth,
            num_anchors_per_level=num_anchors_per_level,
            use_p6=use_p6,
            use_p2=use_p2
        )
    else:
        raise ValueError(f"Okänd arch i meta/config: {arch}")
    return model



# ========= load ckpt =========
def load_model_from_ckpt(weights: str, device: torch.device, verbose: bool) -> Tuple[nn.Module, dict]:
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Viktfil hittas inte: {weights}")
    must(f"• Läser checkpoint: {weights}")
    ckpt = torch.load(weights, map_location=device)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "meta" in ckpt):
        raise RuntimeError("Checkpoint saknar 'state_dict'/'meta' – spara via save_checkpoint_state(...).")
    meta = ckpt["meta"] or {}
    log(f"  meta.keys: {list(meta.keys())}", verbose)
    model = build_model_from_meta(meta)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:    must(f"  [load_state_dict] saknade nycklar: {len(missing)}")
    if unexpected: must(f"  [load_state_dict] oväntade nycklar: {len(unexpected)}")
    model.to(device).eval()
    must(f"• Modell byggd: arch={meta.get('arch')} backbone={meta.get('backbone')} nc={meta.get('num_classes')}")
    return model, meta


# ========= decoded wrapper (utan NMS) =========
class AFDecode(nn.Module):
    def __init__(self, img_size: int, center_mode: str = "v8", wh_mode: str = "softplus"):
        super().__init__()
        self.img_size = int(img_size)
        self.center_mode = center_mode
        self.wh_mode = wh_mode

    @staticmethod
    def _xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
        x, y, w, h = xywh.unbind(-1)
        return torch.stack([x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5], dim=-1)

    def _decode_level(self, p: torch.Tensor):
        # p: [B,A,S,S,D] eller [B,S,S,D] (tolkas A=1)
        if p.dim() == 4:
            p = p.unsqueeze(1)  # [B,1,S,S,D]
        B, A, S, _, D = p.shape
        cell = float(self.img_size) / float(S)

        gy, gx = torch.meshgrid(torch.arange(S, device=p.device),
                                torch.arange(S, device=p.device), indexing="ij")
        gx = gx.float(); gy = gy.float()

        tx = p[..., 0]; ty = p[..., 1]
        tw = p[..., 2]; th = p[..., 3]
        tobj = p[..., 4]
        tcls = p[..., 5:]

        if self.center_mode == "v8":
            px = ((tx.sigmoid() * 2.0 - 0.5) + gx) * cell
            py = ((ty.sigmoid() * 2.0 - 0.5) + gy) * cell
        else:
            px = (tx.sigmoid() + gx) * cell
            py = (ty.sigmoid() + gy) * cell

        if   self.wh_mode == "v8":
            pw = (tw.sigmoid() * 2).pow(2) * cell
            ph = (th.sigmoid() * 2).pow(2) * cell
        elif self.wh_mode == "softplus":
            pw = F.softplus(tw) * cell
            ph = F.softplus(th) * cell
        else:
            pw = tw.clamp(-4, 4).exp() * cell
            ph = th.clamp(-4, 4).exp() * cell

        xyxy = self._xywh_to_xyxy(torch.stack([px, py, pw, ph], dim=-1))
        xyxy[..., 0::2] = xyxy[..., 0::2].clamp(0, self.img_size - 1)
        xyxy[..., 1::2] = xyxy[..., 1::2].clamp(0, self.img_size - 1)

        xyxy = xyxy.reshape(B, -1, 4)
        obj  = tobj.reshape(B, -1, 1)
        cls  = tcls.reshape(B, -1, tcls.shape[-1])
        return xyxy, obj, cls

    def forward(self, preds):
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        boxes, objs, clss = [], [], []
        for p in preds:
            b, o, c = self._decode_level(p)
            boxes.append(b); objs.append(o); clss.append(c)
        boxes = torch.cat(boxes, dim=1)
        obj   = torch.cat(objs,  dim=1)
        cls   = torch.cat(clss,  dim=1)
        return boxes, obj, cls


# ========= main/export =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to checkpoint (.pt/.pth)")
    ap.add_argument("--out", default=None, help="save path (.onnx). Default: runs/export/<n>/model(.onnx|_decoded.onnx)")
    ap.add_argument("--img-size", type=int, default=640, help="(H=W)")
    ap.add_argument("--device", default="cpu", help="'cpu' or cuda. '0'")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--half", action="store_true", help="FP16 (requires CUDA for dummy)")
    ap.add_argument("--simplify", action="store_true", help=" onnxsim after export")
    ap.add_argument("--dynamic-batch", action="store_true", help="Dynamic batch-dimension")
    ap.add_argument("--dynamic-shape", action="store_true", help="Dynamic H/W (only format=raw)")
    ap.add_argument("--format", choices=["raw", "decoded"], default="decoded",
                    help="raw = raw data per nivå; decoded = boxes/obj/cls")
    ap.add_argument("--center-mode", default="v8", choices=["v8", "sigmoid"], help="Decode-center (decoded)")
    ap.add_argument("--wh-mode", default="softplus", choices=["softplus", "v8", "exp"], help="Decode-wh (decoded)")
    ap.add_argument("--verbose", action="store_true", help="Detailed logg")
    args = ap.parse_args()

    # device
    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    must(f"• Device: {device}")

    # ladda modell
    model, meta = load_model_from_ckpt(args.weights, device, verbose=args.verbose)

    # img_size
    meta_img_size = int(meta.get("img_size", args.img_size))
    img_size = int(args.img_size) if args.img_size else meta_img_size
    must(f"• img_size: {img_size}")

    if args.half and device.type == "cuda":
        model.half()
        must("• FP16: ON")

    # dummy input
    B = 1
    H = W = img_size
    dtype = torch.float16 if (args.half and device.type == "cuda") else torch.float32
    dummy = torch.zeros((B, 3, H, W), device=device, dtype=dtype)

    # torrkörning (forward) för att se att något händer
    with torch.inference_mode():
        y = model(dummy)
    if isinstance(y, (list, tuple)):
        must(f"• Torrkörning OK: {len(y)} utgång(ar) (raw head-nivåer).")
    else:
        must("• Torrkörning OK: 1 utgång (monolitisk).")

    export_dir = next_run_dir("runs/export")
    out_path = Path(args.out) if args.out else Path(export_dir) / ("model_decoded.onnx" if args.format == "decoded" else "model.onnx")
    must(f"• Export-katalog: {export_dir}")
    must(f"• Skriver: {out_path}")

    # wrappers
    if args.format == "raw":
        class RawWrapper(nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core
            def forward(self, x):
                y = self.core(x)
                if isinstance(y, (list, tuple)):
                    return tuple(y)
                return (y,)

        wrapper = RawWrapper(model)

        # Hämta output layout och namn
        with torch.inference_mode():
            outs = wrapper(dummy)
        output_names = [f"p{i}" for i in range(len(outs))]
        log(f"  output_names={output_names}", args.verbose)

        dynamic_axes = None
        if args.dynamic_batch or args.dynamic_shape:
            dynamic_axes = {"images": {0: "batch"}}
            for name in output_names:
                dynamic_axes[name] = {0: "batch"}
            if args.dynamic_shape:
                dynamic_axes["images"][2] = "height"
                dynamic_axes["images"][3] = "width"
                must("• Dynamic shape: ON (råt läge)")

        # Export
        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    dummy,
                    str(out_path),
                    opset_version=args.opset,
                    input_names=["images"],
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                )
            must("✓ ONNX export (raw) klar")
        except Exception as e:
            raise RuntimeError(f"ONNX-export (raw) misslyckades: {e}") from e

    else:  # decoded
        if args.dynamic_shape:
            must("! Ignorerar --dynamic-shape i decoded-läge (kräver fast img_size).")

        class DecodedWrapper(nn.Module):
            def __init__(self, core, img_size: int, center_mode: str, wh_mode: str):
                super().__init__()
                self.core = core
                self.decode = AFDecode(img_size=img_size, center_mode=center_mode, wh_mode=wh_mode)
            def forward(self, x):
                y = self.core(x)
                boxes, obj, cls = self.decode(y)
                return boxes, obj, cls

        wrapper = DecodedWrapper(model, img_size=img_size, center_mode=args.center_mode, wh_mode=args.wh_mode)
        output_names = ["boxes_xyxy", "obj_logits", "cls_logits"]
        dynamic_axes = {"images": {0: "batch"},
                        "boxes_xyxy": {0: "batch"}, "obj_logits": {0: "batch"}, "cls_logits": {0: "batch"}}

        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    dummy,
                    str(out_path),
                    opset_version=args.opset,
                    input_names=["images"],
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    do_constant_folding=True,
                )
            must("✓ ONNX export (decoded) klar")
        except Exception as e:
            raise RuntimeError(f"ONNX-export (decoded) misslyckades: {e}") from e

    # simplify
    if args.simplify:
        try:
            import onnx, onnxsim
            model_onnx = onnx.load(str(out_path))
            model_simplified, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(model_simplified, str(out_path))
                must("✓ ONNX simplified")
            else:
                must("! onnxsim returnerade ok=False (sparar osimplifierad)")
        except Exception as e:
            must(f"! onnxsim misslyckades: {e}")

    must("Klart.")


if __name__ == "__main__":
    main()

