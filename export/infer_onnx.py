import argparse, os, sys, json, time
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

# ============ Utils ============
def next_run_dir(base: str) -> str:
    p = Path(base); p.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        cand = p / str(n)
        try:
            cand.mkdir()
            return str(cand.resolve())
        except FileExistsError:
            n += 1

def letterbox(im, new_size=640, color=(114,114,114)):
    h, w = im.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
    return im_padded, scale, (left, top)

def nms_np(boxes, scores, iou_th=0.5, max_det=300):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) >= max_det:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_th]
    return np.array(keep, dtype=np.int64)

def draw(img, boxes, scores, classes, names):
    out = img.copy()
    for b, s, c in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, b.tolist())
        name = names[int(c)] if 0 <= int(c) < len(names) else str(int(c))
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, f"{name} {s:.2f}", (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out

def build_session(model_path, providers, intra, inter):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if intra is not None and intra >= 0:
        so.intra_op_num_threads = int(intra)
    if inter is not None and inter >= 0:
        so.inter_op_num_threads = int(inter)

    prov = []
    p = providers.lower()
    if p == "cuda":
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif p == "tensorrt":
        # fall back chain
        prov = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        prov = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), sess_options=so, providers=prov)

# ============ Main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path till decoded .onnx")
    ap.add_argument("--img", default=None)
    ap.add_argument("--img_dir", default=None)
    ap.add_argument("--img_size", type=int, default=640, help="Måste matcha ONNX decoded img_size")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--no_letterbox", action="store_true")
    ap.add_argument("--save_txt", action="store_true")
    ap.add_argument("--names", default=None, help="Komma-separerad lista eller fil (en per rad)")
    # --- nya benchmark-flaggar ---
    ap.add_argument("--providers", default="cpu", choices=["cpu","cuda","tensorrt"])
    ap.add_argument("--warmup", type=int, default=10, help="Antal tomma pass före mätning")
    ap.add_argument("--runs", type=int, default=1, help="Antal mätta pass per bild")
    ap.add_argument("--intra", type=int, default=0, help="intra_op_num_threads (0=auto)")
    ap.add_argument("--inter", type=int, default=0, help="inter_op_num_threads (0=auto)")
    args = ap.parse_args()

    # class names
    names = []
    if args.names and Path(args.names).exists():
        with open(args.names, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
    elif args.names and "," in args.names:
        names = [s.strip() for s in args.names.split(",")]
    else:
        names = [str(i) for i in range(80)]

    # ort session
    sess = build_session(args.model, args.providers, args.intra, args.inter)
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]  # ["boxes_xyxy","obj_logits","cls_logits"]

    run_dir = next_run_dir("runs/onnx_infer/decoded")
    (Path(run_dir)/"labels").mkdir(parents=True, exist_ok=True)
    (Path(run_dir)/"json").mkdir(parents=True, exist_ok=True)

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD  = np.array([0.229, 0.224, 0.225], np.float32)

    # gather paths
    if args.img and Path(args.img).exists():
        paths = [args.img]
    elif args.img_dir and Path(args.img_dir).exists():
        exts = (".jpg",".jpeg",".png",".bmp")
        paths = sorted([str(p) for p in Path(args.img_dir).glob("*") if p.suffix.lower() in exts])
    else:
        raise SystemExit("Ange --img eller --img_dir")

    # Warmup (syntetiskt eller första bilden)
    dummy = np.random.rand(1,3,args.img_size,args.img_size).astype(np.float32)
    for _ in range(max(0, args.warmup)):
        _ = sess.run(out_names, {in_name: dummy})

    # timing accumulators
    rows = []  # per-bild/loop rader för CSV
    pre_times = []; infer_times = []; post_times = []; total_times = []

    for p in paths:
        img0 = cv2.imread(p)
        if img0 is None:
            print(f"! kunde inte läsa {p}")
            continue

        # Preprocess separat för mätning
        t_pre0 = time.perf_counter()
        if args.no_letterbox:
            lb = cv2.resize(img0, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
            scale = min(args.img_size/img0.shape[0], args.img_size/img0.shape[1])
            padx = pady = 0
        else:
            lb, scale, (padx, pady) = letterbox(img0, args.img_size)

        im = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im = (im - MEAN) / STD
        im = np.transpose(im, (2,0,1))[None]  # [1,3,H,W]
        t_pre1 = time.perf_counter()

        # Multiple measured runs per image
        for run_idx in range(max(1, args.runs)):
            t_inf0 = time.perf_counter()
            boxes, obj_log, cls_log = sess.run(out_names, {in_name: im})
            t_inf1 = time.perf_counter()

            # Postprocess
            t_post0 = time.perf_counter()
            obj = 1/(1+np.exp(-obj_log[...,0]))  # [1,N]
            if cls_log.shape[-1] > 1:
                cls_sig = 1/(1+np.exp(-cls_log[0]))          # [N,C]
                confs = cls_sig.max(axis=-1)                 # [N]
                cls_id = cls_sig.argmax(axis=-1).astype(np.int64)
                scores = obj[0] * confs
            else:
                cls_id = np.zeros_like(obj[0], dtype=np.int64)
                scores = obj[0]

            m = scores > args.conf
            boxes_p = boxes[0][m]
            scores_p = scores[m]
            cls_p = cls_id[m]
            final_b, final_s, final_c = [], [], []
            for c in np.unique(cls_p):
                mc = (cls_p == c)
                keep = nms_np(boxes_p[mc], scores_p[mc], args.iou, args.max_det)
                if keep.size:
                    final_b.append(boxes_p[mc][keep])
                    final_s.append(scores_p[mc][keep])
                    final_c.append(np.full((keep.size,), int(c), dtype=np.int64))

            if final_b:
                boxes_pad = np.concatenate(final_b, 0)
                scores_pad= np.concatenate(final_s, 0)
                classes   = np.concatenate(final_c, 0)
            else:
                boxes_pad = np.zeros((0,4), np.float32)
                scores_pad= np.zeros((0,), np.float32)
                classes   = np.zeros((0,), np.int64)

            # back-map till original
            boxes_px = boxes_pad.copy()
            boxes_px[:,[0,2]] -= padx
            boxes_px[:,[1,3]] -= pady
            boxes_px /= max(scale, 1e-6)
            h0, w0 = img0.shape[:2]
            boxes_px[:,[0,2]] = np.clip(boxes_px[:,[0,2]], 0, w0-1)
            boxes_px[:,[1,3]] = np.clip(boxes_px[:,[1,3]], 0, h0-1)
            t_post1 = time.perf_counter()

            pre_ms   = (t_pre1 - t_pre0)*1000.0
            inf_ms   = (t_inf1 - t_inf0)*1000.0
            post_ms  = (t_post1 - t_post0)*1000.0
            total_ms = (t_post1 - t_pre0)*1000.0

            pre_times.append(pre_ms); infer_times.append(inf_ms); post_times.append(post_ms); total_times.append(total_ms)
            rows.append({
                "image": p, "run": run_idx+1, "pre_ms": round(pre_ms,3),
                "infer_ms": round(inf_ms,3), "post_ms": round(post_ms,3),
                "total_ms": round(total_ms,3)
            })

        # draw & save på sista körningen för bilden
        vis = img0.copy()
        vis = draw(vis, boxes_px, scores_pad, classes, names)
        out_img = str(Path(run_dir)/f"{Path(p).stem}_pred.jpg")
        cv2.imwrite(out_img, vis)
        if args.save_txt and boxes_px.shape[0] > 0:
            h, w = img0.shape[:2]
            xyxy = boxes_px
            cx = (xyxy[:,0]+xyxy[:,2])/(2*w)
            cy = (xyxy[:,1]+xyxy[:,3])/(2*h)
            bw = (xyxy[:,2]-xyxy[:,0])/w
            bh = (xyxy[:,3]-xyxy[:,1])/h
            with open(Path(run_dir)/"labels"/f"{Path(p).stem}.txt","w",encoding="utf-8") as f:
                for c,x,y,ww,hh,s in zip(classes,cx,cy,bw,bh,scores_pad):
                    f.write(f"{int(c)} {x:.6f} {y:.6f} {ww:.6f} {hh:.6f} {s:.4f}\n")

        # json
        rec = []
        for b,s,c in zip(boxes_px.tolist(), scores_pad.tolist(), classes.tolist()):
            name = names[c] if 0 <= c < len(names) else str(c)
            rec.append({"bbox_xyxy":[float(x) for x in b], "score":float(s), "class_id":int(c), "class_name":name})
        with open(Path(run_dir)/"json"/f"{Path(p).stem}.json","w",encoding="utf-8") as f:
            json.dump({"image": p, "detections": rec}, f, ensure_ascii=False, indent=2)

        print(f"✓ {out_img}")

    # --- sammanfattning ---
    def stats(x):
        x = np.array(x, dtype=np.float64)
        return dict(mean=float(np.mean(x)), std=float(np.std(x)),
                    p50=float(np.percentile(x,50)), p90=float(np.percentile(x,90)), p95=float(np.percentile(x,95)))
    summary = {
        "providers": args.providers,
        "warmup": args.warmup,
        "runs_per_image": args.runs,
        "counts": len(rows),
        "pre_ms": stats(pre_times),
        "infer_ms": stats(infer_times),
        "post_ms": stats(post_times),
        "total_ms": stats(total_times),
        "throughput_images_per_s": (1000.0/np.mean(total_times)) if total_times else None
    }

    # skriv ut och spara
    print("\n=== Inference timing (ms) ===")
    for k in ["pre_ms","infer_ms","post_ms","total_ms"]:
        s = summary[k]
        print(f"{k:9s} mean {s['mean']:.2f} | std {s['std']:.2f} | p50 {s['p50']:.2f} | p90 {s['p90']:.2f} | p95 {s['p95']:.2f}")
    if summary["throughput_images_per_s"]:
        print(f"Throughput ≈ {summary['throughput_images_per_s']:.2f} img/s")

    # save files
    with open(Path(run_dir)/"timings.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(Path(run_dir)/"timings.csv","w",encoding="utf-8") as f:
        f.write("image,run,pre_ms,infer_ms,post_ms,total_ms\n")
        for r in rows:
            f.write(f"{r['image']},{r['run']},{r['pre_ms']},{r['infer_ms']},{r['post_ms']},{r['total_ms']}\n")

    print(f"\nAllt sparat i: {run_dir}")

if __name__ == "__main__":
    main()
