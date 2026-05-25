import csv
import os
import time
import numpy as np
import torch
from yololite import YoloLite
from ultralytics import YOLO

# ==========================================
# KONFIGURATION
# *** IMPORTANT NOTICE FOR yololite ***
# This script assumes that you have no train/det folder or that it is empty.
# All models are speed/evauated on the test split of the repo and evaluated on test as well.
# This benchmark requires version 1.1.9 of yololite to work correctly.
# ==========================================
DATASETS = [
            "data.yaml",
            "data1.yaml, 
            
            ] 
EPOCHS = 50
BATCH_SIZE = 8 
IMG_SIZE = 640


MODELS_TO_TEST = {
    "YoloLite": ["yololite_mnv4_n.pt", "yololite_cs3_n.pt", "yololite_mnv4_s.pt"], #"yololite_cs3_s.pt", "yololite_hg2_n.pt", "yololite_hg2_s.pt" ,"yololite_hg2_n.pt"
    "Ultralytics": ["yolov5n", "yolov10n.pt", "yolo26n.pt", "yolov8n", "yolo26s.pt"] 
    
}

OUTPUT_FILE = 'benchmark.csv'


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def init_csv():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset', 'Framework', 'Model', 'mAP50', 'mAP50-95', 'CPU ONNX Latency (ms)', 'CPU FPS'])

def save_result(dataset, framework, model_name, map50, map_total, latency):
    fps = 1000 / latency if latency > 0 else 0
    with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, framework, model_name, f"{map50:.3f}", f"{map_total:.3f}", f"{latency:.2f}", f"{fps:.1f}"])

# ==========================================
# PIPELINE
# ==========================================

def run_pipeline():
    init_csv()
    print("🚀 Starting Benchmark...")
    i = 1
    for dataset in DATASETS:
        print(f"\n{'='*60}\nEvaluating Dataset: {dataset}\n{'='*60}")
        
        for framework, model_list in MODELS_TO_TEST.items():
            for model_name in model_list:
                print(f"\n--- Proceessing {framework}: {model_name} ---")
                
                map50, map_total, latency = 0.0, 0.0, 0.0
                
                try:
                    # ---------------------------------------------------------
                    # 1. TRAINING AND EVALUATION (mAP)
                    # ---------------------------------------------------------
                    print("[1/3] Training and evaluating model...")
                    if framework == "YoloLite":
                        model = YoloLite(model_name)
                        
                        
                        model.train(data=dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, workers=4, accumulate=2)
                        
                        model = YoloLite(f"runs/det/{i}/weights/best.pt")
                        
                        eval_results = model.val(data=dataset, split="test")
                        map50 = eval_results.get("map_50", torch.tensor(0.0)).item()
                        map_total = eval_results.get("map", torch.tensor(0.0)).item()
                        i += 1
                    elif framework == "Ultralytics":
                        model = YOLO(model_name)
                        train_results = model.train(data=dataset, epochs=EPOCHS, batch=BATCH_SIZE, imgsz=IMG_SIZE, workers=0)
                        
                        # Ultralytics runs/detect/train/weights/best.pt 
                        best_weight_path = train_results.save_dir / "weights/best.pt" if hasattr(train_results, 'save_dir') else model_name
                        best_model = YOLO(best_weight_path)
                        eval_results = best_model.val(data=dataset, split="test")
                        
                        map50 = eval_results.box.map50
                        map_total = eval_results.box.map
                    
                    print(f"✅ mAP50: {map50:.3f} | mAP50-95: {map_total:.3f}")

                    # ---------------------------------------------------------
                    # 2. EXPORT TO ONNX
                    # ---------------------------------------------------------
                    print("[2/3] Exports to onnx for cpu benchmark...")
                    onnx_path = None
                    if framework == "YoloLite":
                        onnx_path = model.export(format='decoded', simplify=True, verbose=False)
                        inf_model = YoloLite(onnx_path)
                    elif framework == "Ultralytics":
                        # YOLO export returnerar sträng till filen
                        onnx_path = best_model.export(format='onnx', simplify=True)
                        inf_model = YOLO(onnx_path, task='detect')

                    # ---------------------------------------------------------
                    # 3. CPU INFERENCE (Latens)
                    # ---------------------------------------------------------
                    print("[3/3] Benchmarking CPU Latency...")
                    

                    # Tidtagnings-loop
                    times = []
                    
                    if framework == "YoloLite":
                        image_dir = os.path.join(os.path.dirname(dataset), 'test', 'images')
                        print(image_dir)
                        
                        for image in os.listdir(image_dir):
                            img_path = os.path.join(image_dir, image)

                            res = inf_model.predict(img_path, device='cpu', draw=False)[0]
                            # Se till att du mäter samma sak som Ultralytics (total_ms vs inference)
                            times.append(res["speed"]["total_ms"])
                    else:
                        image_dir = os.path.join(os.path.dirname(dataset), 'test', 'images')
                        for image in os.listdir(image_dir):
                            img_path = os.path.join(image_dir, image)
                            res = inf_model.predict(img_path, device='cpu', verbose=False)[0]
                            # Ultralytics speed returnerar ms per pre/infer/post process. 
                            # Vi summerar dem för att få en rättvis "total_ms" motsvarighet.
                            total_ms = sum(res.speed.values())
                            times.append(total_ms)

                    latency = np.mean(times)
                    print(f"✅ CPU Latency: {latency:.2f} ms")

                    # Spara resultat
                    save_result(dataset, framework, model_name, map50, map_total, latency)

                except Exception as e:
                    print(f"❌ Error with {model_name}: {e}")
                    save_result(dataset, framework, model_name, 0, 0, 0) # Spara fail

if __name__ == '__main__':
    run_pipeline()
    print(f"\n🏆 Pipeline done! Results saved to {os.path.abspath(OUTPUT_FILE)}")
