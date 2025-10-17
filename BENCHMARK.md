


# Benchmark

A short benchmark was done on five different datasets fetched from roboflow 100.

https://universe.roboflow.com/roboflow-100

The mAP numbers for YOLOv5 and YOLOv7 models can be found in their paper
"Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark" - https://arxiv.org/abs/2211.13523v3

The benchmark was done using standard training. 

    python tools/train.py --model "configs/models/edge_s.yaml" --batch_size 4 --epochs 200 

Configs for each model can be found under configs/model.  

| Dataset              | YOLOv5 | YOLOv7 | edge_l | edge_m | edge_s | edge_n |
|----------------------|------:|------:|------:|-------:|------:|------:|
| chess pieces         | 0.977 | 0.830 | 0.914 | 0.964 | 0.964 | 0.681 |
| circuit voltages     | 0.797 | 0.257 | 0.749 | 0.687  | 0.757 | 0.621 |
| sedimentary features | 0.327 | 0.244 | 0.181 | 0.199  | 0.213 | 0.154 |
| soccer players       | 0.666 | 0.399 | 0.756 | 0.758  | 0.734 | 0.714 |
| solar panels         | 0.413 | 0.261 | 0.317 | 0.313  | 0.310 | 0.270 |

NOTE!

The models perform very similar to eachother, and sometimes the smaller model wins. This might be to several different factors. Larger models need more steps to converge. See this benchmark as a quick demonstration of the models potential to be trained on different datasets. 

# Speed

Speed testing was done by first converting each model to onnx with export_onnx.py 

    python export/export_onnx.py --weights runs/train/1/weights/best_model_state.pt --simplify

After exporting all images was tested on a test image with the following command

    python export/infer_onnx.py --model runs/export/medium/model_decoded.onnx --img "testimg.jpg" --providers "cpu" 

cuda

    python export/infer_onnx.py --model runs/export/medium/model_decoded.onnx --img "testimg.jpg" --providers "cuda"

Hardware: 
CPU AMD Ryzen 5 5500
GPU NVIDIA GeForce RTX 4060


| Device | edge_l | edgle_m | edge_s | edge_n |
|------: |------: |------:  |-------:|------:|
| CPU    | 67.0   | 45.57   | 40.18  | 23.88 |
| GPU    | 24.95  | 23.05   | 20.83  | 20.2  | 


# Training logs

When training first a folder will be created runs/train 

Under train a subfolder x will be created 1, 2, 3 ....

When training starts a sanity_check.jpg is created, this show a 4x4 picuture of 4 images from the train_loader that gets sent to the model. 
This is a good way to see if labels were loaded correctly. 

merged_config.yaml is created as the complete configuration for the training, training parameters, model and dataset is saved here.

During validation steps last_b0.jpg and last_b1.jpg will be created these are two random images from the validation loop with the models predictions. 
You can check these to see the models progress. 


More information about plots are comming. 






