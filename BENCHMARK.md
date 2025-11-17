


# Benchmark

A short benchmark was done on five different datasets fetched from roboflow 100.

https://universe.roboflow.com/roboflow-100

The mAP numbers for YOLOv5 and YOLOv7 models can be found in their paper
"Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark" - https://arxiv.org/abs/2211.13523v3

**NEW VALUES AFTER UPDATED LOSS FUNCTION!**

yololite_m and yololite_n refers to the models under models_v2 folder. 
Some datasets are still in progress, table will be updated.

Benchmark was done with epochs = 100 and batch size = 8 and img_size 640.
    python tools/train.py --model "configs/models/edge_{x}.yaml" --batch_size 8 --epochs 100 

| Dataset               | YOLOv5 | YOLOv7 | edge_m | edge_n | yololite_m | yololite_n |
|-----------------------|:------:|:------:|:------:|:------:|:------:|:------:|
| axial mri             | 0.638  | 0.549  | 0.584 | 0.412 | 0.546  | 0.642  |
| bccd ouzjz            | 0.912  | 0.922  | 0.895 | 0.895 | 0.884  | 0.881  |
| chess pieces          | 0.977  | 0.830  | 0.985 | 0.989 | 0.990  | 0.988  |
| circuit voltages      | 0.797  | 0.257  | 0.871 | 0.755 | 0.826  | 0.833  |
| farcry6 videogame     | 0.619  | 0.216  | 0.494 | 0.412 | 0.526  | 0.448  |
| gauge u2lwv           | 0.642  | 0.668  | 0.600 | 0.589 | 0.630  | 0.629  |
| lettuce pallets       | 0.945  | 0.966  | 0.915 | 0.901 | 0.898  | 0.899  |
| mask wearing          | 0.788  | 0.513  | 0.575 | 0.481 | 0.757  | 0.725  |
| sedimentary features  | 0.327  | 0.244  | 0.339 | 0.250 | 0.364  | 0.355  |
| shark teeth           | 0.948  | 0.863  | 1.000 | 0.985 | 1.000  | 0.989  |
| sign language         | 0.870  | 0.255  | 0.954 | 0.915 |   —    | 0.961  |
| signatures xc8up      | 0.961  | 0.932  | 0.819 | 0.785 | 0.883  | 0.902  |
| soccer players        | 0.660  | 0.399  | 0.768 | 0.687 | 0.782  | 0.783  |
| solar panels          | 0.413  | 0.261  | 0.481 | 0.317 | 0.623  | 0.576  |
| street work           | 0.478  | 0.708  | 0.555 | 0.496 | 0.631  | 0.620  |
| thermal cheetah       | 0.931  | 0.513  | 0.854 | 0.708 | 0.834  | 0.810  |
| thermal dogs          | 0.967  | 0.957  | 0.935 | 0.906 | 0.916  | 0.940  |
| valentines chocolate  | 0.110  | 0.059  | 0.978 | 0.951 | 0.981  | 0.983  |
| weed crop aerial      | 0.820  | 0.615  | 0.544 | 0.435 | 0.592  | 0.581  |
| x ray                 | 0.722  | 0.506  | 0.837 | 0.800 | 0.843  | 0.835  |
| currency              | 0.583  | 0.514  | 0.963 | 0.914 | 0.979  | 0.977 |
| cable damage          | 0.910  | 0.574  | 0.820 | 0.762 | 0.---  | 0.838 |
| apples                | 0.779  | 0.791  | 0.687 | 0.692 | 0.---  | 0.742 |
| secondary chains      | 0.341  | 0.312  | 0.209 | 0.257 | 0.284  | 0.285 |
| marbels               | 0.992  | 0.473  | 0.799 | 0.715 | 0.823  | 0.803 |
| leaf disease          | 0.531  | 0.560  | 0.543 | 0.516 | 0.544  | 0.538 |
| pests                 | 0.136  | 0.029  | 0.135 | 0.090 | 0.218  | 0.196 |
| bacteria              | 0.162  | 0.001  | 0.000 | 0.000 | 0.000  | 0.000 |
| mitosis               | 0.931  | 0.739  | 0.947 | 0.898 | 0.946  | 0.944 |
| aerial pool           | 0.513  | 0.791  | 0.372 | 0.337 | 0.391  | 0.391 |
| poker cards           | 0.886  | 0.251  | 0.992 | 0.981 | 0.991  | 0.995 |
| bone fracture         | 0.085  | 0.090  | 0.202 | 0.105 | 0.259  | 0.199 |
| cotton                | 0.569  | 0.591  | 0.284 | 0.399 | 0.502  | 0.367 |
| cells                 | 0.249  | 0.085  | 0.522 | 0.349 | 0.582  | 0.582 |
| aerial spheres        | 0.993  | 0.539  | 0.967 | 0.956 | 0.965  | 0.970 |


# Speed
**ALL SPEED MEASURMENTS INCLUDE PRE/POST OPS!**
Speed testing was done by first converting each model to onnx with export_onnx.py 

    python export/export_onnx.py --weights runs/train/1/weights/best_model_state.pt --simplify

After exporting all images was tested on a test image with the following commands

    python export/infer_onnx.py --model runs/export/medium/model_decoded.onnx --img "testimg.jpg"  

cuda

    python export/infer_onnx.py --model runs/export/medium/model_decoded.onnx --img "testimg.jpg" --provider "cuda"

Hardware: 
CPU AMD Ryzen 5 5500
GPU NVIDIA GeForce RTX 4060


| Device | edge_l | edgle_m | edge_s | edge_n |
|------: |------: |------:  |-------:|------:|
| CPU    | 67.0ms  | 45.57ms    | 40.18ms   | 23.88ms  |
| GPU    | 24.95ms   | 23.05ms    | 20.83ms   | 20.2ms   | 

V2_Models  NaN == Not tested 

| Device | Yololite_n | Yololite_s | Yololite_m | Yololite_l |
|------: |------: |------:  |-------:|------:|
| CPU    | 87.05ms  | NaN    | 156.59ms   | NaN  |
| GPU    | 21.07ms   | NaN   | 25.65ms   | NaN   | 


# Training logs

When training first a folder will be created runs/train 

Under train a subfolder x will be created 1, 2, 3 ....

When training starts a sanity_check.jpg is created, this show a 4x4 picuture of 4 images from the train_loader that gets sent to the model. 
This is a good way to see if labels were loaded correctly. 

merged_config.yaml is created as the complete configuration for the training, training parameters, model and dataset is saved here.

During validation steps last_b0.jpg and last_b1.jpg will be created these are two random images from the validation loop with the models predictions. 
You can check these to see the models progress. 


More information about plots are comming. 

















