


# Benchmark
*On-going*

Last update: 2025-12-08
Version: 1.0

## Dataset Source
All datasets are taken from the Roboflow-100 benchmark:
[https://universe.roboflow.com/roboflow-100](https://universe.roboflow.com/roboflow-100)

## Baseline Numbers (YOLOv5 / YOLOv7)
The mAP numbers for YOLOv5 and YOLOv7 shown in the tables below are **not trained by this project**.
They are taken directly from the official Roboflow-100 benchmark paper:

**Ciaglia, F. et al. (2022).  
*Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark*.  
arXiv:2211.13523.**  
[[Paper]](https://arxiv.org/abs/2211.13523)


**NEW VALUES AFTER UPDATED LOSS FUNCTION!**

yololite_m and yololite_n refers to the models under models_v2 folder. 
Some datasets are still in progress, table will be updated.

N.A = training in progress

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
| sign language         | 0.870  | 0.255  | 0.954 | 0.915 | 0.938  | 0.961  |
| signatures xc8up      | 0.961  | 0.932  | 0.819 | 0.785 | 0.883  | 0.902  |
| soccer players        | 0.660  | 0.399  | 0.800 | 0.758 | 0.782  | 0.783  |
| solar panels          | 0.413  | 0.261  | 0.481 | 0.317 | 0.623  | 0.576  |
| street work           | 0.478  | 0.708  | 0.555 | 0.496 | 0.631  | 0.620  |
| thermal cheetah       | 0.931  | 0.513  | 0.854 | 0.708 | 0.834  | 0.810  |
| thermal dogs          | 0.967  | 0.957  | 0.935 | 0.906 | 0.916  | 0.940  |
| valentines chocolate  | 0.110  | 0.059  | 0.978 | 0.951 | 0.981  | 0.983  |
| weed crop aerial      | 0.820  | 0.615  | 0.544 | 0.435 | 0.592  | 0.581  |
| x ray                 | 0.722  | 0.506  | 0.837 | 0.800 | 0.843  | 0.835  |
| currency              | 0.583  | 0.514  | 0.963 | 0.914 | 0.979  | 0.977 |
| cable damage          | 0.910  | 0.574  | 0.820 | 0.762 | 0.863  | 0.838 |
| apples                | 0.779  | 0.791  | 0.687 | 0.692 | 0.751  | 0.742 |
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
| aquarium              | 0.790  | 0.822  | 0.614 | 0.489 | 0.698  | 0.667 |
| vehicles              | 0.454  | 0.464  | 0.379 | 0.349 | 0.360  | 0.348 |
| activity diagrams     | 0.427  | 0.509  | 0.631 | 0.436 | 0.639  | 0.611 |
| peanuts               | 0.995  | 0.997  | 1.000 | 1.000 | 1.000  | 1.000 |
| people in paintings   | 0.575  | 0.678  | 0.309 | 0.217 | 0.374  | 0.344 |
| hand-gestures         | 0.995  | 0.995  | 0.999 | 1.000 | 1.000  | 0.999 |
| truck-movement        | 0.786  | 0.846  | 0.681 | 0.658 | 0.685  | 0.682 |
| cell towers           | 0.939  | 0.942  | 0.881 | 0.860 | N.A  | 0.896 |
| document parts        | 0.677  | 0.666  | 0.629 | 0.651 | N.A  | 0.634 |
| grass weeds           | 0.781  | 0.781  | 0.688 | 0.685 | N.A  | 0.681 |
| trail camera          | 0.966  | 0.969  | 0.901 | 0.853 | N.A  | 0.916 |
| tweeter posts         | 0.708  | 0.495  | 0.724 | 0.634 | N.A  | 0.666 |
| tweeter profile       | 0.988  | 0.990  | 0.983 | 0.974 | N.A  | 0.976 |
| cavity                | 0.782  | 0.799  | 0.625 | 0.573 | 0.612 | 0.634 |
| avatar-recognition    | 0.889  | 0.943  | 0.742 | 0.652 | 0.786 | 0.741 |
| pills                 | 0.869  | 0.867  | 0.861 | 0.849 | 0.860 | 0.854 |
| wall damage           | 0.500  | 0.434  | 0.421 | 0.345 | N.A  | 0.456 |
| corrosion             | 0.768  | 0.764  | 0.539 | 0.541 | 0.543  | 0.534 |
| road traffic          | 0.597  | 0.847  | 0.659 | 0.507 | 0.672  | 0.631 |
| road sign             | 0.963  | 0.944  | 0.927 | 0.938 | N.A  | N.A |
| liver-disease         | 0.592  | 0.583  | 0.528 | 0.533 | N.A   | N.A  |
| wine labels           | 0.569  | 0.632  | 0.501 | 0.390 | N.A   | N.A  |
| radio signal          | 0.673  | 0.653  | 0.591 | 0.587 | N.A   | N.A  |
| paragraphs            | 0.626  | 0.610  | 0.488 | 0.442 | N.A   | N.A  |
| halo infinite         | 0.921  | 0.924  | 0.825 | 0.786 | N.A   | N.A  |
| cables                | 0.688  | 0.722  | 0.773 | 0.729 | N.A   | N.A  |
| construction-safety   | 0.915  | 0.915  | 0.786 | 0.663 | N.A   | 0.813  |
| smoke                 | 0.959  | 0.962  | 0.938 | 0.939 | 0.939 | 0.942  |
| animals               | 0.761  | 0.342  | 0.647 | 0.493 | 0.744 | 0.725  |
| phages                | 0.854  | 0.842  | 0.648 | 0.611 | 0.721 | 0.702 |
| washroom              | 0.619  | 0.634  | 0.547 | 0.500 | N.A   | 0.553 |

# Extreme edge test --img_size 320 --use_p2 (CPU numbers!)

| Dataset               | edge_n | edge_s | edge_m | edge_l |
|-----------------------|:------:|:------:|:------:|:------:|
| Solar panels          | 0.223  | 0.399  | 0.397 | 0.425 | 
| soccer players        | 0.733  | 0.812  | 0.769 | 0.780 | 
| chess pieces          | 0.955  | 0.978  | 0.980 | 0.983 | 
| circuit voltages      | 0.732  | 0.769  | 0.833 | 0.829 | 
| x-ray                 | 0.699  | 0.803  | 0.817 | 0.815 | 
| thermal dogs          | 0.827  | 0.916  | 0.943 | 0.918 | 
| street work           | 0.374  | 0.510  | 0.496 | 0.561 |
| activity diagrams     | 0.372  | 0.455  | 0.491 | 0.506 |
| sign language         | 0.897  | 0.929  | 0.907 | 0.936 | 
| marbels               | 0.825  | 0.887  | 0.871 | 0.887 |
| road-sign             | 0.916  | 0.936  | 0.936 | 0.929 |
| aerial-spheres        | 0.947  | 0.965  | 0.959 | 0.963 |


# Speed for 320 --p2
Hardware: 

CPU AMD Ryzen 5 5500
Image size input 640x480

No further optimization were done (simplify were skipped here but is supported), speeds were calculated with:
    python export/export_onnx.py --weights best_model_state.pt --img_size 320 
    python export/infer_onnx.py --img_dir "test\images" --img_size 320 --model edge_{x}_320.onnx
**Edge_n**

    === Inference timing (ms) ===
    pre_ms    mean 2.50 | std 1.39 | p50 2.41 | p90 2.68 | p95 2.87
    infer_ms  mean 7.31 | std 2.94 | p50 6.94 | p90 7.64 | p95 7.93
    post_ms   mean 0.73 | std 0.16 | p50 0.75 | p90 0.93 | p95 1.01
    total_ms  mean 10.54 | std 3.12 | p50 10.10 | p90 10.92 | p95 13.93
    Throughput ≈ 94.90 img/s
    Throughput ≈ 136.85 img/s (Model only)

**Edge_s**

    === Inference timing (ms) ===
    pre_ms    mean 2.35 | std 0.80 | p50 2.16 | p90 2.63 | p95 2.68
    infer_ms  mean 16.05 | std 2.96 | p50 16.19 | p90 17.71 | p95 17.87
    post_ms   mean 0.72 | std 0.18 | p50 0.73 | p90 0.97 | p95 1.06
    total_ms  mean 19.11 | std 2.98 | p50 19.16 | p90 20.93 | p95 21.66
    Throughput ≈ 52.32 img/s
    Throughput ≈ 62.30 img/s (Model only)


**Edge_m**

    === Inference timing (ms) ===
    pre_ms    mean 2.43 | std 0.71 | p50 2.46 | p90 2.68 | p95 2.87
    infer_ms  mean 22.70 | std 4.18 | p50 23.22 | p90 24.84 | p95 25.15
    post_ms   mean 0.76 | std 0.18 | p50 0.75 | p90 1.07 | p95 1.08
    total_ms  mean 25.89 | std 4.14 | p50 26.01 | p90 28.23 | p95 28.69
    Throughput ≈ 38.62 img/s
    Throughput ≈ 44.05 img/s (Model only)


**Edge_l**
    
    === Inference timing (ms) ===
    pre_ms    mean 2.44 | std 0.87 | p50 2.31 | p90 2.71 | p95 2.85
    infer_ms  mean 35.17 | std 6.32 | p50 35.97 | p90 38.54 | p95 39.12
    post_ms   mean 0.78 | std 0.15 | p50 0.80 | p90 0.93 | p95 1.09
    total_ms  mean 38.39 | std 6.42 | p50 39.53 | p90 41.79 | p95 42.11
    Throughput ≈ 26.05 img/s
    Throughput ≈ 28.43 img/s (Model only)


# Speed for 320 
Hardware: 

CPU AMD Ryzen 5 5500
The numbers below shows the numbers WITHOUT p2, due to a misstake during export, the P2 head was not included!

No further optimization were done (simplify were skipped here but is supported), speeds were calculated with:
    python export/export_onnx.py --weights best_model_state.pt --img_size 320 
    python export/infer_onnx.py --img_dir "test\images" --img_size 320 --model edge_{x}_320.onnx
**Edge_n**

    === Inference timing (ms) ===
    pre_ms    mean 3.38 | std 2.24 | p50 2.53 | p90 4.34 | p95 6.85
    infer_ms  mean 4.75 | std 1.07 | p50 4.37 | p90 6.02 | p95 6.62
    post_ms   mean 1.08 | std 1.28 | p50 0.68 | p90 1.64 | p95 3.00
    total_ms  mean 9.21 | std 3.74 | p50 7.83 | p90 12.85 | p95 16.55
    Throughput ≈ 108.59 img/s
    Throughput ≈ 210.38 img/s (Model only)

**Edge_s**

    === Inference timing (ms) ===
    pre_ms    mean 2.95 | std 1.57 | p50 2.61 | p90 3.07 | p95 4.76
    infer_ms  mean 9.80 | std 1.46 | p50 10.36 | p90 11.22 | p95 11.40
    post_ms   mean 0.70 | std 0.23 | p50 0.61 | p90 0.97 | p95 1.13
    total_ms  mean 13.45 | std 1.57 | p50 13.69 | p90 15.09 | p95 15.64
    Throughput ≈ 74.33 img/s
    Throughput ≈ 102.06 img/s (Model only)

**Edge_m**

    === Inference timing (ms) ===
    pre_ms    mean 3.29 | std 1.72 | p50 2.91 | p90 4.05 | p95 5.76
    infer_ms  mean 11.70 | std 2.73 | p50 10.62 | p90 16.11 | p95 17.51
    post_ms   mean 0.65 | std 0.14 | p50 0.60 | p90 0.87 | p95 0.91
    total_ms  mean 15.64 | std 3.29 | p50 14.18 | p90 20.41 | p95 21.53
    Throughput ≈ 63.94 img/s
    Throughput ≈ 85.49 img/s (Model only)

**Edge_l**

    === Inference timing (ms) ===
    pre_ms    mean 3.61 | std 1.97 | p50 3.03 | p90 4.50 | p95 6.72
    infer_ms  mean 16.25 | std 2.51 | p50 16.31 | p90 19.53 | p95 20.69
    post_ms   mean 0.77 | std 0.22 | p50 0.74 | p90 0.88 | p95 1.08
    total_ms  mean 20.62 | std 3.40 | p50 19.65 | p90 24.85 | p95 26.43
    Throughput ≈ 48.49 img/s
    Throughput ≈ 61.55 img/s (Model only)


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


| Device | edge_l | edge_m | edge_s | edge_n |
|------: |------: |------:  |-------:|------:|
| CPU    | 67.0ms  | 45.57ms    | 40.18ms   | 23.88ms  |
| GPU    | 24.95ms   | 23.05ms    | 20.83ms   | 20.2ms   | 

V2_Models  NaN == Not tested 

| Device | Yololite_n | Yololite_s | Yololite_m | Yololite_l |
|------: |------: |------:  |-------:|------:|
| CPU    | 87.05ms  | NaN    | 156.59ms   | NaN  |
| GPU    | 21.07ms   | NaN   | 25.65ms   | NaN   | 

# Params and flops
*Updates for all models will come shortly*

| Modell         |   Params (M) |     MACs (G) |    FLOPs (G)  |
| -------------- | -----------: | -----------: | -----------: | 
| **edge_n**     |  **0.553 M** |  **0.755 G** |  **1.511 G** | 
| **edge_s**     |  **2.359 M** |  **3.026 G** |  **6.053 G** |
| **edge_m**     |  **2.950 M** |  **3.870 G** |  **7.739 G** |    
| **yololite_n** |  **8.923 M** | **11.473 G** | **22.946 G** |     
| **yololite_m** | **17.916 M** | **27.239 G** | **54.478 G** |  
   

# Training logs

When training first a folder will be created runs/train 

Under train a subfolder x will be created 1, 2, 3 ....

When training starts a sanity_check.jpg is created, this show a 4x4 picuture of 4 images from the train_loader that gets sent to the model. 
This is a good way to see if labels were loaded correctly. 

merged_config.yaml is created as the complete configuration for the training, training parameters, model and dataset is saved here.

During validation steps last_b0.jpg and last_b1.jpg will be created these are two random images from the validation loop with the models predictions. 
You can check these to see the models progress. 


More information about plots are comming. 

































