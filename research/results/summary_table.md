| Model | Method | Accuracy (sample) | Attention on lesions | on healthy leaf | on background | Lesion share of leaf | Lesion focus ratio | Pointing-game hit rate | IoU (CAM>0.5 vs lesion) | Spearman(CAM area, lesion area) | CAM 'affected area' on healthy leaves (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | Grad-CAM | 0.920 | 0.107 | 0.619 | 0.275 | 0.145 | 0.950 | 0.511 | 0.111 | 0.063 | 78.878 |
| MobileNetV2 | Grad-CAM++ | 0.920 | 0.104 | 0.588 | 0.308 | 0.145 | 0.969 | 0.539 | 0.118 | -0.026 | 86.360 |
| ResNet50 | Grad-CAM | 0.997 | 0.105 | 0.545 | 0.350 | 0.145 | 1.074 | 0.642 | 0.119 | 0.125 | 88.709 |
| ResNet50 | Grad-CAM++ | 0.997 | 0.096 | 0.507 | 0.397 | 0.145 | 1.073 | 0.687 | 0.129 | 0.041 | 97.681 |
| EfficientNetB0 | Grad-CAM | 0.957 | 0.114 | 0.551 | 0.335 | 0.145 | 1.089 | 0.618 | 0.129 | 0.049 | 79.314 |
| EfficientNetB0 | Grad-CAM++ | 0.957 | 0.131 | 0.612 | 0.257 | 0.145 | 1.203 | 0.659 | 0.142 | 0.015 | 66.123 |
| DenseNet121 | Grad-CAM | 0.962 | 0.101 | 0.528 | 0.370 | 0.145 | 1.066 | 0.630 | 0.117 | -0.036 | 84.788 |
| DenseNet121 | Grad-CAM++ | 0.962 | 0.078 | 0.423 | 0.499 | 0.145 | 1.039 | 0.647 | 0.089 | -0.106 | 100.000 |
