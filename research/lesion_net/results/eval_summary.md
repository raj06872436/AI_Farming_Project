| Model | Explanation | Accuracy | Macro-F1 | Background + leaf silhouette acc. | Background-only acc. (chance 0.067) | Acc. grey background | Acc. swapped background | Explanation on lesions | Explanation on background | Lesion focus | Pointing hit | IoU | Severity ρ | Lesion % on healthy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | Grad-CAM++ | 0.930 | 0.919 | 0.255 | 0.191 | 0.760 | 0.716 | 0.094 | 0.288 | 0.960 | 0.492 | 0.109 | — | — |
| ResNet50 | Grad-CAM++ | 0.998 | 0.998 | 0.336 | 0.374 | 0.451 | 0.600 | 0.087 | 0.381 | 1.060 | 0.636 | 0.117 | — | — |
| EfficientNetB0 | Grad-CAM++ | 0.964 | 0.963 | 0.262 | 0.178 | 0.812 | 0.623 | 0.121 | 0.247 | 1.220 | 0.630 | 0.133 | — | — |
| DenseNet121 | Grad-CAM++ | 0.962 | 0.962 | 0.301 | 0.289 | 0.564 | 0.576 | 0.071 | 0.487 | 1.033 | 0.604 | 0.081 | — | — |
| LesionNet_A0_plain | own lesion map | 0.996 | 0.997 | 0.472 | 0.327 | 0.675 | 0.637 | 0.055 | 0.623 | 1.092 | 0.534 | 0.048 | 0.268 | 55.402 |
| LesionNet_A3_full | own lesion map | 0.995 | 0.995 | 0.272 | 0.195 | 0.987 | 0.818 | 0.441 | 0.067 | 6.060 | 0.971 | 0.505 | 0.900 | 0.546 |
| LesionNet_A5_mobilenet | own lesion map | 0.960 | 0.959 | 0.265 | 0.203 | 0.942 | 0.732 | 0.340 | 0.082 | 4.121 | 0.949 | 0.410 | 0.784 | 1.339 |
