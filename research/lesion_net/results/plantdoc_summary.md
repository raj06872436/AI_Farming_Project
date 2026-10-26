PlantDoc field photos, 1012 images, 12 classes shared with PlantVillage. Chance: 15-way 0.067, 12-way 0.083. Answering 'diseased' every time scores 0.885 on 'Healthy vs diseased'.

| Model | Accuracy (15-way) | Accuracy (12 PlantDoc classes) | Macro-F1 (12) | Right crop | Healthy vs diseased | Lesion % healthy leaves | Lesion % diseased leaves |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | 0.236 | 0.236 | 0.189 | 0.689 | 0.889 | — | — |
| ResNet50 | 0.196 | 0.196 | 0.147 | 0.652 | 0.885 | — | — |
| EfficientNetB0 | 0.202 | 0.202 | 0.148 | 0.689 | 0.875 | — | — |
| DenseNet121 | 0.201 | 0.201 | 0.158 | 0.674 | 0.876 | — | — |
| LesionNet_A3_full | 0.207 | 0.208 | 0.172 | 0.707 | 0.891 | 16.532 | 24.715 |
