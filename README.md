# 🌿 AGRI-X AI — Plant Disease Detection Framework

> **An Explainable Multi-Model Deep Learning Framework for Plant Disease Detection and Smart Agricultural Decision Support**

## 🏗️ Architecture

```
AI_Farming_Project/
├── src/
│   ├── config/settings.py        # Central configuration
│   ├── entity/model_entity.py    # Data classes for results
│   ├── data/dataset.py           # Dataset management & augmentation
│   ├── models/                   # Model builders (MobileNetV2, ResNet50, EfficientNetB0, DenseNet121, ViT)
│   ├── evaluation/               # Metrics, CV, statistical analysis, ablation, compression
│   ├── visualization/            # Training plots, evaluation plots, Grad-CAM
│   ├── pipelines/                # Training, evaluation, hyperparameter tuning orchestrators
│   ├── api/predictor.py          # High-level prediction API
│   └── utils/                    # Logger, helpers, recommendation engine
├── train.py                      # CLI: Train all models
├── evaluate.py                   # CLI: Advanced evaluation (CV, ablation, stats)
├── app.py                        # Streamlit dashboard
├── requirements.txt
└── .env                          # Configuration
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all models (MobileNetV2, ResNet50, EfficientNetB0, DenseNet121)
python train.py

# 3. Run advanced evaluation (K-Fold CV, ablation, stats, compression)
python evaluate.py

# 4. Launch the interactive dashboard
streamlit run app.py
```

## 📊 Models
| Model | Type | Params | Key Strength |
|-------|------|--------|-------------|
| MobileNetV2 | CNN | ~3.5M | Lightweight, mobile-ready |
| ResNet50 | CNN | ~25.6M | Deep residual learning |
| EfficientNetB0 | CNN | ~5.3M | Optimal accuracy/efficiency |
| DenseNet121 | CNN | ~8.1M | Feature reuse |
| ViT | Transformer | ~6M | Attention-based (optional) |

## 📈 Research Features
- **K-Fold Cross Validation** (5-fold stratified)
- **Statistical Analysis** (confidence intervals, paired t-tests)
- **Ablation Study** (base → +augmentation → +fine-tuning)
- **Compression Study** (size vs speed vs accuracy)
- **Grad-CAM Explainability** (disease region visualization)
- **Disease Severity Estimation** (mild/moderate/severe)
- **Recommendation Engine** (pesticide, fungicide, irrigation, fertilizer)

## 🌱 Dataset
PlantVillage dataset — 15 classes across Tomato, Potato, and Bell Pepper (healthy + diseased).

## 📄 License
Research and educational use.
