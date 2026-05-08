# 🌿 AGRI-X AI — Explainable Multi-Model Deep Learning Framework for Plant Disease Detection

> **An end-to-end, research-grade framework combining transfer learning, explainable AI (Grad-CAM), and smart agricultural decision support for automated plant disease identification.**

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-3776AB?logo=python&logoColor=white)](#)
[![TensorFlow 2.21+](https://img.shields.io/badge/TensorFlow-2.21+-FF6F00?logo=tensorflow&logoColor=white)](#)
[![Streamlit 1.55+](https://img.shields.io/badge/Streamlit-1.55+-FF4B4B?logo=streamlit&logoColor=white)](#)
[![License: Research](https://img.shields.io/badge/License-Research%20%26%20Education-green)](#)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage Guide](#-usage-guide)
- [Model Architectures](#-model-architectures)
- [Research Methodology](#-research-methodology)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Dataset](#-dataset)
- [Results & Evaluation](#-results--evaluation)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [References](#-references)

---

## 🔍 Overview

AGRI-X AI is a comprehensive deep learning framework that addresses the critical challenge of early and accurate plant disease detection in agriculture. The framework provides:

- **Multi-architecture comparison** of 4 state-of-the-art CNN models (MobileNetV2, ResNet50, EfficientNetB0, DenseNet121) with optional Vision Transformer support
- **Explainable AI** through Gradient-weighted Class Activation Mapping (Grad-CAM) for transparent disease region identification
- **Disease severity estimation** using activation area analysis from Grad-CAM heatmaps
- **Smart agricultural recommendations** including pesticide, fungicide, irrigation, and fertilizer guidance
- **Research-grade evaluation** with K-fold cross-validation, ablation studies, statistical analysis, and compression studies

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-Model Training** | Two-phase transfer learning (frozen → fine-tuned) across 4+ architectures |
| 🔥 **Grad-CAM Explainability** | Visual explanations highlighting disease-affected regions on leaf images |
| 📊 **Research Analytics** | K-fold CV, ablation study, statistical significance tests, compression analysis |
| 🎯 **Disease Severity** | Heuristic severity estimation (Mild/Moderate/Severe) from Grad-CAM activation |
| 💊 **Smart Recommendations** | Per-disease agricultural guidance (pesticide, fungicide, irrigation, fertilizer) |
| 📈 **Interactive Dashboard** | Professional Streamlit app with model comparison, prediction, and research views |
| ⚙️ **Hyperparameter Tuning** | Keras Tuner integration for automated hyperparameter optimization |
| 🏗️ **Modular Architecture** | Clean separation of concerns: config → data → models → evaluation → visualization |

---

## 🏗️ Architecture

The framework follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard (app.py)                  │
├─────────────────────────────────────────────────────────────────┤
│                     API Layer (src/api/)                         │
│                 PlantDiseasePredictor                            │
├──────────────────────┬──────────────────────────────────────────┤
│  Pipelines Layer     │  Visualization Layer                     │
│  • Training          │  • Training Plots                        │
│  • Evaluation        │  • Evaluation Plots                      │
│  • Hyperparameter    │  • Grad-CAM Heatmaps                    │
├──────────────────────┼──────────────────────────────────────────┤
│  Models Layer        │  Evaluation Layer                        │
│  • BaseModelBuilder  │  • MetricsCalculator                     │
│  • MobileNetV2       │  • CrossValidator                        │
│  • ResNet50          │  • StatisticalAnalyzer                   │
│  • EfficientNetB0    │  • AblationStudy                         │
│  • DenseNet121       │  • CompressionStudy                      │
│  • ViT (optional)    │                                          │
├──────────────────────┴──────────────────────────────────────────┤
│                     Data Layer (src/data/)                       │
│              DatasetManager — Loading, Splitting, Augmentation   │
├─────────────────────────────────────────────────────────────────┤
│  Config (src/config/)  │  Entity (src/entity/)  │  Utils         │
│  Settings dataclasses  │  Result dataclasses    │  Logger, Seed  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
AI_Farming_Project_Final/
│
├── src/                              # Core framework modules
│   ├── config/
│   │   └── settings.py               # Central configuration dataclass (all hyperparams)
│   ├── entity/
│   │   └── model_entity.py           # TrainingResult, ModelMetrics, CVResult, AblationResult
│   ├── data/
│   │   └── dataset.py                # DatasetManager: loading, splitting, augmentation
│   ├── models/
│   │   ├── base_model.py             # Abstract BaseModelBuilder (transfer learning interface)
│   │   ├── mobilenet.py              # MobileNetV2 builder
│   │   ├── resnet.py                 # ResNet50 builder
│   │   ├── efficientnet.py           # EfficientNetB0 builder
│   │   ├── densenet.py               # DenseNet121 builder
│   │   ├── vit_model.py              # Vision Transformer builder (optional)
│   │   └── model_factory.py          # Factory pattern: get_model(name) → keras.Model
│   ├── evaluation/
│   │   ├── metrics.py                # Accuracy, F1, AUC, ROC, PR, confusion matrix
│   │   ├── cross_validation.py       # K-Fold stratified cross-validation
│   │   ├── statistical_analysis.py   # CI, paired t-tests, comparison tables
│   │   ├── ablation_study.py         # base → +augmentation → +fine-tuning
│   │   └── compression_study.py      # Size, FLOPs, speed benchmarks
│   ├── visualization/
│   │   ├── training_plots.py         # Accuracy/loss curves, model comparison
│   │   ├── evaluation_plots.py       # Confusion matrix, ROC, PR, bar charts
│   │   └── gradcam.py                # Grad-CAM heatmap generation & overlay
│   ├── pipelines/
│   │   ├── training_pipeline.py      # Full training orchestrator
│   │   ├── evaluation_pipeline.py    # Post-training evaluation orchestrator
│   │   └── hyperparameter_pipeline.py # Keras Tuner hyperparameter search
│   ├── api/
│   │   └── predictor.py              # PlantDiseasePredictor: predict → explain → recommend
│   └── utils/
│       ├── logger.py                 # Professional logging (file + console)
│       ├── helpers.py                # Seed, Timer, JSON/CSV I/O
│       └── recommendations.py        # Rule-based agricultural recommendation engine
│
├── app.py                            # Streamlit dashboard (4 pages)
├── app_utils.py                      # Dashboard CSS & helper functions
├── train.py                          # CLI: python train.py
├── evaluate.py                       # CLI: python evaluate.py
│
├── PlantVillage/                     # Dataset (15 class directories)
├── saved_models/                     # Trained model checkpoints (.keras)
├── reports/                          # Generated evaluation outputs
│   ├── graphs/                       # Training history plots
│   ├── gradcam/                      # Grad-CAM sample visualizations
│   ├── confusion_matrix/             # Per-model confusion matrices
│   ├── roc_curves/                   # ROC curve plots
│   ├── metrics/                      # CSV metric reports
│   └── summary/                      # Dataset summary JSON
├── logs/                             # Training log files
├── research_bundle/                  # Model registry & research figures
│
├── .env                              # Environment configuration
├── .streamlit/config.toml            # Streamlit theme configuration
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🚀 Setup & Installation

### Prerequisites

- **Python 3.10+** (tested with Python 3.13.11)
- **pip** package manager
- **8GB+ RAM** recommended for training
- **GPU (optional)**: CUDA-compatible GPU for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/raj06872436/AI_Farming_Project.git
cd AI_Farming_Project

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The PlantVillage dataset should be placed in the `PlantVillage/` directory with the following structure:

```
PlantVillage/
├── Pepper__bell___Bacterial_spot/    # 997 images
├── Pepper__bell___healthy/           # 1,478 images
├── Potato___Early_blight/            # 1,000 images
├── Potato___Late_blight/             # 1,000 images
├── Potato___healthy/                 # 152 images
├── Tomato_Bacterial_spot/            # 2,127 images
├── Tomato_Early_blight/              # 1,000 images
├── Tomato_Late_blight/               # 1,909 images
├── Tomato_Leaf_Mold/                 # 952 images
├── Tomato_Septoria_leaf_spot/        # 1,771 images
├── Tomato_Spider_mites_Two_spotted_spider_mite/  # 1,676 images
├── Tomato__Target_Spot/              # 1,404 images
├── Tomato__Tomato_YellowLeaf__Curl_Virus/  # 3,208 images
├── Tomato__Tomato_mosaic_virus/      # 373 images
└── Tomato_healthy/                   # 1,591 images
```

**Total: 20,638 images across 15 classes (3 plants × 5 categories)**

---

## 📖 Usage Guide

### 1. Train All Models

```bash
python train.py
```

This will:
- Load and preprocess the PlantVillage dataset (70/15/15 split)
- For each model (MobileNetV2, ResNet50, EfficientNetB0, DenseNet121):
  - **Phase 1**: Train classifier head only (frozen backbone, 5 epochs)
  - **Phase 2**: Fine-tune top layers (unfrozen, 15 epochs)
  - Evaluate on test set and compute all metrics
  - Generate training history plots and Grad-CAM samples
- Save models to `saved_models/` and reports to `reports/`

### 2. Run Advanced Evaluation

```bash
python evaluate.py
```

This will:
- Load saved models and compute comprehensive metrics
- Run 5-fold stratified cross-validation
- Perform statistical analysis with 95% confidence intervals
- Execute ablation study (base → +augmentation → +fine-tuning)
- Run compression study (size vs speed vs accuracy tradeoffs)
- Perform hyperparameter tuning via Keras Tuner

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

Opens the interactive Streamlit dashboard at `http://localhost:8501`.

### 4. Use the Prediction API Programmatically

```python
from src.config.settings import Config
from src.api.predictor import PlantDiseasePredictor
from PIL import Image

config = Config()
predictor = PlantDiseasePredictor(config)

# Predict
image = Image.open("leaf.jpg")
result = predictor.predict(image, model_name="MobileNetV2", top_k=3)

print(f"Disease: {result['display_name']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Severity: {result['severity']}")
print(f"Recommendation: {result['recommendation']}")
```

---

## 🧠 Model Architectures

### Comparison Table

| Architecture | Parameters | Size | Inference | Key Innovation |
|-------------|-----------|------|-----------|---------------|
| **MobileNetV2** | ~2.3M | 13.4 MB | ~25 ms | Depthwise separable convolutions, inverted residuals |
| **ResNet50** | ~25.6M | 97.1 MB | ~65 ms | Skip connections for deep residual learning |
| **EfficientNetB0** | ~5.3M | 20.5 MB | ~35 ms | Compound scaling (depth × width × resolution) |
| **DenseNet121** | ~8.1M | 31.2 MB | ~45 ms | Dense connectivity for feature reuse |
| **ViT** (optional) | ~6M | ~24 MB | ~50 ms | Self-attention mechanism for global context |

### Transfer Learning Pipeline

```
ImageNet Pretrained Backbone (frozen)
        ↓
GlobalAveragePooling2D
        ↓
BatchNormalization
        ↓
Dense(256, ReLU) → Dropout(0.3)
        ↓
Dense(128, ReLU) → Dropout(0.3)
        ↓
Dense(15, Softmax) → Output Predictions
```

**Training phases:**
1. **Phase 1** (5 epochs): Train only the classifier head (backbone frozen), LR = 1e-4
2. **Phase 2** (15 epochs): Fine-tune top 50 layers of backbone, LR = 1e-5

---

## 🔬 Research Methodology

### 1. Data Augmentation
- Random rotation (±30°)
- Horizontal flip
- Zoom (±20%)
- Brightness adjustment (0.8–1.2×)
- Width/height shift (±15%)

### 2. Evaluation Metrics
- **Per-model**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Per-class**: Precision, Recall, F1 (15-class breakdown)
- **Visualization**: Confusion matrix, ROC curves (one-vs-rest), Precision-Recall curves

### 3. K-Fold Cross-Validation
- 5-fold stratified cross-validation
- Reports mean ± standard deviation for all metrics
- Ensures statistical robustness of results

### 4. Statistical Analysis
- 95% confidence intervals for accuracy, precision, recall, F1
- Paired t-tests between model performances
- Inter-model comparison tables

### 5. Ablation Study
- **Base**: No augmentation, frozen backbone only
- **+Augmentation**: Added data augmentation
- **+Fine-tuning**: Unfroze top backbone layers
- Measures incremental contribution of each component

### 6. Compression Study
- Model parameter count comparison
- File size analysis (MB)
- Inference speed benchmarking (ms per image)
- Accuracy vs. size tradeoff visualization
- Deployment efficiency scoring

### 7. Explainability (Grad-CAM)
- Gradient-weighted Class Activation Mapping
- Heatmap overlay on original leaf images
- Disease activation area percentage estimation
- Severity classification: Mild (<35%), Moderate (35-65%), Severe (>65%)

---

## 🖥️ Streamlit Dashboard

The dashboard features 4 interactive pages:

### 🔬 Disease Detection
- Upload leaf image for AI-powered diagnosis
- Model selector (choose from trained architectures)
- Prediction with confidence score and top-5 classes
- Grad-CAM explainability visualization
- Disease severity estimation
- Agricultural recommendations (pesticide, fungicide, irrigation, fertilizer)

### 📊 Research Dashboard
- Training history plots (accuracy/loss curves)
- Grad-CAM gallery with sample visualizations
- Browse all generated evaluation reports

### 📈 Model Comparison
- Multi-model comparison table
- Interactive Plotly bar charts (accuracy & model size)
- Architecture analysis with strengths/weaknesses
- Multi-metric radar chart

### 📋 Dataset Explorer
- PlantVillage dataset statistics
- Interactive class distribution chart
- Per-class breakdown with percentages

---

## 🌱 Dataset

**PlantVillage Dataset** — A curated collection of plant leaf images for disease classification.

| Plant | Classes | Total Images |
|-------|---------|-------------|
| Bell Pepper | Bacterial Spot, Healthy | 2,475 |
| Potato | Early Blight, Late Blight, Healthy | 2,152 |
| Tomato | 8 diseases + Healthy (10 classes) | 16,011 |
| **Total** | **15 classes** | **20,638 images** |

**Preprocessing**: All images resized to 224×224×3, normalized to [0, 1] range.

---

## ⚙️ Configuration

All settings are configurable via the `.env` file:

```env
# Dataset
DATASET_PATH=PlantVillage
IMAGE_SIZE=224
BATCH_SIZE=32
NUM_CLASSES=15

# Training
INITIAL_EPOCHS=5
FINE_TUNE_EPOCHS=15
LEARNING_RATE=0.0001
FINE_TUNE_LR=0.00001
SEED=42

# Models (comma-separated)
MODELS=MobileNetV2,ResNet50,EfficientNetB0,DenseNet121

# Cross Validation
K_FOLDS=5
```

---

## 📊 Results & Evaluation

After training, results are saved to:

| Output | Location | Format |
|--------|----------|--------|
| Trained models | `saved_models/` | `.keras` checkpoints |
| Training plots | `reports/graphs/` | PNG images |
| Confusion matrices | `reports/confusion_matrix/` | PNG images |
| ROC curves | `reports/roc_curves/` | PNG images |
| Metric tables | `reports/metrics/` | CSV files |
| Grad-CAM samples | `reports/gradcam/` | PNG images |
| Dataset summary | `reports/summary/` | JSON |
| Training logs | `logs/` | `.log` files |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit changes (`git commit -m "Add new model architecture"`)
4. Push to the branch (`git push origin feature/new-model`)
5. Open a Pull Request

---

## 📚 References

1. **MobileNetV2**: Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR 2018*.
2. **ResNet**: He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*.
3. **EfficientNet**: Tan, M. & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *ICML 2019*.
4. **DenseNet**: Huang, G., et al. (2017). "Densely Connected Convolutional Networks." *CVPR 2017*.
5. **Grad-CAM**: Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV 2017*.
6. **PlantVillage**: Hughes, D. P. & Salathe, M. (2015). "An open access repository of images on plant health to enable the development of mobile disease diagnostics."
7. **Vision Transformer**: Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words." *ICLR 2021*.

---

## 📄 License

This project is released for **research and educational purposes only**. The PlantVillage dataset is subject to its own licensing terms.

---

<p align="center">
  <strong>🌿 AGRI-X AI — Empowering Agriculture with Explainable AI</strong><br>
  <em>Built with TensorFlow, Keras, and Streamlit</em>
</p>
