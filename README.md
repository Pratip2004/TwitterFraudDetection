# TwitterFraudDetection

# EfficientTruthBOOSTING v2 — Cross-Attention Multimodal Fake News Detection

A multimodal deep learning pipeline for detecting fake news using **cross-attention fusion** of text, image, URL, and user engagement features. Built on the [Fakeddit](https://github.com/entitize/Fakeddit) dataset.

## Architecture

```
                    ┌──────────────┐
                    │   Raw Post   │
                    └──────┬───────┘
           ┌───────────────┼───────────────┬──────────────┐
           ▼               ▼               ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌────────────┐ ┌───────────┐
    │  Text (TF-  │ │ Image (Eff- │ │  URL/Link  │ │  Context  │
    │  IDF 2000)  │ │ NetV2B0     │ │  Features  │ │ Engagement│
    │             │ │  128x128)   │ │  (8-dim)   │ │  (4-dim)  │
    └──────┬──────┘ └──────┬──────┘ └─────┬──────┘ └─────┬─────┘
           │               │              │              │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐      │
    │  Modality   │ │  Modality   │ │  Modality  │      │
    │  Projection │ │  Projection │ │  Projection│      │
    │  (128-dim)  │ │  (128-dim)  │ │  (128-dim) │      │
    └──────┬──────┘ └──────┬──────┘ └─────┬──────┘      │
           │               │              │              │
           └───────────────┼──────────────┘              │
                           ▼                             │
                ┌─────────────────────┐                  │
                │  Multi-Head Cross-  │                  │
                │  Attention Fusion   │         ┌────────▼────────┐
                │  (3 heads, 128-dim) │         │  User Engagement│
                └──────────┬──────────┘         │  Embedding      │
                           │                    │  (128-dim)      │
                           │                    └────────┬────────┘
                           └──────────┬─────────────────┘
                                      ▼
                              ┌──────────────┐
                              │ FC 256 → 128 │
                              │   + Dropout  │
                              └──────┬───────┘
                                     ▼
                              ┌──────────────┐
                              │   Sigmoid    │
                              │  Real / Fake │
                              └──────────────┘
```

## Features

| Feature | Details |
|---|---|
| **Text** | TF-IDF vectorization (2000 features) |
| **Image** | EfficientNetV2B0 embeddings at 128×128, cached to Drive |
| **URL** | 8 handcrafted features (length, HTTPS, shortener detection, etc.) |
| **Context** | Hashtag count, mentions, text length, has_url, engagement score |
| **Fusion** | 3-head multi-head attention over projected modalities |

## Accuracy Improvements

- **Class weights** — balances minority class misclassification
- **Label smoothing (0.1)** — prevents overconfident predictions
- **LR warmup (3 epochs) + cosine decay** — smoother convergence
- **Early stopping** on validation AUC with best weight restoration

## Experiments

Three train/test/val split configurations are run and compared:

| Split | Train | Test | Val |
|---|---|---|---|
| 70:20:10 | 70% | 20% | 10% |
| 75:15:10 | 75% | 15% | 10% |
| 80:10:10 | 80% | 10% | 10% |

Additionally, **Content-Only vs Content+Context** models are compared to quantify the contribution of engagement features.

## Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1 (binary, micro, macro), AUC-ROC
- **Regression**: MSE, MAE, RMSE
- **Statistical**: Chi-Square test of independence on confusion matrix
- **Per-class**: All metrics broken down for Real (0) and Fake (1)

## Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Setup and Imports | Install dependencies, GPU configuration |
| 2 | Configuration | All hyperparameters and paths |
| 3 | Load Dataset | Mount Drive, load Fakeddit CSV |
| 4 | Data Preprocessing | Text cleaning, label encoding, filtering |
| 5 | Feature Extraction | TF-IDF, EfficientNetV2B0 embeddings, URL features, context features |
| 6 | Model Architecture | ModalityProjection, AttentionFusionLayer, UserEngagementEmbedding |
| 7 | Helper Functions | Splitting, training, evaluation, saving utilities |
| 8-10 | Experiments 1-3 | Train/load models for all 3 splits |
| 11 | Results Tables | Test/Val comparison, split sizes, per-class metrics (Real/Fake) |
| 12 | Chi-Square Test | Statistical significance of predictions |
| 13 | Confusion Matrices | 2×3 grid for all splits |
| 14 | ROC Curves | Overlay comparison across splits |
| 15 | Training History | Accuracy and loss curves |
| 16 | Final Summary | Combined metrics table, CSV export |
| 17-21 | Content vs Context | Content-only model, comparison tables, visualisations |

## Setup

### Requirements

```
tensorflow>=2.15
scikit-learn
pandas
numpy
matplotlib
Pillow
requests
tabulate
scipy
```

### Platform

| Platform | GPU | How to set |
|---|---|---|
| **Google Colab** | T4 / A100 | `Runtime → Change runtime type → GPU` |
| **Kaggle** | T4 x2 | `Settings → Accelerator → GPU T4 x2` |

### Configuration

Key hyperparameters in Section 2:

```python
D_MODEL = 128        # Projection dimension
NUM_HEADS = 3        # Attention heads
TFIDF_MAX = 2000     # TF-IDF vocabulary size
IMG_SIZE = 128       # Image resolution
BATCH_SIZE = 256     # Training batch size
EPOCHS = 50          # Max epochs (with early stopping)
LR = 0.001           # Base learning rate
```

### Paths

```python
DRIVE_DIR = "/content/drive/MyDrive"
EMBEDDINGS_PATH = "<DRIVE_DIR>/image_embeddings_fakeddit_128.npy"
MODEL_SAVE_DIR = "<DRIVE_DIR>/truthboosting_v2_models/"
```

## Usage

1. Upload `fakeddit_unified.csv` to Google Drive (`MyDrive/`)
2. Open `Cross_Attention_v2_Unified_Colab.ipynb` in Colab
3. Set runtime to GPU (T4 or A100)
4. **Run All** — first run computes and caches image embeddings (~30-60 min), subsequent runs load from cache (~2 min)

### Saved Outputs

```
truthboosting_v2_models/
├── truthboosting_v2_70_20_10.pkl     # Model weights + config
├── truthboosting_v2_75_15_10.pkl
├── truthboosting_v2_80_10_10.pkl
├── truthboosting_v2_content_only.pkl
├── confusion_matrices_all.png
├── roc_comparison_all.png
├── training_history_all.png
├── performance_comparison_all_splits.png
├── f1_variants_comparison.png
├── chi_square_comparison.png
└── results_comparison_all.csv
```

## Model Loading (Inference)

```python
import pickle
from tensorflow import keras

# Load saved model
with open("truthboosting_v2_70_20_10.pkl", "rb") as f:
    pkl_data = pickle.load(f)

model = build_model()  # Rebuild architecture
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.set_weights(pkl_data['model_weights'])

# Also available: pkl_data['tfidf_vectorizer'], pkl_data['link_scaler'], pkl_data['context_scaler']
```

## Dataset

| Field | Description |
|---|---|
| `id` | Unique post identifier |
| `text` | Post text content |
| `url` | Source URL |
| `image_url` | Associated image URL |
| `hashtag_count` | Number of hashtags |
| `mentions_count` | Number of user mentions |
| `text_length` | Character count of text |
| `has_url` | Binary: contains URL |
| `engagement_score` | User engagement metric |
| `label` | 0 = Real, 1 = Fake |

**Size**: ~700K posts after preprocessing

## License

Academic use only. Based on the Fakeddit dataset.
