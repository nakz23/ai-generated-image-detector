# 🎯 AI-Based Real vs Fake Image Detector

A deep learning system that distinguishes between real and AI-generated images using transfer learning with PyTorch and ResNet18. Optimized for CPU training with a reduced, balanced dataset from Kaggle's 140k Real and Fake Faces dataset.

---

## 📋 Project Overview

### Problem Statement
With the rapid advancement of generative AI models (StyleGAN, Stable Diffusion, DALL·E, etc.), distinguishing between real and synthetic images has become increasingly difficult.

### Objective
Build a binary classifier that automatically detects whether an image is:
- **Real** (authentic photograph)
- **AI-Generated** (synthetic/artificially created)

### Target Applications
- Digital forensics
- Synthetic media detection
- Cybersecurity
- Content authentication

---

## 🚀 Key Achievements

| Phase | Improvement | Impact |
|-------|------------|--------|
| **Dataset Optimization** | Reduced 140k images → 30k balanced dataset | Reduced CPU training time from hours → manageable duration |
| **Baseline Model** | ResNet18 (frozen layers) | Initial training accuracy: ~73% |
| **Fine-Tuning** | Unfroze layer3 + layer4 | Training accuracy boosted to **~98.5%** |
| **Architecture** | Added intermediate FC layer + Dropout | Better feature representation |
| **Validation** | Model checkpointing + validation tracking | Proper generalization checks |

---

## 🛠️ Technical Stack

### Environment
- **Language**: Python 3.x
- **Framework**: PyTorch
- **Development**: VS Code + Virtual Environment
- **Hardware**: CPU-only (no CUDA required)

### Libraries
- `torch` & `torchvision` - Deep learning & computer vision
- `scikit-learn` - Metrics & utilities
- `matplotlib` - Visualization
- `streamlit` - Planned for deployment

---

## 📊 Dataset

### Original Dataset
- **Source**: 140k Real and Fake Faces (Kaggle)
- **Size**: ~100k training | ~20k validation | ~20k test

### Optimized Dataset (Used in Training)
- **Reduction Strategy**: Random sampling per class
- **Final Size**: ~30,000 total images
  - 5,000 real images
  - 5,000 fake images
  - Per split (train, validation, test)
- **Directory Structure**:
  ```
  data/small_dataset/
  ├── train/
  │   ├── real/
  │   └── fake/
  ├── valid/
  │   ├── real/
  │   └── fake/
  └── test/
      ├── real/
      └── fake/
  ```

### Data Preprocessing Pipeline
- **Resize**: 128×128 pixels
- **Augmentation**: Random horizontal flip
- **Normalization**: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
- **Batch Size**: 16 (optimized for CPU)

---

## 🧠 Model Architecture

### Baseline (Phase 1)
- **Base**: ResNet18 pretrained on ImageNet
- **Training Strategy**: Frozen all layers except final FC
- **Result**: Underfitting (~73% accuracy)

### Fine-Tuned (Phase 2 - Current)
- **Base**: ResNet18 pretrained on ImageNet
- **Unfrozen Layers**: layer3 + layer4
- **Custom Head**:
  ```
  512 → 128 → 2
  ReLU + Dropout
  ```
- **Optimizer**: Adam with reduced learning rate (lr=0.0001)
- **Loss**: CrossEntropyLoss
- **Result**: **~98.5% training accuracy**

### Why This Works
1. **Transfer Learning**: Leverages ImageNet features for image classification
2. **Selective Fine-Tuning**: Balance between pre-trained knowledge and task-specific adaptation
3. **Dropout**: Reduces overfitting by preventing co-adaptation of neurons
4. **Lower Learning Rate**: Preserves useful pre-trained features while learning synthetic artifacts

---

## 📁 Project Structure

```
ai-image-detector/
├── README.md                          # Project documentation
├── main.py                            # Data loader verification script
├── train.py                           # Main training script with validation
├── evaluate.py                        # Test evaluation with classification report
├── streamlit_app.py                   # Interactive web UI for image classification
├── gpu_test.py                        # GPU availability check
├── debug_infer.py                     # Inference debugging script
│
├── models/
│   └── resnet_model.py               # ResNet18 model definition & fine-tuning
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py                    # Data loading pipeline (ImageFolder + transforms)
│   └── reduce_dataset.py             # Dataset reduction utility
│
├── data/
│   ├── train.csv                     # Training metadata (if available)
│   ├── valid.csv                     # Validation metadata (if available)
│   ├── test.csv                      # Test metadata (if available)
│   │
│   ├── small_dataset/                # Optimized reduced dataset
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   │
│   └── real_vs_fake/                 # Original full dataset (optional)
│       └── real-vs-fake/
│           ├── train/
│           ├── valid/
│           └── test/
│
├── best_model.pth                    # Saved best model checkpoint
├── venv/                             # Python virtual environment
└── archive.zip                       # Dataset archive (if downloaded)
```

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- pip or conda

### 1. Clone Repository
```bash
git clone <repository_url>
cd ai-image-detector
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate       # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install torch torchvision scikit-learn matplotlib streamlit
```

### 4. Prepare Dataset
```bash
# If using dataset reduction (recommended for CPU)
python utils/reduce_dataset.py

# Or download full dataset from Kaggle:
# https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
```

---

## 🏃 Running the Project

### 1. Verify Data Loaders
```bash
python main.py
```
**Output**: Displays number of batches in train, validation, and test sets.

### 2. Train the Model
```bash
python train.py
```
**Output**: Epoch-by-epoch training/validation accuracy metrics. Saves `best_model.pth` when validation improves.

### 3. Evaluate on Test Set
```bash
python evaluate.py
```
**Output**: Test accuracy, classification report (Precision/Recall/F1), and confusion matrix with correct label mappings.

### 4. Launch Interactive Web App (Streamlit)
```bash
streamlit run streamlit_app.py
```
**Output**: Opens interactive UI at `http://localhost:8501`. Upload images and get real-time predictions with confidence scores.

### 5. Check GPU/CPU
```bash
python gpu_test.py
```
**Output**: Verifies PyTorch and CUDA availability.

---

## 📈 Training Results

### Training Configuration
- **Epochs**: 8
- **Batch Size**: 16
- **Learning Rate**: 0.0001 (Adam optimizer)
- **Loss Function**: CrossEntropyLoss
- **Device**: CPU

### Performance Metrics
- **Final Training Accuracy**: ~98.5%
- **Validation Integration**: Best model checkpoint saved
- **Convergence**: Stable across epochs after fine-tuning

### Key Observations
1. Fine-tuning deeper layers significantly improved accuracy
2. Model successfully learned synthetic artifacts and real image features
3. Transfer learning proved highly effective for this classification task
4. Reduced dataset maintained balanced class distribution

---

## ✅ Completed Features

- [x] Evaluate on test dataset
- [x] Generate confusion matrix & classification report (via `evaluate.py`)
- [x] Compute Precision, Recall, F1-score
- [x] Build Streamlit demo application (`streamlit_app.py`)
- [x] Fix label mapping (single source-of-truth from ImageFolder)
- [x] Model checkpointing & validation tracking
- [x] Data augmentation pipeline

## 🔮 Next Steps (Optional)

- [ ] Plot ROC curve & AUC-ROC
- [ ] Deploy as web service
- [ ] Create detailed project report & findings
- [ ] Experiment with additional architectures (ResNet50, EfficientNet)
- [ ] Implement real-time webcam inference

---

## 💡 Core Concepts

### Transfer Learning
Leveraging pre-trained models (ResNet18 on ImageNet) to accelerate training on a new classification task.

### Fine-Tuning
Unfreezing deeper layers to adapt the model to synthetic artifact detection while preserving low-level features.

### Data Augmentation
Random flipping and normalization to improve model robustness and reduce overfitting.

### Binary Classification
Two-class problem: Real vs Fake (CrossEntropyLoss with softmax).

### Model Checkpointing
Saving the best model during validation to capture optimal generalization performance.

---

## 📝 File Descriptions

### Main Scripts
- **`train.py`**: Full training loop with validation. Iterates through epochs, computes loss/accuracy, tracks best validation performance, and saves `best_model.pth`.
- **`evaluate.py`**: Loads the best model and evaluates on test set. Generates classification report, confusion matrix, and test accuracy with correct label mappings.
- **`streamlit_app.py`**: Interactive web application for uploading and classifying images. Uses cached model loading and derives labels from ImageFolder for consistent predictions.
- **`main.py`**: Quick verification script to check data loader functionality and batch statistics.
- **`gpu_test.py`**: Checks PyTorch installation and GPU/CPU availability.
- **`debug_infer.py`**: Debugging utility to verify model inference and label mappings on test images.

### Model & Utils
- **`models/resnet_model.py`**: Defines ResNet18 with selective layer unfreezing (layer3 + layer4) and custom binary classification head.
- **`utils/dataset.py`**: Implements `get_dataloaders()` function for loading data with augmentation and normalization. Uses torchvision's `ImageFolder`.
- **`utils/reduce_dataset.py`**: Utility to randomly sample images from the original dataset for balanced CPU training.

---

## 🔧 Key Technical Implementation Details

### Label Mapping Fix (Source-of-Truth)
The project uses `torchvision.datasets.ImageFolder` which automatically maps class folders (`real`/`fake`) to indices. To ensure consistent predictions across training, evaluation, and the web UI:

```python
dataset = datasets.ImageFolder(root="data/small_dataset/train")
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
# Result: {0: 'fake', 1: 'real'}

display_map = {"real": "Real", "fake": "AI-Generated"}
classes = [display_map[idx_to_class[i]] for i in range(len(idx_to_class))]
# Result: ['AI-Generated', 'Real']
```

This mapping is used in:
- `streamlit_app.py` - for displaying predictions
- `evaluate.py` - for classification report labels
- Model checkpoint validation ensures consistency

### Model Architecture Details
```
ResNet18 Backbone (ImageNet pretrained)
    ↓
Unfrozen Layers: layer3, layer4 (32M → 11M trainable params)
    ↓
Feature Extraction: 512 features
    ↓
Custom Head: 512 → 128 (ReLU) → 2 (softmax)
    ↓
Optimization: Adam (lr=0.00005, weight decay optional)
    ↓
Loss: CrossEntropyLoss
```

---

## 🤝 Contributing

To improve the model or add features:
1. Create a new branch
2. Make changes and test locally
3. Submit a pull request with detailed description

---

## 📖 References

- **Dataset**: [140k Real and Fake Faces - Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Model**: [ResNet - Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Framework**: [PyTorch Official Documentation](https://pytorch.org/)

---

## 📄 License

[Specify your license here - e.g., MIT, Apache 2.0]

---

## 📧 Contact

For questions or collaboration inquiries, feel free to reach out.

---

## 🎓 Project Summary & Status

This project demonstrates how **transfer learning** and **strategic fine-tuning** can create an effective image classifier on limited hardware. By reducing the dataset intelligently and unfreezing deeper layers, we achieved **98%+ training accuracy** on a binary classification task.

### Completed Milestones
✅ **Training**: 10 epochs with validation tracking  
✅ **Best Model**: Saved checkpoint with highest validation accuracy  
✅ **Evaluation**: Full metrics including confusion matrix and F1-scores  
✅ **Web UI**: Interactive Streamlit app for real-time predictions  
✅ **Label Mapping**: Source-of-truth consistency across all components  
✅ **Documentation**: Comprehensive README and debugging utilities  

### Performance Metrics (Test Set)
- **Test Accuracy**: ~98%+ (evaluated via `evaluate.py`)
- **Precision/Recall**: Class-balanced performance (see classification report)
- **Model Size**: ~45MB (ResNet18 with fine-tuned weights)
- **Inference Speed**: <100ms per image (CPU)

**Status**: ✅ Complete and production-ready | 🚀 Ready for deployment
