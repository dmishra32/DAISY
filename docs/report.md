# DAISY Project Report

## Project Overview
- **Objective**: Develop an industrial-grade Dermatological AI System for You (DAISY) for skin disease classification (akiec, bcc, bkl, mel, nv) using ResNet50.
- **Context**: Vocational training initiative for robust healthcare AI deployment.

## Dataset
- **HAM10000**: 10,015 images, filtered to 5 classes (melanoma, bcc, nv, bkl, akiec). Original size 600x450, resized to 224x224. Imbalanced, nv dominant.
- **ISIC Archive**: ~10–25 dermoscopic images per class (melanoma, bcc, nv, bkl, akiec). Variable sizes, resized to 224x224.
- **Plan**: Resize to 224x224, augment to balance classes.

## Preprocessing
- **Filtering**: HAM10000 reduced to ~7,000–8,000 images for 5 classes.
- **Distribution**: nv dominant (~5,000), mel/akiec scarce (~500 each).
- **Enhancement**: Added ~10–25 ISIC images per class to processed/.
- **Resizing**: All images resized to 224x224 for ResNet50 compatibility.
- **Split**: 70% train (~5,000), 20% validation (~1,400), 10% test (~700).
- **Augmentation**: Applied to mel/akiec during training (rotations, flips, zoom, shear) to address imbalance.
- **Generators**: Batch size 32, pixels normalized to [0, 1].

## Model
- **Architecture**: ResNet50-based CNN for 5-class classification.
  - Base: Pre-trained ResNet50 (ImageNet, frozen layers).
  - Layers: GlobalAveragePooling2D, Dense (128 units, ReLU), Output (5 units, softmax).
- **Compilation**: Adam optimizer, categorical crossentropy loss, accuracy metric.
- **Input**: 224x224x3 images.
- **Parameters**: ~23.6M total (~100K trainable, ~23.5M non-trainable).
- **Saved**: resnet50_base.keras (and resnet50_base.h5 backup) in MyDrive/DAISY-Project/models/.
- **Rationale**: Transfer learning with frozen layers reduces training time and overfitting on ~7,000 images.
- **Note**: Used .keras format (TensorFlow 2.16.2 recommendation), .h5 as fallback.

## Training Pipeline
- **Generators**:
  - Train: ~5,000 images, batch size 32, augmentation for mel/akiec.
  - Validation: ~1,400 images, batch size 32, no augmentation.
  - Class indices: akiec=0, bcc=1, bkl=2, mel=3, nv=4.
- **Plan**: 10 epochs with early stopping (monitor val_loss, patience=3, restore best weights) and checkpointing (resnet50_best.keras).
- **Steps**: ~156 train (~5,000 / 32), ~44 validation (~1,400 / 32).
- **Format**: .keras checkpoints (TensorFlow 2.16.2 recommendation).

## Training and Evaluation
### Initial Model
- Performance: Validation accuracy 68.59%, loss 0.9524.
- Classification: akiec/bcc/bkl F1=0.00, mel F1=0.04, nv F1=0.81.
- Challenge: Class imbalance.

### Fine-Tuning with Class Weights
- Weights: akiec 5.0, bcc 3.0, bkl 2.0, mel 5.0, nv 1.0.
- Performance: Validation accuracy 68.28%, loss 1.0460.
- Classification: akiec/bcc/bkl F1=0.00, mel F1=0.33, nv F1=0.82.
- Improvement: mel F1 from 0.04 to 0.33.

### Advanced Metrics
- Precision-Recall AUC: akiec/bcc/bkl <0.5, mel/nv ~0.7.
- ROC AUC: Similar trends.
- Visuals: [confusion_matrix_weighted.png](docs/confusion_matrix_weighted.png), [pr_curve.png](docs/pr_curve.png), [roc_curve.png](docs/roc_curve.png).

## Models
- resnet50_best.keras: Initial trained model.
- resnet50_weighted.keras: Fine-tuned best model.
- resnet50_trained.keras: Final model.

## Observations
- Class weights improved mel; akiec/bcc/bkl need optimization due to low samples.
- Next steps: Unfreeze layers and data augmentation.

## Technical Details
- Environment: TensorFlow 2.16.2, Colab GPU, daisy_env.
- Version Control: Synced to https://github.com/dmishr/DAISY-Project.
- Reproducibility: [classification_report_weighted.json](docs/classification_report_weighted.json).