## Dataset Notes
- HAM10000: 10,015 images, filtered to 5 classes (melanoma, bcc, nv, bkl, akiec). Images 600x450, resize to 224x224. Imbalanced, nv dominant.
- ISIC Archive: ~10–25 dermoscopic images for melanoma, bcc, nv, bkl, akiec. Variable sizes, resize to 224x224.
- Plan: Resize to 224x224, augment to balance classes.
## Preprocessing Notes
- Filtered HAM10000 for 5 classes (melanoma, bcc, nv, bkl, akiec): ~7,000–8,000 images.
- Class distribution: nv dominant (~5,000), mel/akiec scarce (~500 each).
- Added ~10–25 ISIC images per class to processed/.
- Resized images to 224x224 for ResNet50 compatibility.
- Split into train (70%), validation (20%), test (10%) sets.
- Applied augmentation (rotations, flips, zoom, shear) to mel/akiec during training to address imbalance.
- Created data generators with batch size 32, normalized pixels to [0, 1].