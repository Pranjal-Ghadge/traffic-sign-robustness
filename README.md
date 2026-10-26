# Traffic Sign Classification Under Varying Environmental Conditions

## 1. Project Overview

This project evaluates the robustness of a traffic sign classification model under different environmental conditions. It uses pretrained DINOv2-Small for feature extraction and compares an MLP classifier with a Linear SVM classifier.

The model is evaluated on clean traffic sign images and images affected by environmental corruptions of varying severity.

## 2. Technologies Used

- Python
- PyTorch and Torchvision
- DINOv2-Small
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- ImageCorruptions
- tqdm

## 3. Dataset

The project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

Place the training images in:

`data/GTSRB/Train/`

The dataset is divided into training and validation sets using an 80:20 split with random seed 42.

**Note:** The reported validation metrics use a split of the GTSRB training dataset, not the official test set.

## 4. Model Architecture

### Feature Extraction

Pretrained DINOv2-Small extracts a 384-dimensional feature vector from each traffic sign image.

### MLP Classifier

The MLP architecture is:

`384 → 256 → ReLU → Dropout → 128 → ReLU → Dropout → 43`

### Linear SVM

A Linear Support Vector Machine is trained on the same extracted features as a comparison model.

## 5. Environmental Corruptions

The project evaluates 14 corruption types at five severity levels, giving 70 corruption conditions.

Examples include:

- Gaussian noise
- Impulse noise
- Shot noise
- Fog
- Snow
- Frost
- Motion blur
- Defocus blur
- Zoom blur
- Brightness
- Contrast
- Elastic transformation
- Pixelation
- JPEG compression

## 6. Setup

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Ensure that the dataset is placed in the expected directory before running the scripts.

## 7. Running the Project

Run commands from the project root directory.

Test DINOv2:

```bash
python .\src\test_dinov2.py
```

Extract features:

```bash
python .\src\extract_features.py
```

Train the MLP:

```bash
python .\src\train_mlp.py
```

Train the Linear SVM:

```bash
python .\src\train_svm.py
```

Evaluate one corruption condition:

```bash
python .\src\evaluate_corruptions.py
```

Evaluate all corruption conditions:

```bash
python .\src\evaluate_all_corruptions.py
```

Generate result plots:

```bash
python .\src\plot_results.py
```

**Note:** Feature extraction and training commands may overwrite existing outputs. The full corruption evaluation script skips conditions already recorded in its results CSV.

## 8. Results

Clean validation performance:

| Metric | DINOv2 + MLP | DINOv2 + Linear SVM |
|---|---:|---:|
| Validation accuracy | 96.30% | 90.67% |
| Validation macro-F1 | 0.9656 | 0.9203 |

The corruption evaluation results are stored in:

- `results/gtsrb_corruption_results.csv`
- `results/plots/accuracy_vs_severity.pdf`
- `results/plots/macro_f1_vs_severity.pdf`

Confusion matrices are stored in the `results/` directory.

Gaussian noise at severity level 3 produced 46.72% accuracy and a macro-F1 score of 0.4097 in the individual evaluation.

## 9. Model and Feature Files

- `models/gtsrb_mlp.pth` — trained MLP checkpoint
- `models/gtsrb_linear_svm.joblib` — trained Linear SVM model
- `data/processed_features_gtsrb/` — extracted feature arrays and labels

## 10. Limitations

The reported validation results are based on a validation split of the GTSRB training dataset. Evaluation on the official test set would provide an additional measure of generalization.