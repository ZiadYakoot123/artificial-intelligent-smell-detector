# CNN Smell Detector

This project trains a small convolutional neural network (CNN) to predict human odor descriptors from molecular SMILES strings. It is a multi-label task: one molecule can have several smell labels at the same time.

## Project Contents

- `cnn_smell_detector.py`: data loading, molecular signal-to-image conversion, CNN, training, evaluation, and inference.
- `smell_cnn.pt`: trained checkpoint with model weights, smell names, and precision-oriented thresholds.
- `requirements.txt`: runtime dependencies.
- `openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv`: default training dataset.
- `openpom/data/curated_datasets/`: additional preserved CSV datasets.

The previous DeepChem/DGL implementation and unavailable Git LFS checkpoints were removed. All CSV data files were preserved.

## Dataset Format

The default CSV has 4,983 molecules and 138 smell labels. Its columns are:

1. `nonStereoSMILES`: molecular structure encoded as SMILES.
2. `descriptors`: human-readable odor descriptors separated by semicolons.
3. 138 binary smell columns such as `fruity`, `green`, `sweet`, and `woody`.

Rows with invalid SMILES are skipped. The label columns are read from the CSV header, so the script does not hard-code the smell vocabulary.

## Signal-to-Image Encoding

Each molecule is converted deterministically to a `3 x 64 x 64` floating-point image:

- Channel 0 stores the symmetric bond-order adjacency matrix.
- Channel 1 stores atom atomic-number and degree signals on the diagonal and neighboring positions.
- Channel 2 stores a 4,096-bit Morgan fingerprint reshaped to `64 x 64`.

This representation combines local molecular topology with a fixed-length chemical fingerprint. The same SMILES always generates the same tensor, and no molecule image is loaded from an external source.

## Installation

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a CPU-only PyTorch installation, install the CPU wheel according to the official PyTorch instructions before installing the remaining requirements.

## Train and Evaluate

Run the default 15-epoch experiment:

```bash
python cnn_smell_detector.py --epochs 15 --batch-size 64
```

This performs a deterministic 70/10/20 train/validation/test split using seed `7`, trains the CNN, prints evaluation metrics, and writes `smell_cnn.pt`.

Useful options:

```bash
python cnn_smell_detector.py --csv path/to/data.csv --epochs 30
python cnn_smell_detector.py --epochs 1
python cnn_smell_detector.py --min-threshold 0.5
```

The model uses square-root capped positive-class weights to reduce the effect of very rare smells. Per-label thresholds are tuned on validation data and have a default minimum of `0.7` because precision is the current priority.

## Evaluation

The test metrics include:

- Micro and macro ROC-AUC for ranking quality.
- Micro and macro average precision for imbalanced labels.
- Micro and macro binary precision after thresholding.
- Micro recall after thresholding.
- Precision@1, Precision@3, and Precision@5 for ranked smell suggestions.

The latest CPU run produced:

| Metric | Value |
| --- | ---: |
| Micro ROC-AUC | 0.8948 |
| Macro ROC-AUC | 0.8411 |
| Micro average precision | 0.3517 |
| Micro precision at threshold 0.7 | 0.6127 |
| Micro recall at threshold 0.7 | 0.1559 |
| Precision@1 | 0.5888 |

Precision and recall are intentionally not treated as interchangeable. Raising the threshold increases precision but reduces the number of returned smells.

## Predict a Known Molecule

Use the trained checkpoint to predict one SMILES string:

```bash
python cnn_smell_detector.py \
  --predict-smiles "CCO" \
  --output smell_cnn.pt \
  --top-count 10
```

The output is JSON containing smell names and probabilities. Labels passing the saved per-smell thresholds are returned first. If none pass, the command returns the highest-ranked labels so every valid molecule receives a useful result.

## Reproducibility

The seed controls Python, NumPy, and PyTorch random generators. The split seed, image dimensions, CNN architecture, optimizer, class-weight cap, and thresholds are stored or exposed in the training script. Results can still vary slightly between CPU and GPU backends.

## Limitations

- Odor perception is subjective and the labels are noisy.
- A random molecule split can contain chemically similar structures in both train and test sets.
- Rare smell classes have limited test examples, so macro metrics can be unstable.
- The image encoding is a fixed experiment, not a chemically complete representation.
- A high ROC-AUC does not guarantee high precision at the selected operating threshold.
