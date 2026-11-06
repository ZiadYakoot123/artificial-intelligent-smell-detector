# Molecular Smell Detection

This repository contains scaffold-aware, multi-label odor prediction
baselines. The production pipeline does **not** reshape fingerprints into
images: Morgan fingerprints remain flat vectors, and the graph baseline
operates directly on RDKit atom/bond topology.

## Pipeline

`molecular_smell_pipeline.py` provides:

- `MorganMLP`: 2,048- or 4,096-bit Morgan vectors with BatchNorm and Dropout.
- `MPNN`: edge-aware message passing over atoms and bonds.
- `train_gbdt`: optional one-XGBoost-classifier-per-smell baseline.
- `scaffold_split`: deterministic Bemis-Murcko group partitioning. A scaffold
  is assigned to exactly one of train, validation, or test.
- `FocalLoss`: suppresses easy clean-molecule negatives.
- `tune_thresholds`: independently optimizes each smell threshold on validation
  data (F1 by default, or F2 with `--beta 2`).
- `comparative_evaluation`: reports default 0.5 versus validation-tuned
  thresholds on the untouched test set.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run both neural baselines:

```bash
python molecular_smell_pipeline.py --model both --bits 2048 --epochs 15
```

Run one baseline, use 4,096 bits, and prioritize recall during calibration:

```bash
python molecular_smell_pipeline.py --model mlp --bits 4096 --beta 2
```

The output is a comparative table containing Micro/Macro F1, Precision,
Recall, ROC-AUC, and Average Precision for both the default and dynamically
tuned thresholds. ROC-AUC and Average Precision are ranking metrics and do not
depend on the selected threshold.

The original `cnn_smell_detector.py` remains only as a compatibility entry
point and forwards to this pipeline. The old pseudo-2D CNN and image
conversion are intentionally not supported.
