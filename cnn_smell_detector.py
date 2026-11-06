"""Compatibility entry point for the scaffold-aware molecular smell pipeline.

The former pseudo-image CNN has been removed.  Use
``molecular_smell_pipeline.py`` for flat Morgan/MLP, MPNN, calibration, and
scaffold-split experiments.
"""

from molecular_smell_pipeline import (FocalLoss, MPNN, MorganMLP,
                                       MolecularDataset, evaluate,
                                       load_dataset, main,
                                       scaffold_split, tune_thresholds)

__all__ = [
    "FocalLoss", "MPNN", "MorganMLP", "MolecularDataset", "evaluate",
    "load_dataset", "scaffold_split", "tune_thresholds",
]


if __name__ == "__main__":
    main()
