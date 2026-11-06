"""Export flat Morgan features and labels for inspection or downstream tools."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from molecular_smell_pipeline import DEFAULT_CSV, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default="generated_features")
    parser.add_argument("--bits", type=int, choices=(2048, 4096), default=2048)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.csv, args.bits)
    np.save(output_dir / f"morgan_{args.bits}.npy", dataset.fingerprints)
    frame = pd.DataFrame(dataset.labels, columns=dataset.label_names)
    frame.insert(0, "descriptors", dataset.descriptors)
    frame.insert(0, "smiles", dataset.smiles)
    frame.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Exported {len(dataset.smiles)} molecules with {args.bits}-bit Morgan vectors")
    print(f"Features: {output_dir / f'morgan_{args.bits}.npy'}")
    print(f"Metadata: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
