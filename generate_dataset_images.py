"""Export RDKit molecular depictions and labels for inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit.Chem import Draw

from molecular_smell_pipeline import DEFAULT_CSV, load_dataset


def clean_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_"
                   for character in value)[:80]


def render_molecule(molecule, labels: list[str], output_path: Path) -> None:
    """Render a chemical structure rather than a fingerprint pseudo-image."""
    structure = Draw.MolToImage(molecule, size=(768, 600), kekulize=False)
    canvas = Image.new("RGB", (768, 720), "white")
    canvas.paste(structure, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text = "Odor labels: " + (", ".join(labels) if labels else "none")
    draw.multiline_text((16, 620), text, fill="black", font=font, spacing=6)
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default="generated_images")
    parser.add_argument("--features-dir", default="generated_features")
    parser.add_argument("--bits", type=int, choices=(2048, 4096), default=2048)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.csv, args.bits)
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_dir / f"morgan_{args.bits}.npy", dataset.fingerprints)
    frame = pd.DataFrame(dataset.labels, columns=dataset.label_names)
    frame.insert(0, "descriptors", dataset.descriptors)
    frame.insert(0, "smiles", dataset.smiles)
    frame.to_csv(output_dir / "metadata.csv", index=False)
    for index, (molecule, values) in enumerate(zip(dataset.molecules, dataset.labels)):
        active_labels = [
            name for name, value in zip(dataset.label_names, values) if value > 0
        ]
        suffix = clean_filename(active_labels[0] if active_labels else "unlabeled")
        render_molecule(molecule, active_labels, output_dir / f"{index:05d}_{suffix}.png")
    print(f"Exported {len(dataset.smiles)} molecules with {args.bits}-bit Morgan vectors")
    print(f"Images: {output_dir}")
    print(f"Features: {features_dir / f'morgan_{args.bits}.npy'}")
    print(f"Metadata: {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
