"""Generate labeled PNG views of the molecular signal images."""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from cnn_smell_detector import load_dataset

DEFAULT_CSV = "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
DEFAULT_OUTPUT = "generated_images"


def clean_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:80]


def render_labeled_image(image: np.ndarray, labels: list[str], output_path: Path) -> None:
    """Render channels as RGB and add active smell labels below the image."""
    rgb = np.transpose(image, (1, 2, 0))
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)
    rendered = Image.fromarray(rgb, mode="RGB").resize((384, 384), Image.Resampling.NEAREST)
    label_text = "Labels: " + (", ".join(labels) if labels else "none")
    canvas = Image.new("RGB", (384, 480), "white")
    canvas.paste(rendered, (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    wrapped = textwrap.wrap(label_text, width=58)
    draw.multiline_text((8, 392), "\n".join(wrapped), fill="black", font=font, spacing=4)
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images, labels, smiles, descriptors = load_dataset(args.csv)
    header = list(pd.read_csv(args.csv, nrows=0).columns[2:])
    count = len(images) if args.limit is None else min(args.limit, len(images))
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=["image", "smiles", "descriptors", "labels"])
        writer.writeheader()
        for index in range(count):
            active_labels = [name for name, value in zip(header, labels[index]) if value]
            label_suffix = clean_filename(active_labels[0] if active_labels else "unlabeled")
            filename = f"{index:05d}_{label_suffix}.png"
            render_labeled_image(images[index], active_labels, output_dir / filename)
            writer.writerow({
                "image": filename,
                "smiles": smiles[index],
                "descriptors": descriptors[index],
                "labels": ";".join(active_labels),
            })
    print(f"Generated {count} labeled images in {output_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
