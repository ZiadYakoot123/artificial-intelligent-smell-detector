"""Simple CNN smell detector trained from the curated CSV dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import (average_precision_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

IMAGE_SIZE = 64
MORGAN_BITS = IMAGE_SIZE * IMAGE_SIZE
DEFAULT_CSV = "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def smiles_to_image(smiles: str) -> np.ndarray:
    """Encode a molecule as a deterministic 3-channel signal image.

    Channel 0 is the bond-order adjacency matrix, channel 1 contains atom
    identity and degree signals, and channel 2 is a Morgan fingerprint laid
    out as a 64x64 bit image.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    image = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    atom_count = min(molecule.GetNumAtoms(), IMAGE_SIZE)
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        if index >= IMAGE_SIZE:
            continue
        image[1, index, index] = min(atom.GetAtomicNum(), 100) / 100.0
        image[1, index, min(index + 1, IMAGE_SIZE - 1)] = atom.GetDegree() / 4.0

    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if start >= IMAGE_SIZE or end >= IMAGE_SIZE:
            continue
        order = float(bond.GetBondTypeAsDouble()) / 3.0
        image[0, start, end] = order
        image[0, end, start] = order

    fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        molecule, radius=2, nBits=MORGAN_BITS
    )
    bits = np.zeros((MORGAN_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    image[2] = bits.reshape(IMAGE_SIZE, IMAGE_SIZE)
    image[1, :atom_count, :atom_count] += np.eye(atom_count, dtype=np.float32) * 0.25
    return image


def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 3:
        raise ValueError("CSV must contain SMILES, descriptors, and smell columns")
    smiles_column = frame.columns[0]
    label_columns = list(frame.columns[2:])
    images, labels, kept_smiles, kept_descriptors = [], [], [], []
    for _, row in frame.iterrows():
        try:
            images.append(smiles_to_image(str(row[smiles_column])))
        except ValueError:
            continue
        kept_smiles.append(str(row[smiles_column]))
        kept_descriptors.append(str(row[frame.columns[1]]))
        labels.append(row[label_columns].to_numpy(dtype=np.float32))
    if not images:
        raise ValueError("No valid SMILES were found in the CSV")
    return np.stack(images), np.stack(labels), kept_smiles, kept_descriptors


class SmellDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


class SmellCNN(nn.Module):
    def __init__(self, output_count: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, output_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def train_epoch(model, loader, optimizer, loss_function, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def predict(model, loader, device):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            predictions.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def tune_thresholds(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Choose per-smell thresholds on validation data for precision-focused F-score."""
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)
    for label_index in range(labels.shape[1]):
        if labels[:, label_index].sum() == 0:
            continue
        candidates = np.linspace(0.1, 0.9, 17)
        scores = [
            fbeta_score(labels[:, label_index], predictions[:, label_index] >= threshold,
                        beta=0.5, zero_division=0)
            for threshold in candidates
        ]
        thresholds[label_index] = candidates[int(np.argmax(scores))]
    return thresholds


def evaluate(predictions: np.ndarray, labels: np.ndarray,
             thresholds: np.ndarray | None = None) -> dict[str, float]:
    valid = labels.sum(axis=0) > 0
    if thresholds is None:
        thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)
    decisions = predictions >= thresholds
    metrics = {
        "roc_auc_micro": float(roc_auc_score(labels[:, valid], predictions[:, valid], average="micro")),
        "roc_auc_macro": float(roc_auc_score(labels[:, valid], predictions[:, valid], average="macro")),
        "average_precision_micro": float(average_precision_score(labels[:, valid], predictions[:, valid], average="micro")),
        "average_precision_macro": float(average_precision_score(labels[:, valid], predictions[:, valid], average="macro")),
        "precision_micro": float(precision_score(labels[:, valid], decisions[:, valid], average="micro", zero_division=0)),
        "precision_macro": float(precision_score(labels[:, valid], decisions[:, valid], average="macro", zero_division=0)),
        "recall_micro": float(recall_score(labels[:, valid], decisions[:, valid], average="micro", zero_division=0)),
    }
    for count in (1, 3, 5):
        if count > predictions.shape[1]:
            continue
        top_indices = np.argpartition(predictions, -count, axis=1)[:, -count:]
        hits = np.take_along_axis(labels, top_indices, axis=1).sum()
        metrics[f"precision_at_{count}"] = float(hits / (len(labels) * count))
    return metrics


def predict_smiles(smiles: str, checkpoint_path: str, top_count: int = 10) -> list[dict[str, float | str]]:
    """Predict the most likely smell labels for one SMILES string."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    labels = checkpoint["labels"]
    model = SmellCNN(len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image = torch.from_numpy(smiles_to_image(smiles)).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(image))[0].numpy()
    thresholds = np.asarray(checkpoint.get("thresholds", np.full(len(labels), 0.5)))
    ranked = np.argsort(probabilities)[::-1]
    selected = [index for index in ranked if probabilities[index] >= thresholds[index]]
    selected = selected[:top_count] or list(ranked[:top_count])
    return [{"smell": labels[index], "probability": float(probabilities[index])}
            for index in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="smell_cnn.pt")
    parser.add_argument("--weight-cap", type=float, default=5.0)
    parser.add_argument("--min-threshold", type=float, default=0.7)
    parser.add_argument("--predict-smiles", help="Predict smells for one SMILES using --output")
    parser.add_argument("--top-count", type=int, default=10)
    args = parser.parse_args()

    if args.predict_smiles:
        print(json.dumps(predict_smiles(args.predict_smiles, args.output, args.top_count), indent=2))
        return

    set_seed(args.seed)
    images, labels, smiles, descriptors = load_dataset(args.csv)
    indices = np.arange(len(labels))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=args.seed)
    train_indices, validation_indices = train_test_split(train_indices, test_size=0.125, random_state=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(SmellDataset(images[train_indices], labels[train_indices]), args.batch_size, shuffle=True)
    validation_loader = DataLoader(SmellDataset(images[validation_indices], labels[validation_indices]), args.batch_size)
    test_loader = DataLoader(SmellDataset(images[test_indices], labels[test_indices]), args.batch_size)
    model = SmellCNN(labels.shape[1]).to(device)
    positive = labels[train_indices].sum(axis=0)
    negative = len(train_indices) - positive
    class_weights = np.sqrt(np.divide(negative, np.maximum(positive, 1)))
    weights = torch.tensor(np.minimum(class_weights, args.weight_cap), dtype=torch.float32, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, loss_function, device)
        validation_predictions, validation_labels = predict(model, validation_loader, device)
        print(f"epoch={epoch + 1:02d} loss={loss:.4f} validation_ap={evaluate(validation_predictions, validation_labels)['average_precision_micro']:.4f}")

    validation_predictions, validation_labels = predict(model, validation_loader, device)
    thresholds = np.maximum(
        tune_thresholds(validation_predictions, validation_labels),
        args.min_threshold,
    )
    predictions, test_labels = predict(model, test_loader, device)
    metrics = evaluate(predictions, test_labels, thresholds)
    print(json.dumps(metrics, indent=2))
    checkpoint = {"state_dict": model.state_dict(), "labels": list(pd.read_csv(args.csv, nrows=0).columns[2:]), "thresholds": thresholds, "seed": args.seed}
    torch.save(checkpoint, args.output)

    print("Known-label checks:")
    for row_index in test_indices[:5]:
        ranked = np.argsort(predictions[list(test_indices[:5]).index(row_index)])[::-1][:5]
        expected = [name for name, value in zip(checkpoint["labels"], labels[row_index]) if value]
        print(smiles[row_index], "expected=", expected, "top=", [checkpoint["labels"][index] for index in ranked])


if __name__ == "__main__":
    main()
