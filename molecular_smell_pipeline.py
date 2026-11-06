"""Scaffold-aware molecular smell prediction baselines.

The module deliberately keeps Morgan fingerprints as flat vectors.  The graph
baseline consumes RDKit molecular graphs directly and does not depend on DGL
or a pseudo-image representation.
"""

from __future__ import annotations

import argparse
import importlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (average_precision_score, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset

DEFAULT_CSV = "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv"
_MORGAN_GENERATORS = {
    bits: rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    for bits in (2048, 4096)
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class MolecularDataset:
    smiles: list[str]
    descriptors: list[str]
    labels: np.ndarray
    label_names: list[str]
    fingerprints: np.ndarray
    molecules: list[Chem.Mol]


def _fingerprint(molecule: Chem.Mol, n_bits: int) -> np.ndarray:
    try:
        generator = _MORGAN_GENERATORS[n_bits]
    except KeyError as error:
        raise ValueError("n_bits must be 2048 or 4096") from error
    bit_vector = generator.GetFingerprint(molecule)
    result = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, result)
    return result


def load_dataset(csv_path: str | Path, n_bits: int = 2048) -> MolecularDataset:
    """Load valid molecules and flat Morgan fingerprints from the curated CSV."""
    frame = pd.read_csv(csv_path)
    if frame.shape[1] < 3:
        raise ValueError("CSV must contain SMILES, descriptors, and smell columns")
    smiles_name, descriptor_name = frame.columns[:2]
    label_names = list(frame.columns[2:])
    smiles, descriptors, labels, fingerprints, molecules = [], [], [], [], []
    for _, row in frame.iterrows():
        value = str(row[smiles_name])
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        smiles.append(value)
        descriptors.append(str(row[descriptor_name]))
        labels.append(row[label_names].to_numpy(dtype=np.float32))
        fingerprints.append(_fingerprint(molecule, n_bits))
        molecules.append(molecule)
    if not labels:
        raise ValueError("No valid SMILES were found in the CSV")
    return MolecularDataset(smiles, descriptors, np.stack(labels), label_names,
                            np.stack(fingerprints), molecules)


def scaffold_split(
    molecules: Sequence[Chem.Mol],
    seed: int = 7,
    fractions: tuple[float, float, float] = (0.7, 0.1, 0.2),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition whole Bemis-Murcko scaffold groups into train/validation/test."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    groups: dict[str, list[int]] = {}
    for index, molecule in enumerate(molecules):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        groups.setdefault(scaffold, []).append(index)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    targets = np.asarray(fractions) * len(molecules)
    buckets: list[list[int]] = [[], [], []]
    for group in sorted(shuffled, key=len, reverse=True):
        remaining = targets - np.asarray([len(bucket) for bucket in buckets])
        bucket = int(np.argmax(remaining))
        buckets[bucket].extend(group)
    if any(not bucket for bucket in buckets):
        raise ValueError("Scaffold split produced an empty partition; adjust fractions")
    return tuple(np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets)


class FingerprintDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, fingerprints: np.ndarray, labels: np.ndarray):
        self.fingerprints = torch.from_numpy(fingerprints)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fingerprints[index], self.labels[index]


class MorganMLP(nn.Module):
    """MLP for a flat Morgan vector; no spatial assumptions are introduced."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class FocalLoss(nn.Module):
    """Multi-label sigmoid focal loss for sparse odor descriptors."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        cross_entropy = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * cross_entropy).mean()


def _atom_features(atom: Chem.Atom) -> list[float]:
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        float(atom.IsInRing()),
        atom.GetMass() / 250.0,
        float(atom.GetChiralTag()),
    ]


def _bond_features(bond: Chem.Bond) -> list[float]:
    return [
        bond.GetBondTypeAsDouble() / 3.0,
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


@dataclass
class Graph:
    nodes: torch.Tensor
    edges: torch.Tensor
    edge_features: torch.Tensor
    graph_index: torch.Tensor


def molecule_to_graph(molecule: Chem.Mol) -> Graph:
    nodes = torch.tensor([_atom_features(atom) for atom in molecule.GetAtoms()],
                         dtype=torch.float32)
    edges, edge_features = [], []
    for bond in molecule.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = _bond_features(bond)
        edges.extend([(start, end), (end, start)])
        edge_features.extend([features, features])
    if edges:
        edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        bond_tensor = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_tensor = torch.empty((2, 0), dtype=torch.long)
        bond_tensor = torch.empty((0, 4), dtype=torch.float32)
    return Graph(nodes, edge_tensor, bond_tensor,
                 torch.zeros(len(nodes), dtype=torch.long))


def collate_graphs(items: list[tuple[Graph, torch.Tensor]]) -> tuple[Graph, torch.Tensor]:
    node_parts, edge_parts, bond_parts, graph_parts, labels = [], [], [], [], []
    offset = 0
    for graph_index, (graph, label) in enumerate(items):
        node_parts.append(graph.nodes)
        edge_parts.append(graph.edges + offset)
        bond_parts.append(graph.edge_features)
        graph_parts.append(torch.full((len(graph.nodes),), graph_index, dtype=torch.long))
        labels.append(label)
        offset += len(graph.nodes)
    return Graph(torch.cat(node_parts), torch.cat(edge_parts, dim=1),
                 torch.cat(bond_parts), torch.cat(graph_parts)), torch.stack(labels)


class GraphDataset(Dataset[tuple[Graph, torch.Tensor]]):
    def __init__(self, molecules: Sequence[Chem.Mol], labels: np.ndarray):
        self.graphs = [molecule_to_graph(molecule) for molecule in molecules]
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Graph, torch.Tensor]:
        return self.graphs[index], self.labels[index]


class MPNN(nn.Module):
    """Small edge-aware message-passing network over atom/bond topology."""

    def __init__(self, output_size: int, hidden_size: int = 128, steps: int = 3):
        super().__init__()
        self.steps = steps
        self.node_projection = nn.Linear(8, hidden_size)
        self.message = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.update = nn.GRUCell(hidden_size, hidden_size)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                  nn.GELU(), nn.Dropout(0.25),
                                  nn.Linear(hidden_size, output_size))

    def forward(self, graph: Graph) -> torch.Tensor:
        hidden = self.node_projection(graph.nodes)
        source, target = graph.edges
        for _ in range(self.steps):
            messages = self.message(torch.cat((hidden[source], graph.edge_features), dim=1))
            aggregate = torch.zeros_like(hidden).index_add_(0, target, messages)
            hidden = self.update(aggregate, hidden)
        pooled = torch.zeros((graph.graph_index.max().item() + 1, hidden.shape[1]),
                             device=hidden.device).index_add_(0, graph.graph_index, hidden)
        counts = torch.bincount(graph.graph_index, minlength=len(pooled)).clamp_min(1)
        return self.head(pooled / counts.unsqueeze(1))


def predict(model: nn.Module, loader: DataLoader, device: torch.device,
            graph_model: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for features, batch_labels in loader:
            if graph_model:
                graph = features
                graph = Graph(graph.nodes.to(device), graph.edges.to(device),
                              graph.edge_features.to(device), graph.graph_index.to(device))
                logits = model(graph)
            else:
                logits = model(features.to(device))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch_labels.numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                loss_function: nn.Module, device: torch.device,
                graph_model: bool = False) -> float:
    model.train()
    total = 0.0
    for features, labels in loader:
        if graph_model:
            graph = Graph(features.nodes.to(device), features.edges.to(device),
                          features.edge_features.to(device), features.graph_index.to(device))
            logits = model(graph)
        else:
            logits = model(features.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


def tune_thresholds(predictions: np.ndarray, labels: np.ndarray, beta: float = 1.0,
                    candidates: Iterable[float] | None = None) -> np.ndarray:
    """Tune one threshold per label on validation predictions only."""
    candidates = np.asarray(
        list(np.linspace(0.01, 0.99, 99) if candidates is None else candidates)
    )
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float32)
    for index in range(labels.shape[1]):
        if labels[:, index].sum() == 0:
            continue
        scores = [fbeta_score(labels[:, index], predictions[:, index] >= threshold,
                              beta=beta, zero_division=0) for threshold in candidates]
        thresholds[index] = candidates[int(np.argmax(scores))]
    return thresholds


def evaluate(predictions: np.ndarray, labels: np.ndarray,
             thresholds: np.ndarray | None = None) -> dict[str, float]:
    """Return ranking metrics and thresholded micro/macro metrics."""
    valid = (labels.sum(axis=0) > 0) & (labels.sum(axis=0) < len(labels))
    if not valid.any():
        raise ValueError("Evaluation requires at least one positive label")
    thresholds = np.full(labels.shape[1], 0.5) if thresholds is None else thresholds
    decisions = predictions >= thresholds
    return {
        "micro_f1": float(f1_score(labels[:, valid], decisions[:, valid], average="micro", zero_division=0)),
        "macro_f1": float(f1_score(labels[:, valid], decisions[:, valid], average="macro", zero_division=0)),
        "micro_precision": float(precision_score(labels[:, valid], decisions[:, valid], average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels[:, valid], decisions[:, valid], average="macro", zero_division=0)),
        "micro_recall": float(recall_score(labels[:, valid], decisions[:, valid], average="micro", zero_division=0)),
        "macro_recall": float(recall_score(labels[:, valid], decisions[:, valid], average="macro", zero_division=0)),
        "micro_roc_auc": float(roc_auc_score(labels[:, valid], predictions[:, valid], average="micro")),
        "macro_roc_auc": float(roc_auc_score(labels[:, valid], predictions[:, valid], average="macro")),
        "micro_average_precision": float(average_precision_score(labels[:, valid], predictions[:, valid], average="micro")),
        "macro_average_precision": float(average_precision_score(labels[:, valid], predictions[:, valid], average="macro")),
    }


def comparative_evaluation(predictions: np.ndarray, labels: np.ndarray,
                           thresholds: np.ndarray) -> pd.DataFrame:
    """Build the requested default-vs-calibrated test comparison table."""
    rows = []
    for name, selected in (("default_0.5", np.full(labels.shape[1], 0.5)),
                           ("validation_tuned", thresholds)):
        rows.append({"threshold": name, **evaluate(predictions, labels, selected)})
    return pd.DataFrame(rows)


def train_gbdt(fingerprints: np.ndarray, labels: np.ndarray, seed: int = 7):
    """Train one XGBoost classifier per smell (optional xgboost dependency)."""
    try:
        xgboost = importlib.import_module("xgboost")
    except ImportError as error:
        raise ImportError("Install xgboost to use the GBDT baseline") from error
    models = []
    for index in range(labels.shape[1]):
        model = xgboost.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=seed
        )
        model.fit(fingerprints, labels[:, index])
        models.append(model)
    return models


def _loaders(dataset: MolecularDataset, split: tuple[np.ndarray, ...],
             batch_size: int, graph_model: bool) -> tuple[DataLoader, ...]:
    loaders = []
    for indices in split:
        if graph_model:
            source = GraphDataset([dataset.molecules[i] for i in indices], dataset.labels[indices])
            loaders.append(DataLoader(source, batch_size=batch_size,
                                      shuffle=len(loaders) == 0, collate_fn=collate_graphs))
        else:
            source = FingerprintDataset(dataset.fingerprints[indices], dataset.labels[indices])
            loaders.append(DataLoader(source, batch_size=batch_size,
                                      shuffle=len(loaders) == 0))
    return tuple(loaders)


def run_experiment(dataset: MolecularDataset, model_name: str, epochs: int = 15,
                   batch_size: int = 64, seed: int = 7, beta: float = 1.0) -> pd.DataFrame:
    set_seed(seed)
    split = scaffold_split(dataset.molecules, seed)
    graph_model = model_name == "mpnn"
    train_loader, validation_loader, test_loader = _loaders(dataset, split, batch_size, graph_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module = (MPNN(dataset.labels.shape[1]) if graph_model else
                        MorganMLP(dataset.fingerprints.shape[1], dataset.labels.shape[1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = FocalLoss()
    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, loss_function, device, graph_model)
    validation_predictions, validation_labels = predict(model, validation_loader, device, graph_model)
    thresholds = tune_thresholds(validation_predictions, validation_labels, beta=beta)
    test_predictions, test_labels = predict(model, test_loader, device, graph_model)
    result = comparative_evaluation(test_predictions, test_labels, thresholds)
    result.insert(0, "model", model_name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--model", choices=("mlp", "mpnn", "both"), default="both")
    parser.add_argument("--bits", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="F-score beta for validation threshold tuning; use 2 for recall")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    dataset = load_dataset(args.csv, args.bits)
    models = ("mlp", "mpnn") if args.model == "both" else (args.model,)
    table = pd.concat([run_experiment(dataset, name, args.epochs, args.batch_size,
                                       args.seed, args.beta) for name in models])
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
