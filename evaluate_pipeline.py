"""Run scaffold-aware model evaluation and save comparison charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from molecular_smell_pipeline import (
    FocalLoss,
    MPNN,
    MorganMLP,
    _loaders,
    comparative_evaluation,
    load_dataset,
    predict,
    scaffold_split,
    set_seed,
    train_epoch,
    tune_thresholds,
)


def evaluate_model(dataset, model_name: str, epochs: int, batch_size: int, seed: int,
                   beta: float) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    set_seed(seed)
    split = scaffold_split(dataset.molecules, seed)
    graph_model = model_name == "mpnn"
    train_loader, validation_loader, test_loader = _loaders(
        dataset, split, batch_size, graph_model
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (
        MPNN(dataset.labels.shape[1])
        if graph_model
        else MorganMLP(dataset.fingerprints.shape[1], dataset.labels.shape[1])
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss = FocalLoss()
    history = []
    for epoch in range(epochs):
        history.append(train_epoch(
            model, train_loader, optimizer, loss, device, graph_model
        ))
        print(f"{model_name} epoch {epoch + 1}/{epochs}: loss={history[-1]:.4f}")
    validation_predictions, validation_labels = predict(
        model, validation_loader, device, graph_model
    )
    thresholds = tune_thresholds(validation_predictions, validation_labels, beta)
    test_predictions, test_labels = predict(model, test_loader, device, graph_model)
    table = comparative_evaluation(test_predictions, test_labels, thresholds)
    table.insert(0, "model", model_name)
    return table, {
        "history": np.asarray(history),
        "thresholds": thresholds,
        "test_predictions": test_predictions,
        "test_labels": test_labels,
    }


def save_charts(results: pd.DataFrame, artifacts: dict[str, dict[str, np.ndarray]],
                output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "metrics.csv", index=False)

    metric_names = [
        "micro_f1", "macro_f1", "micro_precision", "macro_precision",
        "micro_recall", "macro_recall", "micro_roc_auc", "macro_roc_auc",
        "micro_average_precision", "macro_average_precision",
    ]
    labels = [name.replace("_", " ").title() for name in metric_names]
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)
    for axis, model_name in zip(axes, artifacts):
        frame = results[results["model"] == model_name].set_index("threshold")
        frame[metric_names].T.plot.bar(ax=axis, color=["#4c78a8", "#f58518"])
        axis.set_title(f"{model_name.upper()} — default vs validation-tuned thresholds")
        axis.set_ylabel("Score")
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(title="Threshold")
    figure.savefig(output_dir / "metric_comparison.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for model_name, artifact in artifacts.items():
        axis.plot(np.arange(1, len(artifact["history"]) + 1),
                  artifact["history"], marker="o", label=model_name.upper())
    axis.set_title("Training loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Focal loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--bits", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    dataset = load_dataset(args.csv, args.bits)
    tables, artifacts = [], {}
    for model_name in ("mlp", "mpnn"):
        table, artifact = evaluate_model(
            dataset, model_name, args.epochs, args.batch_size, args.seed, args.beta
        )
        tables.append(table)
        artifacts[model_name] = artifact
    results = pd.concat(tables, ignore_index=True)
    save_charts(results, artifacts, Path(args.output_dir))
    print("\n" + results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved metrics and charts to {args.output_dir}/")


if __name__ == "__main__":
    main()
