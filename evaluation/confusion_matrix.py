"""Confusion matrix computation and export."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> np.ndarray:
    return sk_confusion_matrix(y_true, y_pred, labels=range(len(class_names)))


def save_confusion_matrix_csv(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_path)


def save_confusion_matrix_image(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    normalize: bool = True,
    figsize: tuple[int, int] = (14, 12),
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_cm = cm.astype(float)
    if normalize:
        row_sums = display_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display_cm = display_cm / row_sums

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        display_cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
