"""Classification metrics and the Region Isolation Accuracy (RIA) metric.

RIA is computed from the segmentation log already produced by
preprocessing.segmentation.process_dataset, rather than re-running OpenCV a
second time at evaluation -- it measures "did the classical CV pipeline find
a bacterial region," independent of CNN performance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    average: str = "macro",
) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(class_names)), zero_division=0
        )
    )
    per_class = {
        name: {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "per_class": per_class,
    }


def compute_region_isolation_accuracy(
    segmentation_log_path: Path,
    min_area: int = config.MIN_CONTOUR_AREA,
) -> dict:
    """Fraction of raw images for which the CV pipeline isolated >=1 connected
    component whose area passed min_area, overall and per class.
    """
    log_df = pd.read_csv(segmentation_log_path)
    log_df["isolated"] = (log_df["status"] == "ok") & (log_df["largest_area"] >= min_area)

    overall_ria = float(log_df["isolated"].mean())
    per_class_ria = log_df.groupby("label")["isolated"].mean().astype(float).to_dict()
    mean_components = float(log_df["num_components"].mean())
    failed_files = log_df.loc[~log_df["isolated"], "filepath"].tolist()

    return {
        "overall_ria": overall_ria,
        "per_class_ria": per_class_ria,
        "mean_components_per_image": mean_components,
        "failed_files": failed_files,
    }


def evaluate_model(model, test_dataset, class_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Run model.predict over test_dataset and compute classification metrics."""
    y_true = np.concatenate([labels.numpy() for _images, labels in test_dataset])
    predictions = model.predict(test_dataset)
    y_pred = np.argmax(predictions, axis=1)
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    return y_true, y_pred, metrics


def save_metrics_report(metrics: dict, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
