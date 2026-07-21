"""Report-figure visualizations: training curves, confusion matrix heatmap,
pipeline-stage walkthroughs, raw-vs-segmented comparison, and augmentation
sample export.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation.confusion_matrix import save_confusion_matrix_image
from preprocessing import edge_detection, morphology, rgb_to_hsv, thresholding
from preprocessing.segmentation import extract_regions


def plot_training_history(history_path: Path, output_path: Path) -> None:
    """history_path: history.json written by models.train, or a training_log.csv."""
    history_path = Path(history_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.suffix == ".json":
        with open(history_path) as f:
            history = json.load(f)
    else:
        df = pd.read_csv(history_path)
        history = df.to_dict(orient="list")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: list[str], output_path: Path, normalize: bool = True) -> None:
    save_confusion_matrix_image(cm, class_names, output_path, normalize=normalize)


def visualize_pipeline_stages(image_path: Path, output_path: Path) -> None:
    """Original -> saturation channel -> combined threshold mask -> Canny edges
    -> cleaned morphology mask -> final crop with bbox drawn, in one figure.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    hsv = rgb_to_hsv.bgr_to_hsv(bgr)
    gray = rgb_to_hsv.get_segmentation_channel(hsv, "saturation")

    _, otsu_mask = thresholding.otsu_threshold(gray)
    adaptive_mask = thresholding.adaptive_threshold(gray)
    thresh_mask = thresholding.combine_thresholds(otsu_mask, adaptive_mask, mode="or")

    edges = edge_detection.auto_canny(gray)
    edge_mask = edge_detection.edges_to_mask(edges)

    combined = cv2.bitwise_or(thresh_mask, edge_mask)
    cleaned = morphology.clean_mask(combined)

    regions = extract_regions(bgr, cleaned, output_size=config.IMAGE_SIZE, strategy="largest")

    bbox_image = bgr.copy()
    for _crop, stats in regions:
        x, y, w, h = stats["x"], stats["y"], stats["w"], stats["h"]
        cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    panels = [
        (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "Original"),
        (gray, "Saturation channel"),
        (thresh_mask, "Otsu + adaptive threshold"),
        (edge_mask, "Canny edges (dilated)"),
        (cleaned, "Morphology cleaned"),
        (cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB), "Detected region(s)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (panel, title) in zip(axes.flat, panels):
        cmap = None if panel.ndim == 3 else "gray"
        ax.imshow(panel, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def compare_raw_vs_segmented_metrics(raw_metrics_path: Path, segmented_metrics_path: Path, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_metrics_path) as f:
        raw_metrics = json.load(f)
    with open(segmented_metrics_path) as f:
        segmented_metrics = json.load(f)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    raw_values = [raw_metrics[m] for m in metric_names]
    segmented_values = [segmented_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, raw_values, width, label="Raw images")
    ax.bar(x + width / 2, segmented_values, width, label="Segmented images")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title("Raw vs. Segmented: CNN Classification Performance")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_augmentation_samples(
    manifest_path: Path,
    output_dir: Path = config.AUGMENTED_DIR,
    n_per_class: int = 5,
    seed: int = config.RANDOM_SEED,
) -> None:
    """Snapshot a handful of augmented examples per class purely for report
    figures. This is the ONLY writer of dataset/augmented/ -- it is never
    read back by models.train, which augments on-the-fly during training.
    """
    import tensorflow as tf
    from models.train import build_augmentation_pipeline

    output_dir = Path(output_dir)
    manifest = pd.read_csv(manifest_path)
    augmentation = build_augmentation_pipeline()
    rng = np.random.default_rng(seed)

    for label, group in manifest.groupby("label"):
        sample_paths = rng.choice(group["filepath"].to_numpy(), size=min(n_per_class, len(group)), replace=False)
        class_dir = output_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for path in sample_paths:
            image = tf.io.read_file(str(path))
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, config.IMAGE_SIZE)
            augmented = augmentation(tf.expand_dims(image, 0), training=True)[0]
            out_array = cv2.cvtColor(augmented.numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            out_path = class_dir / f"aug_{Path(path).stem}.jpg"
            cv2.imwrite(str(out_path), out_array)

    print(f"Wrote augmentation samples to {output_dir}")
