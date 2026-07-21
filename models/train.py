"""Train the custom CNN on either dataset/raw or dataset/segmented.

Manifest-driven so the raw-vs-segmented comparison uses identical
train/val/test membership per source image (see
preprocessing.segmentation.build_manifest). Running this script once per
data directory produces two directly comparable results/<run-name>/ folders.

Usage:
    python -m models.train --data-dir dataset/raw --manifest dataset/raw_manifest.csv --run-name raw_v1
    python -m models.train --data-dir dataset/segmented --manifest dataset/segmented_manifest.csv --run-name segmented_v1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from evaluation import confusion_matrix as cm_module
from evaluation import metrics as metrics_module
from models.cnn_model import build_cnn

AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation_pipeline() -> keras.Sequential:
    """Augmentation applied only to the training split, never val/test."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )


def _load_and_resize(path, label, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    return image, label


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    image_size: tuple[int, int],
    batch_size: int,
    training: bool,
    seed: int,
    augmentation: keras.Sequential | None = None,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: _load_and_resize(p, l, image_size), num_parallel_calls=AUTOTUNE)
    if training and augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(1000, seed=seed)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def load_manifest(manifest_path: Path) -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(manifest_path)
    class_names = sorted(manifest["label"].unique())
    label_to_index = {name: i for i, name in enumerate(class_names)}
    manifest["label_idx"] = manifest["label"].map(label_to_index)
    return manifest, class_names


def train(args: argparse.Namespace) -> None:
    manifest, class_names = load_manifest(args.manifest)
    num_classes = len(class_names)
    image_size = (args.image_size, args.image_size)

    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]

    augmentation = build_augmentation_pipeline()
    train_ds = make_dataset(
        train_df["filepath"].to_numpy(), train_df["label_idx"].to_numpy(),
        image_size, args.batch_size, training=True, seed=args.seed, augmentation=augmentation,
    )
    val_ds = make_dataset(
        val_df["filepath"].to_numpy(), val_df["label_idx"].to_numpy(),
        image_size, args.batch_size, training=False, seed=args.seed,
    )
    test_ds = make_dataset(
        test_df["filepath"].to_numpy(), test_df["label_idx"].to_numpy(),
        image_size, args.batch_size, training=False, seed=args.seed,
    )

    class_weight = None
    if args.class_weight:
        classes = np.unique(train_df["label_idx"])
        weights = compute_class_weight("balanced", classes=classes, y=train_df["label_idx"].to_numpy())
        class_weight = dict(zip(classes.tolist(), weights.tolist()))

    run_dir = config.RESULTS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_cnn(input_shape=(*image_size, 3), num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            run_dir / "best_model.keras", save_best_only=True, monitor="val_accuracy"
        ),
        keras.callbacks.EarlyStopping(
            patience=args.patience, restore_best_weights=True, monitor="val_accuracy"
        ),
        keras.callbacks.CSVLogger(run_dir / "training_log.csv"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    model.save(run_dir / "final_model.keras")
    with open(run_dir / "history.json", "w") as f:
        json.dump(history.history, f)
    with open(run_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    y_true, y_pred, class_metrics = metrics_module.evaluate_model(model, test_ds, class_names)
    metrics_module.save_metrics_report(class_metrics, run_dir / "metrics.json")

    cm = cm_module.compute_confusion_matrix(y_true, y_pred, class_names)
    cm_module.save_confusion_matrix_csv(cm, class_names, run_dir / "confusion_matrix.csv")
    cm_module.save_confusion_matrix_image(cm, class_names, run_dir / "confusion_matrix.png")

    print(f"Run '{args.run_name}' complete. Results written to {run_dir}")
    print(json.dumps({k: v for k, v in class_metrics.items() if k != "per_class"}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bacteria CNN")
    parser.add_argument("--data-dir", type=Path, required=True,
                         help="dataset/raw or dataset/segmented (used to derive default manifest path)")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE[0])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--class-weight", dest="class_weight", action="store_true", default=True)
    parser.add_argument("--no-class-weight", dest="class_weight", action="store_false")

    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = args.data_dir.parent / f"{args.data_dir.name}_manifest.csv"
    return args


if __name__ == "__main__":
    tf.random.set_seed(config.RANDOM_SEED)
    train(parse_args())
