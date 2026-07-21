"""Segmentation pipeline orchestrator + dataset validation/manifest building.

Composes rgb_to_hsv -> thresholding -> edge_detection -> morphology into a
single mask-building step, extracts bacterial regions via connected
component analysis, and drives the whole raw/ -> segmented/ conversion plus
the manifest CSVs consumed by models/train.py.

CLI usage:
    python -m preprocessing.segmentation validate --raw-dir dataset/raw
    python -m preprocessing.segmentation segment  --raw-dir dataset/raw --segmented-dir dataset/segmented
    python -m preprocessing.segmentation manifest  --data-dir dataset/raw --out dataset/raw_manifest.csv
    python -m preprocessing.segmentation manifest  --data-dir dataset/segmented --out dataset/segmented_manifest.csv \
        --source-manifest dataset/raw_manifest.csv
    python -m preprocessing.segmentation all
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from preprocessing import edge_detection, morphology, rgb_to_hsv, thresholding

_REGION_SUFFIX_RE = re.compile(r"_region\d+$")


def build_segmentation_mask(
    bgr_image: np.ndarray,
    channel: str = "saturation",
    canny_sigma: float = 0.33,
) -> np.ndarray:
    """Run the full CV pipeline (HSV -> threshold -> Canny -> morphology) and
    return a single cleaned binary mask ready for connected component analysis.
    """
    hsv = rgb_to_hsv.bgr_to_hsv(bgr_image)
    gray = rgb_to_hsv.get_segmentation_channel(hsv, channel)

    _, otsu_mask = thresholding.otsu_threshold(gray)
    adaptive_mask = thresholding.adaptive_threshold(gray)
    thresh_mask = thresholding.combine_thresholds(otsu_mask, adaptive_mask, mode="or")

    edges = edge_detection.auto_canny(gray, sigma=canny_sigma)
    edge_mask = edge_detection.edges_to_mask(edges)

    combined = cv2.bitwise_or(thresh_mask, edge_mask)
    return morphology.clean_mask(combined)


def extract_regions(
    bgr_image: np.ndarray,
    mask: np.ndarray,
    min_area: int = config.MIN_CONTOUR_AREA,
    output_size: tuple[int, int] = config.IMAGE_SIZE,
    padding: int = 5,
    strategy: str = "largest",
) -> list[tuple[np.ndarray, dict]]:
    """Extract bacterial regions from bgr_image using connected components on mask.

    Crops are taken from the ORIGINAL image (not the mask) so color/texture
    information is preserved for the CNN.

    strategy="largest" (default): keep only the single largest-area component,
    so segmented/ has one crop per source image and stays directly comparable
    in size to raw/ for the raw-vs-segmented accuracy study.
    strategy="all": keep every component above min_area (more samples, but
    breaks that direct comparability -- opt in deliberately).
    """
    if strategy not in ("largest", "all"):
        raise ValueError(f"strategy must be 'largest' or 'all', got {strategy!r}")

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    h, w = bgr_image.shape[:2]

    candidates = []
    for label in range(1, num_labels):  # skip label 0 (background)
        x, y, bw, bh, area = stats[label]
        if area < min_area:
            continue
        candidates.append((int(x), int(y), int(bw), int(bh), int(area)))

    if not candidates:
        return []

    if strategy == "largest":
        candidates = [max(candidates, key=lambda c: c[4])]

    results = []
    for x, y, bw, bh, area in candidates:
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w, x + bw + padding)
        y1 = min(h, y + bh + padding)
        crop = bgr_image[y0:y1, x0:x1]
        resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)
        results.append((resized, {"x": x, "y": y, "w": bw, "h": bh, "area": area}))

    return results


def process_image(
    filepath: Path,
    output_dir: Path,
    min_area: int = config.MIN_CONTOUR_AREA,
    output_size: tuple[int, int] = config.IMAGE_SIZE,
    strategy: str = "largest",
) -> dict:
    """Segment a single image and write region crop(s) into output_dir.

    Returns a log row dict: filepath, num_components, largest_area, status.
    """
    bgr_image = cv2.imread(str(filepath))
    if bgr_image is None:
        return {
            "filepath": str(filepath),
            "num_components": 0,
            "largest_area": 0,
            "status": "read_error",
        }

    mask = build_segmentation_mask(bgr_image)
    regions = extract_regions(bgr_image, mask, min_area=min_area, output_size=output_size, strategy=strategy)

    if not regions:
        return {
            "filepath": str(filepath),
            "num_components": 0,
            "largest_area": 0,
            "status": "no_region_found",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    largest_area = max(r[1]["area"] for r in regions)
    for i, (crop, _stats) in enumerate(regions):
        out_path = output_dir / f"{filepath.stem}_region{i}.jpg"
        cv2.imwrite(str(out_path), crop)

    return {
        "filepath": str(filepath),
        "num_components": len(regions),
        "largest_area": largest_area,
        "status": "ok",
    }


def process_dataset(
    raw_dir: Path,
    segmented_dir: Path,
    min_area: int = config.MIN_CONTOUR_AREA,
    output_size: tuple[int, int] = config.IMAGE_SIZE,
    strategy: str = "largest",
    log_path: Path | None = None,
) -> pd.DataFrame:
    """Walk raw_dir/<species>/*.ext, segment each image, write crops to
    segmented_dir/<species>/, and log per-image results.

    The log is read directly by evaluation.metrics.compute_region_isolation_accuracy,
    so re-running the CV pipeline a second time at evaluation is unnecessary.
    """
    raw_dir = Path(raw_dir)
    segmented_dir = Path(segmented_dir)
    if log_path is None:
        log_path = config.RESULTS_DIR / "logs" / "segmentation_log.csv"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    species_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    rows = []
    for species_dir in tqdm(species_dirs, desc="species"):
        label = species_dir.name
        image_paths = sorted(
            p for p in species_dir.iterdir()
            if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS
        )
        for image_path in tqdm(image_paths, desc=label, leave=False):
            row = process_image(
                image_path,
                segmented_dir / label,
                min_area=min_area,
                output_size=output_size,
                strategy=strategy,
            )
            row["label"] = label
            rows.append(row)

    log_df = pd.DataFrame(rows)
    log_df.to_csv(log_path, index=False)
    return log_df


def validate_raw_dataset(raw_dir: Path) -> dict[str, int]:
    """Confirm raw_dir has class subfolders and return per-class image counts,
    warning on empty folders or unexpected file extensions.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Manually download the Mendeley dataset "
            "(https://data.mendeley.com/datasets/cvkgfzp7ck/1) and extract it "
            "here as <species_name>/*.jpg subfolders."
        )

    species_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if not species_dirs:
        raise ValueError(f"{raw_dir} has no class subfolders.")

    counts = {}
    for species_dir in species_dirs:
        all_files = [p for p in species_dir.iterdir() if p.is_file()]
        images = [p for p in all_files if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS]
        bad_files = [p for p in all_files if p.suffix.lower() not in config.VALID_IMAGE_EXTENSIONS]

        if not images:
            print(f"WARNING: {species_dir.name} has no valid images.")
        if bad_files:
            print(f"WARNING: {species_dir.name} has {len(bad_files)} non-image file(s): "
                  f"{[p.name for p in bad_files]}")

        counts[species_dir.name] = len(images)

    print(f"Found {len(counts)} classes, {sum(counts.values())} images total.")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    return counts


def _strip_region_suffix(stem: str) -> str:
    return _REGION_SUFFIX_RE.sub("", stem)


def build_manifest(
    data_dir: Path,
    manifest_path: Path,
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    source_manifest: Path | None = None,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Build a filepath,label,split manifest CSV for data_dir/<species>/*.ext.

    If source_manifest is given (e.g. the raw manifest, when building the
    segmented manifest), split assignment is inherited by matching stripped
    filename stems ("ecoli_003_region0" -> "ecoli_003") instead of being
    recomputed. This guarantees the raw-vs-segmented comparison trains/tests
    on identical membership per original source image -- recomputing an
    independent stratified split here would confound that comparison.
    """
    data_dir = Path(data_dir)
    manifest_path = Path(manifest_path)

    rows = []
    for species_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        label = species_dir.name
        for image_path in sorted(species_dir.iterdir()):
            if image_path.suffix.lower() not in config.VALID_IMAGE_EXTENSIONS:
                continue
            rows.append({
                "filepath": str(image_path),
                "label": label,
                "source_id": _strip_region_suffix(image_path.stem),
            })

    manifest = pd.DataFrame(rows)

    if source_manifest is not None:
        source_df = pd.read_csv(source_manifest)
        split_lookup = source_df.drop_duplicates("source_id").set_index("source_id")["split"]
        manifest["split"] = manifest["source_id"].map(split_lookup)
        missing = manifest["split"].isna().sum()
        if missing:
            print(f"WARNING: {missing} row(s) had no matching source_id in {source_manifest}; "
                  "assigning them to 'train'.")
            manifest["split"] = manifest["split"].fillna("train")
    else:
        train_ratio, val_ratio, test_ratio = split_ratios
        train_df, temp_df = train_test_split(
            manifest, test_size=(val_ratio + test_ratio), stratify=manifest["label"], random_state=seed
        )
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df, test_size=relative_test_ratio, stratify=temp_df["label"], random_state=seed
        )
        train_df = train_df.assign(split="train")
        val_df = val_df.assign(split="val")
        test_df = test_df.assign(split="test")
        manifest = pd.concat([train_df, val_df, test_df]).sort_index()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote manifest with {len(manifest)} rows to {manifest_path}")
    print(manifest["split"].value_counts())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_validate = subparsers.add_parser("validate", help="Validate dataset/raw layout")
    p_validate.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)

    p_segment = subparsers.add_parser("segment", help="Run CV pipeline: raw -> segmented")
    p_segment.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    p_segment.add_argument("--segmented-dir", type=Path, default=config.SEGMENTED_DIR)
    p_segment.add_argument("--min-area", type=int, default=config.MIN_CONTOUR_AREA)
    p_segment.add_argument("--strategy", choices=["largest", "all"], default="largest")

    p_manifest = subparsers.add_parser("manifest", help="Build a filepath,label,split manifest CSV")
    p_manifest.add_argument("--data-dir", type=Path, required=True)
    p_manifest.add_argument("--out", type=Path, required=True)
    p_manifest.add_argument("--source-manifest", type=Path, default=None)

    subparsers.add_parser("all", help="validate + segment + build both manifests")

    args = parser.parse_args()

    if args.command == "validate":
        validate_raw_dataset(args.raw_dir)
    elif args.command == "segment":
        process_dataset(
            args.raw_dir, args.segmented_dir, min_area=args.min_area, strategy=args.strategy
        )
    elif args.command == "manifest":
        build_manifest(args.data_dir, args.out, source_manifest=args.source_manifest)
    elif args.command == "all":
        validate_raw_dataset(config.RAW_DIR)
        process_dataset(config.RAW_DIR, config.SEGMENTED_DIR)
        raw_manifest_path = config.DATASET_DIR / "raw_manifest.csv"
        segmented_manifest_path = config.DATASET_DIR / "segmented_manifest.csv"
        build_manifest(config.RAW_DIR, raw_manifest_path)
        build_manifest(config.SEGMENTED_DIR, segmented_manifest_path, source_manifest=raw_manifest_path)


if __name__ == "__main__":
    main()
