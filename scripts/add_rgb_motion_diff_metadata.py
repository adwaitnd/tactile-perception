#!/usr/bin/env python3
# This code is owned by Adwait Dongare and may not be used for any purposes unless explicitly allowed.
"""Add deterministic RGB before/during motion-diff boxes to sample metadata.

The Feeling-of-Success RGB view is wide and the robot/object interaction tends
to create the strongest before->during motion cue near the object workspace.
This script computes a simple image-difference mask from allowed pre-outcome
frames only, then stores a full-height square crop box centered on that motion
in each sample's metadata.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "dataset-limited"


@dataclass(frozen=True)
class MotionBox:
    box_xyxy: Tuple[int, int, int, int]
    raw_box_xyxy: Optional[Tuple[int, int, int, int]]
    image_size: Tuple[int, int]
    mask_area: int
    score: float
    fallback: bool


def load_manifest_rows(dataset_root: Path) -> List[Dict[str, str]]:
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Expected dataset manifest at {manifest_path}. "
            "This preprocessing step is dataset-level and does not use split CSVs."
        )
    rows: List[Dict[str, str]] = []
    seen = set()
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            metadata_path = row["metadata_path"]
            if metadata_path in seen:
                continue
            seen.add(metadata_path)
            rows.append(row)
    return rows


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def full_height_square_box(
    center_x: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    side = min(image_height, image_width)
    x1 = int(round(center_x - side / 2.0))
    x1 = clamp(x1, 0, image_width - side)
    return x1, 0, x1 + side, side


def compute_motion_box(
    before_path: Path,
    during_path: Path,
    blur_kernel: int = 7,
    threshold_percentile: float = 98.0,
    min_area: int = 150,
) -> MotionBox:
    before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    during = cv2.imread(str(during_path), cv2.IMREAD_COLOR)
    if before is None:
        raise FileNotFoundError(before_path)
    if during is None:
        raise FileNotFoundError(during_path)
    if before.shape != during.shape:
        raise ValueError(f"Shape mismatch: {before_path} {before.shape} vs {during_path} {during.shape}")

    image_height, image_width = before.shape[:2]
    diff = cv2.absdiff(before, during)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    threshold = float(np.percentile(gray, threshold_percentile))
    threshold = max(threshold, 8.0)
    mask = (gray >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not contours:
        fallback_box = full_height_square_box(image_width / 2.0, image_width, image_height)
        return MotionBox(
            box_xyxy=fallback_box,
            raw_box_xyxy=None,
            image_size=(image_width, image_height),
            mask_area=0,
            score=0.0,
            fallback=True,
        )

    points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(points)
    raw_box = (int(x), int(y), int(x + w), int(y + h))
    padded_box = full_height_square_box(x + w / 2.0, image_width, image_height)
    mask_area = int(sum(cv2.contourArea(contour) for contour in contours))
    score = mask_area / float(image_width * image_height)
    return MotionBox(
        box_xyxy=padded_box,
        raw_box_xyxy=raw_box,
        image_size=(image_width, image_height),
        mask_area=mask_area,
        score=score,
        fallback=False,
    )


def update_metadata(
    dataset_root: Path,
    row: Dict[str, str],
    args: argparse.Namespace,
) -> MotionBox:
    metadata_path = dataset_root / row["metadata_path"]
    rgb_dir = dataset_root / row["rgb_path"]
    before_path = rgb_dir / "before.png"
    during_path = rgb_dir / "during.png"

    motion_box = compute_motion_box(
        before_path=before_path,
        during_path=during_path,
        blur_kernel=args.blur_kernel,
        threshold_percentile=args.threshold_percentile,
        min_area=args.min_area,
    )

    with metadata_path.open() as f:
        metadata = json.load(f)

    metadata["rgb_motion_diff"] = {
        "version": 1,
        "method": "absdiff_before_during_percentile_contours",
        "crop_policy": "full_height_square_centered_on_motion_x",
        "uses_frames": ["rgb/before.png", "rgb/during.png"],
        "box_xyxy": list(motion_box.box_xyxy),
        "raw_box_xyxy": list(motion_box.raw_box_xyxy) if motion_box.raw_box_xyxy else None,
        "image_size": {
            "width": motion_box.image_size[0],
            "height": motion_box.image_size[1],
        },
        "mask_area": motion_box.mask_area,
        "score": motion_box.score,
        "fallback": motion_box.fallback,
        "params": {
            "blur_kernel": args.blur_kernel,
            "threshold_percentile": args.threshold_percentile,
            "min_area": args.min_area,
        },
    }

    if not args.dry_run:
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
    return motion_box


def save_overlay(
    dataset_root: Path,
    row: Dict[str, str],
    motion_box: MotionBox,
    output_dir: Path,
) -> None:
    rgb_dir = dataset_root / row["rgb_path"]
    before = cv2.imread(str(rgb_dir / "before.png"), cv2.IMREAD_COLOR)
    during = cv2.imread(str(rgb_dir / "during.png"), cv2.IMREAD_COLOR)
    if before is None or during is None:
        return
    panel = np.concatenate([before, during], axis=1)
    width = before.shape[1]

    x1, y1, x2, y2 = motion_box.box_xyxy
    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(panel, (x1 + width, y1), (x2 + width, y2), (0, 255, 0), 2)
    if motion_box.raw_box_xyxy is not None:
        rx1, ry1, rx2, ry2 = motion_box.raw_box_xyxy
        cv2.rectangle(panel, (rx1, ry1), (rx2, ry2), (0, 180, 255), 1)
        cv2.rectangle(panel, (rx1 + width, ry1), (rx2 + width, ry2), (0, 180, 255), 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(row['metadata_path']).parent.name}_motion_diff.jpg"
    cv2.imwrite(str(output_path), panel)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset directory containing manifest.csv and per-sample folders; metadata JSONs under this root are updated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N manifest rows, useful for checking boxes before updating the whole dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print motion boxes without writing rgb_motion_diff into metadata.json files.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=7,
        help="Gaussian blur size for the before/during difference image; larger values ignore tiny noise and make boxes more stable.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=98.0,
        help="Percentile used to keep only strongest motion pixels; higher values make tighter boxes, lower values include more motion.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=150,
        help="Minimum contour area kept as real motion; increasing this filters small flicker/noise, decreasing it catches subtler movement.",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="Optional directory for before/during side-by-side JPEG overlays showing raw motion and final full-height square crop boxes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest_rows(args.dataset_root)
    if args.limit is not None:
        rows = rows[: args.limit]

    fallback_count = 0
    for i, row in enumerate(rows, start=1):
        motion_box = update_metadata(args.dataset_root, row, args)
        if args.overlay_dir is not None:
            save_overlay(args.dataset_root, row, motion_box, args.overlay_dir)
        fallback_count += int(motion_box.fallback)
        print(
            f"{i:04d}/{len(rows):04d} {row['metadata_path']} "
            f"box={motion_box.box_xyxy} score={motion_box.score:.4f} fallback={motion_box.fallback}"
        )
    print(f"Processed {len(rows)} samples; fallback boxes: {fallback_count}")
    if args.dry_run:
        print("Dry run only; metadata files were not modified.")


if __name__ == "__main__":
    main()
