#!/usr/bin/env python3
# This code is owned by Adwait Dongare and may not be used for any purposes unless explicitly allowed.
"""Convert Feeling of Success H5 samples into PNG files plus a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import flammkuchen as fl
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise SystemExit(
        "Missing dependency: flammkuchen. "
        "Run this script from an environment that has it installed, for example: "
        "conda run -n monty-tactile python scripts/convert_h5_to_png_samples.py "
        "--h5-file dataset-feeling-success/calandra_corl2017_000.h5 "
        "--output-dir processed_samples_test --max-samples 10"
    ) from exc


RGB_FIELDS = {
    "before": "kinectA_rgb_before",
    "during": "kinectA_rgb_during",
    "after": "kinectA_rgb_after",
}

TACTILE_FIELDS = {
    "gelsightA": {
        "before": "gelsightA_before",
        "during": "gelsightA_during",
        "after": "gelsightA_after",
    },
    "gelsightB": {
        "before": "gelsightB_before",
        "during": "gelsightB_during",
        "after": "gelsightB_after",
    },
}

MANIFEST_FIELDNAMES = [
    "sample_id",
    "source_h5",
    "source_index",
    "object_type",
    "is_gripping",
    "metadata_path",
    "rgb_path",
    "tactile_path",
]

PROGRESS_EVERY = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load Feeling of Success H5 files sequentially and extract each sample "
            "into one folder per sample, each containing rgb/, tactile/, and metadata.json."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Directory containing .h5 files. Used when --h5-file is not provided.",
    )
    parser.add_argument(
        "--h5-file",
        type=Path,
        action="append",
        default=[],
        help="Specific H5 file to convert. May be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where sample folders and the manifest CSV are written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to convert across all selected H5 files.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only write the manifest CSV. Do not write PNG files or metadata JSON files.",
    )
    parser.add_argument(
        "--object-name-max-length",
        type=int,
        default=32,
        help="Maximum length of the object_type slug appended to each sample id.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose metadata JSON already exists instead of overwriting them.",
    )
    parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Fail if a target sample metadata JSON already exists. By default files are overwritten.",
    )
    return parser.parse_args()


def selected_h5_files(dataset_dir: Path | None, h5_files: list[Path]) -> list[Path]:
    if h5_files:
        files = h5_files
    elif dataset_dir is not None:
        files = sorted(dataset_dir.glob("*.h5"))
    else:
        raise SystemExit("Provide either --dataset-dir or at least one --h5-file.")

    missing = [path for path in files if not path.is_file()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise SystemExit(f"H5 file(s) not found: {joined}")
    if not files:
        raise SystemExit(f"No .h5 files found in dataset directory: {dataset_dir}")
    return sorted(files)


def normalize_value(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def h5_shorthand(path: Path) -> str:
    match = re.search(r"_(\d+)$", path.stem)
    if match:
        return match.group(1)
    return path.stem


def object_slug(object_type: Any, max_length: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(object_type).strip().lower()).strip("_")
    if not slug:
        slug = "unknown_object"
    return slug[:max_length].rstrip("_")


def sample_id_for(path: Path, index: int, object_type: Any, object_name_max_length: int) -> str:
    return f"{h5_shorthand(path)}_{index:06d}_{object_slug(object_type, object_name_max_length)}"


def as_uint8_image_array(value: Any, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        if array.min() < 0 or array.max() > 255:
            raise ValueError(
                f"{field_name} has dtype {array.dtype} and values outside uint8 range: "
                f"min={array.min()} max={array.max()}"
            )
        array = array.astype(np.uint8)

    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] in {1, 3, 4}:
        return array
    raise ValueError(f"{field_name} has unsupported image shape: {array.shape}")


def save_png(value: Any, output_path: Path, field_name: str) -> None:
    array = as_uint8_image_array(value, field_name)
    image = Image.fromarray(array)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def required_fields(include_images: bool = True) -> set[str]:
    fields = {"object_name", "is_gripping"}
    if not include_images:
        return fields
    fields.update(RGB_FIELDS.values())
    for sensor_fields in TACTILE_FIELDS.values():
        fields.update(sensor_fields.values())
    return fields


def validate_sample(
    sample: dict[str, Any],
    source_h5: Path,
    index: int,
    include_images: bool = True,
) -> None:
    missing = sorted(required_fields(include_images=include_images) - set(sample))
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"{source_h5} sample {index} is missing required field(s): {joined}")


def relative_to_output(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def write_sample(
    sample: dict[str, Any],
    source_h5: Path,
    index: int,
    output_dir: Path,
    skip_existing: bool,
    fail_if_exists: bool,
    object_name_max_length: int,
    manifest_only: bool,
) -> dict[str, Any] | None:
    validate_sample(sample, source_h5, index, include_images=not manifest_only)

    object_type = normalize_value(sample["object_name"])
    is_gripping = bool(normalize_value(sample["is_gripping"]))
    sample_id = sample_id_for(source_h5, index, object_type, object_name_max_length)
    sample_dir = output_dir / sample_id
    rgb_dir = sample_dir / "rgb"
    tactile_dir = sample_dir / "tactile"
    metadata_path = sample_dir / "metadata.json"

    if not manifest_only and metadata_path.exists():
        if fail_if_exists:
            raise FileExistsError(f"Sample already exists: {metadata_path}")
        if skip_existing:
            return None

    manifest_row: dict[str, Any] = {
        "sample_id": sample_id,
        "source_h5": source_h5.as_posix(),
        "source_index": index,
        "object_type": object_type,
        "is_gripping": is_gripping,
        "metadata_path": relative_to_output(metadata_path, output_dir),
        "rgb_path": relative_to_output(rgb_dir, output_dir),
        "tactile_path": relative_to_output(tactile_dir, output_dir),
    }

    if not manifest_only:
        rgb_paths = {}
        for stage, field_name in RGB_FIELDS.items():
            path = rgb_dir / f"{stage}.png"
            save_png(sample[field_name], path, field_name)
            rgb_paths[stage] = relative_to_output(path, output_dir)

        tactile_paths = {}
        for sensor_name, sensor_fields in TACTILE_FIELDS.items():
            tactile_paths[sensor_name] = {}
            for stage, field_name in sensor_fields.items():
                path = tactile_dir / f"{sensor_name}_{stage}.png"
                save_png(sample[field_name], path, field_name)
                tactile_paths[sensor_name][stage] = relative_to_output(path, output_dir)

        metadata = {
            "sample_id": sample_id,
            "source_h5": source_h5.as_posix(),
            "source_index": index,
            "object_type": object_type,
            "is_gripping": is_gripping,
            "rgb_path": manifest_row["rgb_path"],
            "tactile_path": manifest_row["tactile_path"],
            "rgb": rgb_paths,
            "tactile": tactile_paths,
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return manifest_row


def initialize_manifest(manifest_path: Path, append: bool = False) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if append and manifest_path.is_file():
        return
    with manifest_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()


def append_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    with manifest_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_FIELDNAMES)
        writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDNAMES})


def main() -> int:
    args = parse_args()
    if args.skip_existing and args.fail_if_exists:
        raise SystemExit("Use only one of --skip-existing or --fail-if-exists.")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive.")
    if args.object_name_max_length < 1:
        raise SystemExit("--object-name-max-length must be positive.")

    h5_files = selected_h5_files(args.dataset_dir, args.h5_file)
    output_dir = args.output_dir
    manifest_path = output_dir / "manifest.csv"
    initialize_manifest(manifest_path, append=args.skip_existing)

    converted = 0
    skipped = 0

    for h5_path in h5_files:
        if args.max_samples is not None and converted >= args.max_samples:
            break

        print(f"Loading {h5_path}")
        dataset = fl.load(h5_path)

        for index, sample in enumerate(dataset):
            if args.max_samples is not None and converted >= args.max_samples:
                break

            row = write_sample(
                sample=sample,
                source_h5=h5_path,
                index=index,
                output_dir=output_dir,
                skip_existing=args.skip_existing,
                fail_if_exists=args.fail_if_exists,
                object_name_max_length=args.object_name_max_length,
                manifest_only=args.manifest_only,
            )
            if row is None:
                skipped += 1
                continue

            append_manifest_row(manifest_path, row)
            converted += 1
            if converted % PROGRESS_EVERY == 0:
                print(f"Processed {converted} samples")

        del dataset

    action = "Added to manifest" if args.manifest_only else "Converted"
    print(f"{action} {converted} samples into {output_dir}")
    if skipped:
        print(f"Skipped {skipped} existing samples")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
