#!/usr/bin/env python3
# This code is owned by Adwait Dongare and may not be used for any purposes unless explicitly allowed.
"""Generate deterministic Feeling of Success train/eval/test split CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from pathlib import Path

import pandas as pd


FINAL_MANIFEST_FIELDNAMES = [
    "episode_id",
    "dataset_name",
    "task_id",
    "task_name",
    "object_id",
    "object_class",
    "condition_class",
    "clip_type",
    "result",
    "failure_mode",
    "rgb_path",
    "tactile_path",
    "force_path",
    "metadata_path",
    "split_group",
]

SPLIT_ORDER = ["train", "eval", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the single deterministic train/eval/test split used for Feeling of Success "
            "experiments. Intermediate summaries are printed to stdout; only final split CSVs "
            "are written to disk."
        )
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("feeling-of-success/manifest.csv"),
        help="Manifest CSV produced by scripts/convert_h5_to_png_samples.py.",
    )
    parser.add_argument(
        "--visually-difficult-csv",
        type=Path,
        default=Path("possibly-visually-difficult.csv"),
        help="CSV containing the visually challenging object list in an object_name column. Other columns are ignored",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where train.csv, eval.csv, and test.csv are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle split groups before assigning splits.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split share. Fractions or whole-number shares are both accepted.")
    parser.add_argument("--eval-ratio", type=float, default=0.15, help="Eval split share. Fractions or whole-number shares are both accepted.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split share. Fractions or whole-number shares are both accepted.")
    parser.add_argument(
        "--split-group-strategy",
        choices=["object_type", "object_type_dataset"],
        default="object_type",
        help="Grouping unit assigned atomically to one split.",
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "success"}:
        return True
    if normalized in {"false", "0", "no", "failure"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value!r}")


def h5_number(source_h5: object) -> str:
    match = re.search(r"_(\d+)$", Path(str(source_h5)).stem)
    if not match:
        raise ValueError(f"Could not find trailing _<number> in source_h5: {source_h5}")
    return match.group(1)


def display_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.to_string(index=False))


def normalize_split_ratios(raw_split_ratios: dict[str, float]) -> dict[str, float]:
    for split_name, value in raw_split_ratios.items():
        if not math.isfinite(value):
            raise ValueError(f"{split_name} ratio must be finite; got {value}")
        if value < 0:
            raise ValueError(f"{split_name} ratio must be non-negative; got {value}")

    ratio_sum = sum(raw_split_ratios.values())
    if ratio_sum <= 0:
        raise ValueError(f"At least one split ratio must be greater than zero; got {raw_split_ratios}")

    return {split_name: value / ratio_sum for split_name, value in raw_split_ratios.items()}


def load_manifest(manifest_csv: Path, split_group_strategy: str) -> pd.DataFrame:
    required_columns = {
        "sample_id",
        "source_h5",
        "source_index",
        "object_type",
        "is_gripping",
        "metadata_path",
        "rgb_path",
        "tactile_path",
    }

    manifest_df = pd.read_csv(manifest_csv)
    missing_columns = required_columns.difference(manifest_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required manifest column(s): {sorted(missing_columns)}")

    manifest_df = manifest_df.copy()
    manifest_df["source_index"] = manifest_df["source_index"].astype(int)
    manifest_df["is_gripping"] = manifest_df["is_gripping"].map(parse_bool)
    manifest_df["episode_id"] = manifest_df["source_h5"].map(h5_number)

    if split_group_strategy == "object_type":
        manifest_df["split_group"] = manifest_df["object_type"]
    elif split_group_strategy == "object_type_dataset":
        manifest_df["split_group"] = manifest_df["object_type"].astype(str) + "_" + manifest_df["episode_id"].astype(str)
    else:
        raise ValueError(f"Unknown split group strategy: {split_group_strategy}")

    return manifest_df


def load_visually_challenging_objects(csv_path: Path) -> set[str]:
    visual_df = pd.read_csv(csv_path)
    if "object_name" not in visual_df.columns:
        raise ValueError(f"{csv_path} is missing required column: object_name")
    return set(visual_df["object_name"].dropna().astype(str).unique())


def add_visually_challenging_flag(manifest_df: pd.DataFrame, visually_challenging_set: set[str]) -> pd.DataFrame:
    available_object_types = set(manifest_df["object_type"])
    missing_objects = sorted(visually_challenging_set - available_object_types)
    if missing_objects:
        raise ValueError(f"Visually challenging object types not found in dataset: {missing_objects}")

    manifest_df = manifest_df.copy()
    manifest_df["is_visually_challenging"] = manifest_df["object_type"].isin(visually_challenging_set)
    return manifest_df


def summarize_dataset(manifest_df: pd.DataFrame, visually_challenging_set: set[str]) -> pd.DataFrame:
    success_count = int(manifest_df["is_gripping"].sum())
    visually_challenging_samples = int(manifest_df["is_visually_challenging"].sum())
    return pd.DataFrame(
        [
            {
                "samples": len(manifest_df),
                "objectTypes": manifest_df["object_type"].nunique(),
                "files": manifest_df["episode_id"].nunique(),
                "success": success_count,
                "failure": len(manifest_df) - success_count,
                "successRate": round(success_count / len(manifest_df), 4),
                "visuallyChallengingSourceObjects": len(visually_challenging_set),
                "visuallyChallengingObjectsInManifest": manifest_df.loc[
                    manifest_df["is_visually_challenging"], "object_type"
                ].nunique(),
                "visuallyChallengingSamples": visually_challenging_samples,
                "visuallyChallengingSampleFraction": round(visually_challenging_samples / len(manifest_df), 4),
            }
        ]
    )


def build_split_group_table(manifest_df: pd.DataFrame, split_group_strategy: str) -> pd.DataFrame:
    split_group_table = (
        manifest_df.groupby("split_group", as_index=False)
        .agg(
            object_type=("object_type", "first"),
            episode_id=("episode_id", "first"),
            is_gripping_success_count=("is_gripping", "sum"),
            total_samples=("sample_id", "count"),
            source_h5_files=("source_h5", lambda values: ";".join(sorted({Path(value).name for value in values}))),
        )
    )
    if split_group_strategy == "object_type":
        split_group_table["episode_id"] = ""
    split_group_table["is_gripping_success_count"] = split_group_table["is_gripping_success_count"].astype(int)
    split_group_table["is_gripping_failure_count"] = (
        split_group_table["total_samples"] - split_group_table["is_gripping_success_count"]
    )
    return split_group_table[
        [
            "split_group",
            "object_type",
            "episode_id",
            "is_gripping_success_count",
            "is_gripping_failure_count",
            "total_samples",
            "source_h5_files",
        ]
    ].sort_values("split_group")


def assign_splits(
    manifest_df: pd.DataFrame,
    split_group_table: pd.DataFrame,
    split_ratios: dict[str, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled_records = split_group_table.to_dict("records")
    random.Random(seed).shuffle(shuffled_records)
    shuffled_groups = pd.DataFrame(shuffled_records)

    total_samples = int(shuffled_groups["total_samples"].sum())
    train_cutoff = total_samples * split_ratios["train"]
    eval_cutoff = train_cutoff + total_samples * split_ratios["eval"]

    assigned_splits = []
    assigned_samples = 0
    for total in shuffled_groups["total_samples"]:
        if assigned_samples < train_cutoff:
            split = "train"
        elif assigned_samples < eval_cutoff:
            split = "eval"
        else:
            split = "test"
        assigned_splits.append(split)
        assigned_samples += int(total)

    assigned_group_table = shuffled_groups.assign(split=assigned_splits).sort_values("split_group").reset_index(drop=True)
    split_lookup = dict(zip(assigned_group_table["split_group"], assigned_group_table["split"]))

    assigned_manifest_df = manifest_df.copy()
    assigned_manifest_df["split"] = assigned_manifest_df["split_group"].map(split_lookup)
    return assigned_manifest_df, assigned_group_table


def build_final_split_table(manifest_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": manifest_df["episode_id"],
            "dataset_name": "feeling_of_success",
            "task_id": manifest_df["source_index"],
            "task_name": "grasp_outcome",
            "object_id": manifest_df["object_type"],
            "object_class": manifest_df["is_visually_challenging"].map(
                {True: "visually_challenging", False: "normal"}
            ),
            "condition_class": "",
            "clip_type": "before-during-after-images",
            "result": manifest_df["is_gripping"],
            "failure_mode": manifest_df["is_gripping"].map({True: "", False: "unknown"}),
            "rgb_path": manifest_df["rgb_path"],
            "tactile_path": manifest_df["tactile_path"],
            "force_path": "",
            "metadata_path": manifest_df["metadata_path"],
            "split_group": manifest_df["split_group"],
            "split": manifest_df["split"],
        }
    )


def summarize_split_samples(final_split_df: pd.DataFrame) -> pd.DataFrame:
    total_samples = len(final_split_df)
    row_split_summary = (
        final_split_df.groupby("split", as_index=False)
        .agg(
            samples=("task_id", "count"),
            success=("result", "sum"),
            splitGroups=("split_group", "nunique"),
            objects=("object_id", "nunique"),
        )
    )
    row_split_summary["success"] = row_split_summary["success"].astype(int)
    row_split_summary["failure"] = row_split_summary["samples"] - row_split_summary["success"]
    row_split_summary["sampleFractionOverTotalSamples"] = (row_split_summary["samples"] / total_samples).round(4)
    return row_split_summary[
        ["split", "samples", "sampleFractionOverTotalSamples", "success", "failure", "splitGroups", "objects"]
    ]


def summarize_split_objects(final_split_df: pd.DataFrame) -> pd.DataFrame:
    total_samples = len(final_split_df)
    total_objects = final_split_df["object_id"].nunique()
    object_summary = (
        final_split_df.groupby("split", as_index=False)
        .agg(samples=("task_id", "count"), objects=("object_id", "nunique"))
    )
    object_summary["sampleFractionOverTotalSamples"] = (object_summary["samples"] / total_samples).round(4)
    object_summary["objectFractionOverTotalObjects"] = (object_summary["objects"] / total_objects).round(4)
    return object_summary[
        ["split", "objects", "objectFractionOverTotalObjects", "samples", "sampleFractionOverTotalSamples"]
    ]


def summarize_visually_challenging(final_split_df: pd.DataFrame) -> pd.DataFrame:
    total_samples = len(final_split_df)
    total_objects = final_split_df["object_id"].nunique()
    visual_df = final_split_df.assign(isVisuallyChallenging=final_split_df["object_class"].eq("visually_challenging"))

    visual_summary = (
        visual_df.groupby("split", as_index=False)
        .agg(
            totalSamples=("task_id", "count"),
            totalObjects=("object_id", "nunique"),
            visuallyChallengingSamples=("isVisuallyChallenging", "sum"),
            visuallyChallengingObjects=(
                "object_id",
                lambda s: s[visual_df.loc[s.index, "isVisuallyChallenging"]].nunique(),
            ),
            visuallyChallengingSuccess=(
                "result",
                lambda s: int(s[visual_df.loc[s.index, "isVisuallyChallenging"]].sum()),
            ),
        )
    )
    visual_summary["visuallyChallengingSamples"] = visual_summary["visuallyChallengingSamples"].astype(int)
    visual_summary["visuallyChallengingFailure"] = (
        visual_summary["visuallyChallengingSamples"] - visual_summary["visuallyChallengingSuccess"]
    )
    visual_summary["visuallyChallengingSampleFractionOfSplit"] = (
        visual_summary["visuallyChallengingSamples"] / visual_summary["totalSamples"]
    ).round(4)
    visual_summary["visuallyChallengingSampleFractionOverTotalSamples"] = (
        visual_summary["visuallyChallengingSamples"] / total_samples
    ).round(4)
    visual_summary["visuallyChallengingObjectFractionOfSplit"] = (
        visual_summary["visuallyChallengingObjects"] / visual_summary["totalObjects"]
    ).round(4)
    visual_summary["visuallyChallengingObjectFractionOverTotalObjects"] = (
        visual_summary["visuallyChallengingObjects"] / total_objects
    ).round(4)
    return visual_summary[
        [
            "split",
            "totalSamples",
            "totalObjects",
            "visuallyChallengingSamples",
            "visuallyChallengingSampleFractionOfSplit",
            "visuallyChallengingSampleFractionOverTotalSamples",
            "visuallyChallengingObjects",
            "visuallyChallengingObjectFractionOfSplit",
            "visuallyChallengingObjectFractionOverTotalObjects",
            "visuallyChallengingSuccess",
            "visuallyChallengingFailure",
        ]
    ]


def write_split_csvs(final_split_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_ORDER:
        output_path = output_dir / f"{split_name}.csv"
        split_df = final_split_df[final_split_df["split"].eq(split_name)][FINAL_MANIFEST_FIELDNAMES]
        split_df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
        print(f"Wrote {len(split_df):,} rows to {output_path}")


def main() -> int:
    args = parse_args()
    requested_split_shares = {
        "train": args.train_ratio,
        "eval": args.eval_ratio,
        "test": args.test_ratio,
    }
    try:
        split_ratios = normalize_split_ratios(requested_split_shares)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    manifest_df = load_manifest(args.manifest_csv, args.split_group_strategy)
    visually_challenging_set = load_visually_challenging_objects(args.visually_difficult_csv)
    manifest_df = add_visually_challenging_flag(manifest_df, visually_challenging_set)

    display_table("Dataset summary", summarize_dataset(manifest_df, visually_challenging_set))

    split_group_table = build_split_group_table(manifest_df, args.split_group_strategy)
    assigned_manifest_df, assigned_group_table = assign_splits(
        manifest_df=manifest_df,
        split_group_table=split_group_table,
        split_ratios=split_ratios,
        seed=args.seed,
    )
    final_split_df = build_final_split_table(assigned_manifest_df)

    group_summary = (
        assigned_group_table.groupby("split", as_index=False)
        .agg(assignedGroups=("split_group", "count"), assignedSamples=("total_samples", "sum"))
    )
    group_summary["requestedShare"] = group_summary["split"].map(requested_split_shares)
    group_summary["targetFraction"] = group_summary["split"].map(split_ratios)
    group_summary["assignedSampleFraction"] = (group_summary["assignedSamples"] / len(final_split_df)).round(4)

    display_table("Split group assignment summary", group_summary)
    display_table("Samples by split", summarize_split_samples(final_split_df))
    display_table("Objects by split", summarize_split_objects(final_split_df))
    display_table("Visually challenging samples and objects by split", summarize_visually_challenging(final_split_df))

    write_split_csvs(final_split_df, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
