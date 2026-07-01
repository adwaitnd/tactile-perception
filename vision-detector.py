#!/usr/bin/env python3
"""Vision-only grasp outcome detector over during/before RGB images.

The detector uses timm's vit_base_patch16_224 as a frozen RGB encoder by
default. Each sample loads the RGB during and before frames, runs both through
the same ViT, concatenates their embeddings, and trains a small MLP head.
Post-outcome after images are intentionally unsupported.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision import transforms
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    Subset = None
    transforms = None


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "vit_base_patch16_224.augreg_in21k"
RGB_IMAGE_SIZE = 224
DEFAULT_SEED = 42

PairMode = Literal["during-before"]


@dataclass(frozen=True)
class VisionSample:
    row: Dict[str, str]
    sample_dir: Path
    rgb_dir: Path
    label: int


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_torch_generator(seed: int) -> Any:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def require_torch() -> None:
    if torch is None or transforms is None:
        raise RuntimeError("This script needs torch and torchvision installed.")


def resolve_device(device: str = "auto") -> Any:
    require_torch()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested --device {device}, but CUDA is not available.")
        return torch.device(device)
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device {device!r}. Use auto, cpu, mps, cuda, or cuda:N.")


def move_batch_to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_summary_writer(logdir: Path, run_name: Optional[str], args: argparse.Namespace):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging requires the tensorboard package. Install it with `pip install tensorboard`."
        ) from exc

    run_name = run_name or datetime.now().strftime("vision-%Y%m%d-%H%M%S")
    run_dir = logdir / run_name
    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_text(
        "config/args",
        "\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())),
        global_step=0,
    )
    print(f"TensorBoard logging to {run_dir}")
    print(f"Launch with: tensorboard --logdir {logdir} --host 0.0.0.0")
    return writer, run_dir


def bool_label(value: str) -> int:
    value_norm = value.strip().lower()
    if value_norm in {"true", "1", "success", "stable", "yes"}:
        return 1
    if value_norm in {"false", "0", "failure", "unstable", "no"}:
        return 0
    raise ValueError(f"Cannot parse binary grasp label from {value!r}")


def load_split_rows(dataset_root: Path, split: str, csv_path: Optional[Path] = None) -> List[Dict[str, str]]:
    split_path = csv_path or (dataset_root / f"{split}.csv")
    with split_path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resolve_rgb_frame(rgb_dir: Path, phase: str) -> Path:
    path = rgb_dir / f"{phase}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_rgb_transform(model_name: str):
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ModuleNotFoundError as exc:
        raise RuntimeError("This script needs timm installed to build the vision preprocessing.") from exc

    model = timm.create_model(model_name, pretrained=False)
    data_config = resolve_model_data_config(model)
    print(
        "RGB transform: "
        f"resize/crop strategy={data_config.get('crop_mode')} "
        f"input_size={data_config.get('input_size')} "
        f"crop_pct={data_config.get('crop_pct')} "
        f"interpolation={data_config.get('interpolation')}"
    )
    return create_transform(**data_config, is_training=False)


class FeelingSuccessVisionDataset(Dataset):
    """Dataset emitting RGB during/before tensors for the vision baseline."""

    def __init__(
        self,
        dataset_root: str | Path = REPO_ROOT / "dataset-limited",
        split: Literal["train", "eval", "test"] = "train",
        csv_path: Optional[str | Path] = None,
        pair_mode: PairMode = "during-before",
        model_name: str = DEFAULT_MODEL_NAME,
        include_metadata: bool = True,
    ) -> None:
        require_torch()
        if pair_mode != "during-before":
            raise ValueError(f"Unsupported pair mode: {pair_mode}")

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.csv_path = Path(csv_path).expanduser().resolve() if csv_path is not None else None
        self.pair_mode = pair_mode
        self.model_name = model_name
        self.include_metadata = include_metadata
        self.transform = build_rgb_transform(model_name)
        self.samples = self._load_samples()

    def _load_samples(self) -> List[VisionSample]:
        samples: List[VisionSample] = []
        for row in load_split_rows(self.dataset_root, self.split, self.csv_path):
            rgb_dir = self.dataset_root / row["rgb_path"]
            sample_dir = rgb_dir.parent
            samples.append(VisionSample(row, sample_dir, rgb_dir, bool_label(row["result"])))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        during = self.transform(load_rgb_image(resolve_rgb_frame(sample.rgb_dir, "during")))
        before = self.transform(load_rgb_image(resolve_rgb_frame(sample.rgb_dir, "before")))

        output: Dict[str, Any] = {
            "during_image": during,
            "before_image": before,
            "grasp_label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_dir.name,
            "pair_mode": self.pair_mode,
            "object_id": sample.row.get("object_id", ""),
            "condition_class": sample.row.get("condition_class", ""),
            "metadata_path": str(self.dataset_root / sample.row["metadata_path"]),
        }
        if self.include_metadata:
            output["metadata"] = self._read_metadata(sample)
        return output

    def _read_metadata(self, sample: VisionSample) -> Dict[str, Any]:
        metadata_path = self.dataset_root / sample.row["metadata_path"]
        with metadata_path.open() as f:
            return json.load(f)


def collate_keep_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = {"during_image", "before_image", "grasp_label"}
    output: Dict[str, Any] = {}
    for key in tensor_keys:
        output[key] = torch.stack([item[key] for item in batch])
    for key in batch[0].keys() - tensor_keys:
        output[key] = [item[key] for item in batch]
    return output


def build_dataset(
    dataset_root: str | Path,
    split: Literal["train", "eval", "test"],
    csv_path: Optional[str | Path] = None,
    pair_mode: PairMode = "during-before",
    model_name: str = DEFAULT_MODEL_NAME,
    max_samples: Optional[int] = None,
) -> Dataset:
    dataset = FeelingSuccessVisionDataset(
        dataset_root=dataset_root,
        split=split,
        csv_path=csv_path,
        pair_mode=pair_mode,
        model_name=model_name,
    )
    if max_samples is not None:
        return Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def build_dataloader(
    dataset_root: str | Path,
    split: Literal["train", "eval", "test"],
    csv_path: Optional[str | Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    pair_mode: PairMode = "during-before",
    model_name: str = DEFAULT_MODEL_NAME,
    shuffle: Optional[bool] = None,
    max_samples: Optional[int] = None,
    pin_memory: bool = False,
    seed: int = DEFAULT_SEED,
) -> DataLoader:
    require_torch()
    dataset = build_dataset(dataset_root, split, csv_path, pair_mode, model_name, max_samples)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_keep_metadata,
        worker_init_fn=seed_worker,
        generator=build_torch_generator(seed),
    )


if nn is not None:

    class VisionMlpHead(nn.Module):
        """Two-layer MLP over concatenated during/before ViT CLS embeddings."""

        def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, num_classes: int = 2) -> None:
            super().__init__()
            input_dim = embed_dim * 2
            hidden_dim = hidden_dim or embed_dim // 2
            self.probe = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, during_embedding: Any, before_embedding: Any) -> Any:
            return self.probe(torch.cat([during_embedding, before_embedding], dim=1))

    class VisionGraspDetector(nn.Module):
        """Frozen ViT RGB encoder plus trainable grasp MLP head."""

        def __init__(self, encoder: nn.Module, head: nn.Module, train_encoder: bool = False) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = head
            self.train_encoder = train_encoder
            if not train_encoder:
                self.encoder.eval()
                for param in self.encoder.parameters():
                    param.requires_grad_(False)

        def forward(self, during_image: Any, before_image: Any) -> Any:
            if self.train_encoder:
                during_embedding = self.encoder(during_image)
                before_embedding = self.encoder(before_image)
            else:
                with torch.no_grad():
                    during_embedding = self.encoder(during_image)
                    before_embedding = self.encoder(before_image)
            return self.head(during_embedding, before_embedding)


def load_head_checkpoint(module: nn.Module, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        print("No head checkpoint provided; using randomly initialized head.")
        return
    checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
    print(f"Loading head checkpoint from {checkpoint_path_resolved}")
    checkpoint = torch.load(checkpoint_path_resolved, map_location="cpu")
    state_dict = checkpoint.get("head_state_dict", checkpoint.get("state_dict", checkpoint))
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"Loaded head checkpoint; missing={len(missing)} unexpected={len(unexpected)}")


def build_vision_detector(
    model_name: str = DEFAULT_MODEL_NAME,
    checkpoint_head: Optional[str] = None,
    train_encoder: bool = False,
    pretrained: bool = True,
) -> nn.Module:
    require_torch()
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise RuntimeError("This script needs timm installed to load the vision encoder.") from exc

    print(f"Loading timm vision encoder {model_name} pretrained={pretrained}")
    encoder = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    head = VisionMlpHead(embed_dim=encoder.num_features)
    load_head_checkpoint(head, checkpoint_head)
    return VisionGraspDetector(encoder, head, train_encoder=train_encoder)


def train_head(
    detector: nn.Module,
    dataloader: DataLoader,
    device: Any,
    steps: int = 5,
    lr: float = 1e-3,
    eval_dataloader: Optional[DataLoader] = None,
    eval_every_steps: int = 50,
    writer: Optional[Any] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every_steps: int = 50,
    save_full_checkpoints: bool = False,
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, float]:
    detector.to(device)
    detector.train()
    if not detector.train_encoder:
        detector.encoder.eval()

    params = [param for param in detector.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)

    print(f"Training on device={device}; trainable parameters={sum(p.numel() for p in params)}")
    last_loss = 0.0
    last_acc = 0.0
    best_loss = float("inf")
    best_acc = -1.0
    best_eval_loss = float("inf")
    best_eval_acc = -1.0
    data_iter = iter(dataloader)
    for step in range(1, steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = detector(batch["during_image"], batch["before_image"])
        loss = F.cross_entropy(logits, batch["grasp_label"])
        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=1)
        acc = (pred == batch["grasp_label"]).float().mean()
        last_loss = loss.item()
        last_acc = acc.item()
        print(f"step={step} loss={last_loss:.4f} acc={last_acc:.3f}")
        if writer is not None:
            writer.add_scalar("train/loss", last_loss, step)
            writer.add_scalar("train/accuracy", last_acc, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
        metrics = {"loss": last_loss, "accuracy": last_acc}
        if eval_dataloader is not None and eval_every_steps > 0 and step % eval_every_steps == 0:
            eval_metrics = evaluate_head(detector, eval_dataloader, device)
            detector.train()
            if not detector.train_encoder:
                detector.encoder.eval()
            metrics.update(
                {
                    "eval_loss": eval_metrics["loss"],
                    "eval_accuracy": eval_metrics["accuracy"],
                }
            )
            print(
                f"eval step={step} loss={eval_metrics['loss']:.4f} "
                f"acc={eval_metrics['accuracy']:.3f}"
            )
            if writer is not None:
                writer.add_scalar("eval/loss", eval_metrics["loss"], step)
                writer.add_scalar("eval/accuracy", eval_metrics["accuracy"], step)
        if checkpoint_dir is not None:
            save_training_checkpoint(
                detector=detector,
                optimizer=optimizer,
                path=checkpoint_dir / "latest.pth",
                step=step,
                metrics=metrics,
                args=args,
                include_full_model=save_full_checkpoints,
            )
            if last_loss < best_loss:
                best_loss = last_loss
                save_training_checkpoint(
                    detector=detector,
                    optimizer=optimizer,
                    path=checkpoint_dir / "best_loss.pth",
                    step=step,
                    metrics=metrics,
                    args=args,
                    include_full_model=save_full_checkpoints,
                )
            if last_acc > best_acc:
                best_acc = last_acc
                save_training_checkpoint(
                    detector=detector,
                    optimizer=optimizer,
                    path=checkpoint_dir / "best_accuracy.pth",
                    step=step,
                    metrics=metrics,
                    args=args,
                    include_full_model=save_full_checkpoints,
                )
            if "eval_loss" in metrics and metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = metrics["eval_loss"]
                save_training_checkpoint(
                    detector=detector,
                    optimizer=optimizer,
                    path=checkpoint_dir / "best_eval_loss.pth",
                    step=step,
                    metrics=metrics,
                    args=args,
                    include_full_model=save_full_checkpoints,
                )
            if "eval_accuracy" in metrics and metrics["eval_accuracy"] > best_eval_acc:
                best_eval_acc = metrics["eval_accuracy"]
                save_training_checkpoint(
                    detector=detector,
                    optimizer=optimizer,
                    path=checkpoint_dir / "best_eval_accuracy.pth",
                    step=step,
                    metrics=metrics,
                    args=args,
                    include_full_model=save_full_checkpoints,
                )
            if checkpoint_every_steps > 0 and step % checkpoint_every_steps == 0:
                save_training_checkpoint(
                    detector=detector,
                    optimizer=optimizer,
                    path=checkpoint_dir / f"step_{step:06d}.pth",
                    step=step,
                    metrics=metrics,
                    args=args,
                    include_full_model=save_full_checkpoints,
                )
    if writer is not None:
        writer.flush()
    return {"loss": last_loss, "accuracy": last_acc}


def evaluate_head(detector: nn.Module, dataloader: DataLoader, device: Any) -> Dict[str, float]:
    detector.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            logits = detector(batch["during_image"], batch["before_image"])
            labels = batch["grasp_label"]
            loss = F.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += int(labels.numel())
    if total_count == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total_count, "accuracy": total_correct / total_count}


def save_training_checkpoint(
    detector: nn.Module,
    optimizer: Any,
    path: str | Path,
    step: int,
    metrics: Dict[str, float],
    args: Optional[argparse.Namespace],
    include_full_model: bool = False,
) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "metrics": metrics,
        "head_state_dict": detector.head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": checkpoint_safe_args(args),
    }
    if include_full_model:
        checkpoint["model_state_dict"] = detector.state_dict()
    torch.save(checkpoint, path)


def checkpoint_safe_args(args: Optional[argparse.Namespace]) -> Dict[str, Any]:
    if args is None:
        return {}

    def make_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): make_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [make_safe(item) for item in value]
        return str(value)

    return {key: make_safe(value) for key, value in vars(args).items()}


def save_head_checkpoint(detector: nn.Module, path: str | Path, metrics: Dict[str, float]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": detector.head.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"Saved head checkpoint to {path}")


def run_predictions(detector: nn.Module, dataloader: DataLoader, device: Any, output_path: Path) -> None:
    detector.to(device)
    detector.eval()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            logits = detector(batch["during_image"], batch["before_image"])
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for idx, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "object_id": batch["object_id"][idx],
                        "label": int(batch["grasp_label"][idx].cpu()),
                        "pred": int(preds[idx].cpu()),
                        "prob_failure": float(probs[idx, 0].cpu()),
                        "prob_success": float(probs[idx, 1].cpu()),
                    }
                )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "object_id", "label", "pred", "prob_failure", "prob_success"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote predictions to {output_path}")


def describe_batch(batch: Dict[str, Any]) -> None:
    labels = batch["grasp_label"]
    print(f"during image shape: {tuple(batch['during_image'].shape)}")
    print(f"before image shape: {tuple(batch['before_image'].shape)}")
    print(f"label shape: {tuple(labels.shape)} labels: {labels.tolist()}")
    print(f"sample ids: {batch['sample_id'][:3]}")
    print(f"pair modes: {batch['pair_mode'][:3]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset-limited")
    parser.add_argument("--train", dest="train_csv", type=Path, default=None, metavar="CSV", help="Train CSV. When provided, train the MLP head.")
    parser.add_argument("--eval", dest="eval_csv", type=Path, default=None, metavar="CSV", help="Eval CSV. With --train, run periodic eval; without --train, run inference.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pair-mode", choices=["during-before"], default="during-before")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--checkpoint-head", type=str, default=None)
    parser.add_argument("--train-encoder", action="store_true")
    parser.add_argument("--device", default="auto", help="Compute device: auto, cpu, mps, cuda, or cuda:N.")
    parser.add_argument("--smoke-batch", action="store_true", help="Load one dataloader batch and print shapes.")
    parser.add_argument("--smoke-model", action="store_true", help="Run one forward pass through the detector.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training rows; omit to use the full train CSV.")
    parser.add_argument("--train-steps", type=int, default=5, help="Number of optimizer steps for --train.")
    parser.add_argument("--eval-every-steps", type=int, default=50, help="Run eval every N training steps when an eval CSV is available; use 0 to disable.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-head-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for automatic latest/best/periodic training checkpoints; defaults to <tensorboard-run>/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=50,
        help="Write step_XXXXXX.pth every N training steps; use 0 to keep only latest/best checkpoints.",
    )
    parser.add_argument(
        "--save-full-checkpoints",
        action="store_true",
        help="Also store full detector weights in automatic checkpoints; useful only when training encoders, otherwise head checkpoints are enough.",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        default=REPO_ROOT / "runs" / "vision",
        help="Directory where training writes TensorBoard event files; point tensorboard --logdir here to watch loss/accuracy.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional subdirectory name for this run; set it to compare multiple runs in TensorBoard.",
    )
    parser.add_argument("--prediction-output", type=Path, default=REPO_ROOT / "vision-predictions.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_requested = args.train_csv is not None
    inference_requested = args.eval_csv is not None and not training_requested
    train_csv = args.train_csv
    eval_csv = args.eval_csv
    max_train_samples = args.max_train_samples
    train_steps = args.train_steps
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    needs_data = args.smoke_batch or args.smoke_model or training_requested or inference_requested
    if needs_data:
        dataloader = build_dataloader(
            dataset_root=args.dataset_root,
            split="eval" if inference_requested else "train",
            csv_path=eval_csv if inference_requested else train_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pair_mode=args.pair_mode,
            model_name=args.model_name,
            shuffle=training_requested,
            max_samples=max_train_samples if training_requested else None,
            pin_memory=device.type == "cuda",
            seed=args.seed,
        )
        batch = next(iter(dataloader))
        describe_batch(batch)

    if args.smoke_model or training_requested or inference_requested:
        detector = build_vision_detector(
            model_name=args.model_name,
            checkpoint_head=args.checkpoint_head,
            train_encoder=args.train_encoder,
            pretrained=not args.no_pretrained,
        )

    if args.smoke_model:
        detector.to(device)
        detector.eval()
        with torch.no_grad():
            logits = detector(batch["during_image"].to(device), batch["before_image"].to(device))
        print(f"logits shape: {tuple(logits.shape)}")

    if training_requested:
        eval_dataloader = None
        if args.eval_every_steps > 0 and eval_csv is not None and eval_csv.exists():
            eval_dataloader = build_dataloader(
                dataset_root=args.dataset_root,
                split="eval",
                csv_path=eval_csv,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pair_mode=args.pair_mode,
                model_name=args.model_name,
                shuffle=False,
                max_samples=None,
                pin_memory=device.type == "cuda",
                seed=args.seed,
            )
        writer, run_dir = build_summary_writer(args.tensorboard_logdir, args.run_name, args)
        checkpoint_dir = args.checkpoint_dir or (run_dir / "checkpoints")
        print(f"Training checkpoints will be written to {checkpoint_dir}")
        metrics = train_head(
            detector,
            dataloader,
            device,
            steps=train_steps,
            lr=args.lr,
            eval_dataloader=eval_dataloader,
            eval_every_steps=args.eval_every_steps,
            writer=writer,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_steps=args.checkpoint_every_steps,
            save_full_checkpoints=args.save_full_checkpoints,
            args=args,
        )
        writer.close()
        if args.save_head_checkpoint is not None:
            save_head_checkpoint(detector, args.save_head_checkpoint, metrics)

    if inference_requested:
        run_predictions(detector, dataloader, device, args.prediction_output)


if __name__ == "__main__":
    main()
