#!/usr/bin/env python3
"""Wire the limited Feeling-of-Success tactile split into Sparsh T4 grasp input.

This script intentionally stops at data/model setup. It gives us:

* a Dataset that reads dataset-limited/{train,eval,test}.csv
* Sparsh-compatible GelSight preprocessing and 6-channel grasp inputs
* a frozen Sparsh ViT encoder plus a two-layer grasp MLP probe head
* CLI smoke checks for batch/model shapes and tiny head-training runs
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data import Subset
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
SPARSH_ROOT = REPO_ROOT / "sparsh"
SPARSH_GRASP_RESIZE_HW = (320, 240)
DEFAULT_SEED = 42

SensorName = Literal["gelsightA", "gelsightB"]
SensorPolicy = Literal["random", "gelsightA", "gelsightB", "both"]
PairMode = Literal["during-before"]
DeviceName = Literal["auto", "cpu", "mps", "cuda"]


@dataclass(frozen=True)
class TactileSample:
    row: Dict[str, str]
    sample_dir: Path
    tactile_dir: Path
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
        raise RuntimeError(
            "This script needs torch and torchvision. Install the project env from "
            "requirements.lock, or run only `--explain-sparsh` without smoke checks."
        )


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
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        return torch.device("mps")
    if device == "cpu":
        return torch.device("cpu")
    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested --device {device}, but CUDA is not available.")
        return torch.device(device)
    raise ValueError(f"Unsupported device {device!r}. Use auto, cpu, mps, cuda, or cuda:N.")


def move_batch_to_device(batch: Dict[str, Any], device: Any) -> Dict[str, Any]:
    non_blocking = getattr(device, "type", None) == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_summary_writer(logdir: Path, run_name: Optional[str], args: argparse.Namespace):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging requires the tensorboard package. Install it with `pip install tensorboard`."
        ) from exc

    run_name = run_name or datetime.now().strftime("tactile-%Y%m%d-%H%M%S")
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


def load_rgb_portrait(path: Path) -> Image.Image:
    """Match Sparsh's GelSight loader: RGB, portrait orientation, 4:3 H/W crop."""
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w, _ = img.shape

    if h < w:
        img = np.rot90(img, k=3)
        h, w, _ = img.shape

    target_ratio = 4.0 / 3.0
    if not np.isclose(h / w, target_ratio):
        cropped_h = min(h, int(w * target_ratio))
        top = max((h - cropped_h) // 2, 0)
        img = img[top : top + cropped_h, :]

    return Image.fromarray(img).convert("RGB")


def resolve_tactile_frame(tactile_dir: Path, sensor: SensorName, phase: str) -> Path:
    path = tactile_dir / f"{sensor}_{phase}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


class FeelingSuccessTactileDataset(Dataset):
    """Dataset emitting the same keys/shapes as Sparsh T4 grasp stability.

    Returned sample:
        image: FloatTensor [6, 320, 240] for concat_ch_img mode
        grasp_label: LongTensor scalar, 1 for success/stable grasp, 0 otherwise
    """

    def __init__(
        self,
        dataset_root: str | Path = REPO_ROOT / "dataset-limited",
        split: Literal["train", "eval", "test"] = "train",
        csv_path: Optional[str | Path] = None,
        image_size_hw: Tuple[int, int] = SPARSH_GRASP_RESIZE_HW,
        sensor_policy: SensorPolicy = "random",
        pair_mode: PairMode = "during-before",
        include_metadata: bool = True,
    ) -> None:
        require_torch()
        if sensor_policy not in {"random", "gelsightA", "gelsightB", "both"}:
            raise ValueError(f"Unsupported sensor policy: {sensor_policy}")
        if pair_mode not in {"during-before"}:
            raise ValueError(f"Unsupported pair mode: {pair_mode}")

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.csv_path = Path(csv_path).expanduser().resolve() if csv_path is not None else None
        self.sensor_policy = sensor_policy
        self.pair_mode = pair_mode
        self.include_metadata = include_metadata
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size_hw, antialias=True),
                transforms.ToTensor(),
            ]
        )

        self.samples = self._load_samples()

    def _load_samples(self) -> List[TactileSample]:
        samples: List[TactileSample] = []
        for row in load_split_rows(self.dataset_root, self.split, self.csv_path):
            tactile_dir = self.dataset_root / row["tactile_path"]
            sample_dir = tactile_dir.parent
            label = bool_label(row["result"])
            samples.append(TactileSample(row, sample_dir, tactile_dir, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image, sensor, pair = self._load_tactile_tensor(sample.tactile_dir)

        output: Dict[str, Any] = {
            "image": image,
            "grasp_label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_dir.name,
            "sensor": sensor,
            "pair_mode": pair,
            "object_id": sample.row.get("object_id", ""),
            "condition_class": sample.row.get("condition_class", ""),
            "metadata_path": str(self.dataset_root / sample.row["metadata_path"]),
        }

        if self.include_metadata:
            output["metadata"] = self._read_metadata(sample)
        return output

    def _choose_sensor(self) -> SensorName:
        if self.sensor_policy == "random":
            return "gelsightA" if torch.rand(()) >= 0.5 else "gelsightB"
        if self.sensor_policy == "both":
            raise ValueError("Use expand_two_sensors=True in build_dataloader for both sensors.")
        return self.sensor_policy

    def _choose_pair(self) -> Tuple[str, str, str]:
        return "during", "before", "during-before"

    def _load_tactile_tensor(self, tactile_dir: Path) -> Tuple[Any, str, str]:
        sensor = self._choose_sensor()
        phase1, phase2, pair = self._choose_pair()
        image1 = self.transform(load_rgb_portrait(resolve_tactile_frame(tactile_dir, sensor, phase1)))
        image2 = self.transform(load_rgb_portrait(resolve_tactile_frame(tactile_dir, sensor, phase2)))
        return torch.cat([image1, image2], dim=0), sensor, pair

    def _read_metadata(self, sample: TactileSample) -> Dict[str, Any]:
        metadata_path = self.dataset_root / sample.row["metadata_path"]
        with metadata_path.open() as f:
            return json.load(f)


class TwoSensorTactileDataset(Dataset):
    """Expand each manifest row into one sample for GelSight A and one for B."""

    def __init__(self, base: FeelingSuccessTactileDataset) -> None:
        require_torch()
        if base.sensor_policy != "both":
            raise ValueError("TwoSensorTactileDataset expects a base dataset with sensor_policy='both'.")
        self.base = base

    def __len__(self) -> int:
        return len(self.base) * 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base_idx = idx // 2
        sensor: SensorName = "gelsightA" if idx % 2 == 0 else "gelsightB"
        old_policy = self.base.sensor_policy
        self.base.sensor_policy = sensor
        try:
            return self.base[base_idx]
        finally:
            self.base.sensor_policy = old_policy


def collate_keep_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = {"image", "grasp_label"}
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
    sensor_policy: SensorPolicy = "random",
    pair_mode: PairMode = "during-before",
    max_samples: Optional[int] = None,
) -> Dataset:
    dataset = FeelingSuccessTactileDataset(
        dataset_root=dataset_root,
        split=split,
        csv_path=csv_path,
        sensor_policy=sensor_policy,
        pair_mode=pair_mode,
    )
    if sensor_policy == "both":
        expanded: Dataset = TwoSensorTactileDataset(dataset)
        if max_samples is not None:
            return Subset(expanded, range(min(max_samples, len(expanded))))
        return expanded
    if max_samples is not None:
        return Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def build_dataloader(
    dataset_root: str | Path,
    split: Literal["train", "eval", "test"],
    csv_path: Optional[str | Path] = None,
    batch_size: int = 16,
    num_workers: int = 0,
    sensor_policy: SensorPolicy = "random",
    pair_mode: PairMode = "during-before",
    shuffle: Optional[bool] = None,
    max_samples: Optional[int] = None,
    pin_memory: bool = False,
    seed: int = DEFAULT_SEED,
) -> DataLoader:
    require_torch()
    dataset = build_dataset(dataset_root, split, csv_path, sensor_policy, pair_mode, max_samples)
    if shuffle is None:
        shuffle = split == "train"
    loader_kwargs: Dict[str, Any] = {}
    if pin_memory and num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_keep_metadata,
        worker_init_fn=seed_worker,
        generator=build_torch_generator(seed),
        **loader_kwargs,
    )


if nn is not None:

    class GraspMlpHead(nn.Module):
        """Two-layer MLP grasp head over pooled Sparsh patch tokens."""

        def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, num_classes: int = 2) -> None:
            super().__init__()
            hidden_dim = hidden_dim or embed_dim // 4
            self.probe = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, tokens: Any) -> Any:
            pooled = tokens.mean(dim=1)
            return self.probe(pooled)

    class TactileGraspDetector(nn.Module):
        """Sparsh T4-style grasp detector: ViT encoder + MLP head."""

        def __init__(
            self,
            encoder: nn.Module,
            head: nn.Module,
            train_encoder: bool = False,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = head
            self.train_encoder = train_encoder
            if not train_encoder:
                self.encoder.eval()
                for param in self.encoder.parameters():
                    param.requires_grad_(False)

        def forward(self, image: Any) -> Any:
            if self.train_encoder:
                tokens = self.encoder(image)
            else:
                with torch.no_grad():
                    tokens = self.encoder(image)
            return self.head(tokens)


def load_checkpoint_file(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "head_state_dict" in checkpoint:
        return checkpoint["head_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def normalize_checkpoint_keys(state_dict: Dict[str, Any], label: str) -> Dict[str, Any]:
    preferred_prefixes = []
    if label == "encoder":
        preferred_prefixes.extend(
            [
                "teacher_encoder.backbone.",
                "target_encoder.backbone.",
                "encoder.backbone.",
                "backbone.",
                "model_encoder.",
                "encoder.",
            ]
        )
    else:
        preferred_prefixes.extend(["head.", "model_task.", "task.", "module."])

    for prefix in preferred_prefixes:
        matching = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if matching:
            print(f"Using {len(matching)} keys with prefix {prefix!r} for {label}.")
            return matching

    stripped = {}
    generic_prefixes = ("module.",)
    for key, value in state_dict.items():
        for prefix in generic_prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        stripped[key] = value
    return stripped


def load_checkpoint(module: nn.Module, checkpoint_path: Optional[str], label: str) -> None:
    if not checkpoint_path:
        print(f"No {label} checkpoint provided; using randomly initialized weights.")
        return
    checkpoint_path_resolved = Path(checkpoint_path).expanduser().resolve()
    print(f"Loading {label} checkpoint from {checkpoint_path_resolved}")
    state_dict = normalize_checkpoint_keys(
        load_checkpoint_file(checkpoint_path_resolved),
        label,
    )
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded {label} checkpoint; missing={len(missing)} unexpected={len(unexpected)}"
    )


def build_sparsh_t4_detector(
    model_size: Literal["tiny", "small", "base", "large"] = "base",
    image_size_hw: Tuple[int, int] = SPARSH_GRASP_RESIZE_HW,
    checkpoint_encoder: Optional[str] = None,
    checkpoint_head: Optional[str] = None,
    train_encoder: bool = False,
) -> nn.Module:
    """Instantiate the local Sparsh T4 model shape for later training/eval."""
    require_torch()
    sys.path.insert(0, str(SPARSH_ROOT))

    try:
        import tactile_ssl.model as sparsh_model
        from tactile_ssl.model import VIT_EMBED_DIMS
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import Sparsh model dependencies. Install the pinned "
            "environment from requirements.lock before using --smoke-model or training."
        ) from exc

    encoder_factory = getattr(sparsh_model, f"vit_{model_size}")
    encoder = encoder_factory(
        img_size=image_size_hw,
        in_chans=6,
        pos_embed_fn="sinusoidal",
        num_register_tokens=1,
    )
    head = GraspMlpHead(embed_dim=VIT_EMBED_DIMS[f"vit_{model_size}"])

    load_checkpoint(encoder, checkpoint_encoder, "encoder")
    load_checkpoint(head, checkpoint_head, "head")
    return TactileGraspDetector(encoder, head, train_encoder=train_encoder)


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
        logits = detector(batch["image"])
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
            should_save_latest = (
                step == 1
                or step == steps
                or (checkpoint_every_steps > 0 and step % checkpoint_every_steps == 0)
            )
            if should_save_latest:
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
            logits = detector(batch["image"])
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
        "head_state_dict": state_dict_to_cpu(detector.head.state_dict()),
        "optimizer_state_dict": optimizer_state_dict_to_cpu(optimizer.state_dict()),
        "args": checkpoint_safe_args(args),
    }
    if include_full_model:
        checkpoint["model_state_dict"] = state_dict_to_cpu(detector.state_dict())
    torch.save(checkpoint, path)


def state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) and value.is_cuda else value
        for key, value in state_dict.items()
    }


def optimizer_state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cpu_state = dict(state_dict)
    cpu_state["state"] = {
        param_id: state_dict_to_cpu(param_state)
        for param_id, param_state in state_dict.get("state", {}).items()
    }
    return cpu_state


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
            logits = detector(batch["image"])
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for idx, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "object_id": batch["object_id"][idx],
                        "sensor": batch["sensor"][idx],
                        "label": int(batch["grasp_label"][idx].cpu()),
                        "pred": int(preds[idx].cpu()),
                        "prob_failure": float(probs[idx, 0].cpu()),
                        "prob_success": float(probs[idx, 1].cpu()),
                    }
                )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "object_id", "sensor", "label", "pred", "prob_failure", "prob_success"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote predictions to {output_path}")


def print_sparsh_tactile_notes() -> None:
    print(
        "\n".join(
            [
                "Sparsh T4 grasp stability tactile-data notes:",
                "",
                "1. Image size:",
                "   The local T4 grasp config uses data.dataset.config.transforms.resize: [320, 240].",
                "   In torchvision Resize tuple semantics, that means output tensors are H=320, W=240.",
                "   This script's input is [6, 320, 240]: during and before RGB GelSight frames concatenated on channels.",
                "",
                "2. Smaller/larger input images:",
                "   Sparsh loads each GelSight image as RGB, rotates landscape frames into portrait orientation,",
                "   center-crops to the default 4:3 height/width aspect when needed, then resizes to [320, 240].",
                "   The original PNG size does not need to match the model size.",
                "",
                "3. One or both gripping arms:",
                "   The T4 dataset code randomly picks either gelsightA or gelsightB for each sample.",
                "   It does not require both arms simultaneously for the default detector. This script mirrors",
                "   that with --sensor random, and also supports --sensor gelsightA, --sensor gelsightB,",
                "   or --sensor both to evaluate both sensors as separate examples.",
            ]
        )
    )


def describe_batch(batch: Dict[str, Any]) -> None:
    labels = batch["grasp_label"]
    print(f"image shape: {tuple(batch['image'].shape)}")
    print(f"label shape: {tuple(labels.shape)} labels: {labels.tolist()}")
    print(f"sample ids: {batch['sample_id'][:3]}")
    print(f"sensors: {batch['sensor'][:3]}")
    print(f"pair modes: {batch['pair_mode'][:3]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset-limited")
    parser.add_argument("--train", dest="train_csv", type=Path, default=None, metavar="CSV", help="Train CSV. When provided, train the MLP head.")
    parser.add_argument("--eval", dest="eval_csv", type=Path, default=None, metavar="CSV", help="Eval CSV. With --train, run periodic eval; without --train, run inference.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sensor", choices=["random", "gelsightA", "gelsightB", "both"], default="random")
    parser.add_argument("--pair-mode", choices=["during-before"], default="during-before")
    parser.add_argument("--model-size", choices=["tiny", "small", "base", "large"], default="base")
    parser.add_argument("--checkpoint-encoder", type=str, default=None)
    parser.add_argument("--checkpoint-head", type=str, default=None)
    parser.add_argument("--train-encoder", action="store_true")
    parser.add_argument("--device", default="auto", help="Compute device: auto, cpu, mps, cuda, or cuda:N.")
    parser.add_argument("--smoke-batch", action="store_true", help="Load one dataloader batch and print shapes.")
    parser.add_argument("--smoke-model", action="store_true", help="Run one forward pass through the Sparsh detector.")
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
        default=REPO_ROOT / "runs" / "tactile",
        help="Directory where training writes TensorBoard event files; point tensorboard --logdir here to watch loss/accuracy.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional subdirectory name for this run; set it to compare multiple runs in TensorBoard.",
    )
    parser.add_argument("--explain-sparsh", action="store_true", help="Print answers about Sparsh tactile handling.")
    parser.add_argument("--prediction-output", type=Path, default=REPO_ROOT / "tactile-predictions.csv")
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

    if args.explain_sparsh:
        print_sparsh_tactile_notes()
        if not args.smoke_batch and not args.smoke_model and not training_requested and not inference_requested:
            return

    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    if args.smoke_batch or args.smoke_model or training_requested or inference_requested:
        dataloader = build_dataloader(
            dataset_root=args.dataset_root,
            split="eval" if inference_requested else "train",
            csv_path=eval_csv if inference_requested else train_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sensor_policy=args.sensor,
            pair_mode=args.pair_mode,
            shuffle=training_requested,
            max_samples=max_train_samples if training_requested else None,
            pin_memory=device.type == "cuda",
            seed=args.seed,
        )
        batch = next(iter(dataloader))
        describe_batch(batch)

    if args.smoke_model or training_requested or inference_requested:
        detector = build_sparsh_t4_detector(
            model_size=args.model_size,
            checkpoint_encoder=args.checkpoint_encoder,
            checkpoint_head=args.checkpoint_head,
            train_encoder=args.train_encoder,
        )

    if args.smoke_model:
        detector.to(device)
        detector.eval()
        with torch.no_grad():
            logits = detector(batch["image"].to(device))
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
                sensor_policy=args.sensor,
                pair_mode=args.pair_mode,
                shuffle=False,
                max_samples=None,
                pin_memory=device.type == "cuda",
                seed=args.seed,
            )
        writer, run_dir = build_summary_writer(args.tensorboard_logdir, args.run_name, args)
        checkpoint_dir = args.checkpoint_dir or (run_dir / "checkpoints")
        print(f"Training checkpoints will be written to {checkpoint_dir}")
        metrics = train_head(
            detector=detector,
            dataloader=dataloader,
            device=device,
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
