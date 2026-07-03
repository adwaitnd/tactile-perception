#!/usr/bin/env python3
# This code is owned by Adwait Dongare and may not be used for any purposes unless explicitly allowed.
"""Joint vision+tactile grasp outcome detector.

This combines the frozen tactile Sparsh encoder and frozen timm ViT vision
encoder. The tactile input is the Sparsh-style during/before 6-channel GelSight
pair. The vision input is RGB during and before frames encoded independently by
the same ViT. Encoder outputs are concatenated and fed to a shared MLP head.

Post-outcome after images are intentionally unsupported.
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
SPARSH_ROOT = REPO_ROOT / "sparsh"
DEFAULT_VISION_MODEL = "vit_base_patch16_224.augreg_in21k"
SPARSH_GRASP_RESIZE_HW = (320, 240)
DEFAULT_SEED = 42

SensorName = Literal["gelsightA", "gelsightB"]
SensorPolicy = Literal["random", "gelsightA", "gelsightB"]
PairMode = Literal["during-before"]


@dataclass(frozen=True)
class JointSample:
    row: Dict[str, str]
    sample_dir: Path
    rgb_dir: Path
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

    run_name = run_name or datetime.now().strftime("joint-%Y%m%d-%H%M%S")
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


def crop_rgb_to_motion_box(image: Image.Image, metadata: Dict[str, Any], sample_id: str) -> Image.Image:
    try:
        box = metadata["rgb_motion_diff"]["box_xyxy"]
    except KeyError as exc:
        raise KeyError(
            f"Sample {sample_id} metadata is missing rgb_motion_diff.box_xyxy; "
            "run scripts/add_rgb_motion_diff_metadata.py before training vision models."
        ) from exc

    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"Sample {sample_id} has invalid rgb_motion_diff.box_xyxy={box!r}")

    width, height = image.size
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return image.crop((x1, y1, x2, y2))


def load_tactile_rgb_portrait(path: Path) -> Image.Image:
    """Match Sparsh GelSight preprocessing before resizing."""
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


def resolve_frame(folder: Path, name: str) -> Path:
    path = folder / f"{name}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_tactile_frame(tactile_dir: Path, sensor: SensorName, phase: str) -> Path:
    path = tactile_dir / f"{sensor}_{phase}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_vision_transform(model_name: str):
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ModuleNotFoundError as exc:
        raise RuntimeError("This script needs timm installed for vision preprocessing.") from exc

    model = timm.create_model(model_name, pretrained=False)
    data_config = resolve_model_data_config(model)
    data_config["crop_pct"] = 1.0
    print(
        "RGB transform after rgb_motion_diff crop: "
        f"resize/crop strategy={data_config.get('crop_mode')} "
        f"input_size={data_config.get('input_size')} "
        f"crop_pct={data_config.get('crop_pct')} "
        f"interpolation={data_config.get('interpolation')}"
    )
    return create_transform(**data_config, is_training=False)


class FeelingSuccessJointDataset(Dataset):
    """Dataset emitting paired vision and tactile during/before inputs."""

    def __init__(
        self,
        dataset_root: str | Path = REPO_ROOT / "dataset-limited",
        split: Literal["train", "eval", "test"] = "train",
        csv_path: Optional[str | Path] = None,
        sensor_policy: SensorPolicy = "random",
        pair_mode: PairMode = "during-before",
        vision_model_name: str = DEFAULT_VISION_MODEL,
        include_metadata: bool = True,
    ) -> None:
        require_torch()
        if sensor_policy not in {"random", "gelsightA", "gelsightB"}:
            raise ValueError(f"Unsupported sensor policy: {sensor_policy}")
        if pair_mode != "during-before":
            raise ValueError(f"Unsupported pair mode: {pair_mode}")

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.csv_path = Path(csv_path).expanduser().resolve() if csv_path is not None else None
        self.sensor_policy = sensor_policy
        self.pair_mode = pair_mode
        self.include_metadata = include_metadata
        self.vision_transform = build_vision_transform(vision_model_name)
        self.tactile_transform = transforms.Compose(
            [
                transforms.Resize(SPARSH_GRASP_RESIZE_HW, antialias=True),
                transforms.ToTensor(),
            ]
        )
        self.samples = self._load_samples()

    def _load_samples(self) -> List[JointSample]:
        samples: List[JointSample] = []
        for row in load_split_rows(self.dataset_root, self.split, self.csv_path):
            sample_dir = (self.dataset_root / row["rgb_path"]).parent
            samples.append(
                JointSample(
                    row=row,
                    sample_dir=sample_dir,
                    rgb_dir=self.dataset_root / row["rgb_path"],
                    tactile_dir=self.dataset_root / row["tactile_path"],
                    label=bool_label(row["result"]),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sensor = self._choose_sensor()
        metadata = self._read_metadata(sample)

        vision_during = self.vision_transform(
            crop_rgb_to_motion_box(
                load_rgb_image(resolve_frame(sample.rgb_dir, "during")),
                metadata,
                sample.sample_dir.name,
            )
        )
        vision_before = self.vision_transform(
            crop_rgb_to_motion_box(
                load_rgb_image(resolve_frame(sample.rgb_dir, "before")),
                metadata,
                sample.sample_dir.name,
            )
        )

        tactile_during = self.tactile_transform(
            load_tactile_rgb_portrait(resolve_tactile_frame(sample.tactile_dir, sensor, "during"))
        )
        tactile_before = self.tactile_transform(
            load_tactile_rgb_portrait(resolve_tactile_frame(sample.tactile_dir, sensor, "before"))
        )
        tactile_image = torch.cat([tactile_during, tactile_before], dim=0)

        output: Dict[str, Any] = {
            "vision_during": vision_during,
            "vision_before": vision_before,
            "tactile_image": tactile_image,
            "grasp_label": torch.tensor(sample.label, dtype=torch.long),
            "sample_id": sample.sample_dir.name,
            "sensor": sensor,
            "pair_mode": self.pair_mode,
            "object_id": sample.row.get("object_id", ""),
            "condition_class": sample.row.get("condition_class", ""),
            "metadata_path": str(self.dataset_root / sample.row["metadata_path"]),
        }
        if self.include_metadata:
            output["metadata"] = metadata
        return output

    def _choose_sensor(self) -> SensorName:
        if self.sensor_policy == "random":
            return "gelsightA" if torch.rand(()) >= 0.5 else "gelsightB"
        return self.sensor_policy

    def _read_metadata(self, sample: JointSample) -> Dict[str, Any]:
        metadata_path = self.dataset_root / sample.row["metadata_path"]
        with metadata_path.open() as f:
            return json.load(f)


def collate_keep_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = {"vision_during", "vision_before", "tactile_image", "grasp_label"}
    output: Dict[str, Any] = {}
    for key in tensor_keys:
        output[key] = torch.stack([item[key] for item in batch])
    for key in batch[0].keys() - tensor_keys:
        output[key] = [item[key] for item in batch]
    return output


def build_dataloader(
    dataset_root: str | Path,
    split: Literal["train", "eval", "test"],
    csv_path: Optional[str | Path],
    batch_size: int,
    num_workers: int,
    sensor_policy: SensorPolicy,
    vision_model_name: str,
    shuffle: bool,
    max_samples: Optional[int],
    pin_memory: bool,
    seed: int = DEFAULT_SEED,
) -> DataLoader:
    dataset: Dataset = FeelingSuccessJointDataset(
        dataset_root=dataset_root,
        split=split,
        csv_path=csv_path,
        sensor_policy=sensor_policy,
        vision_model_name=vision_model_name,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
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

    class JointMlpHead(nn.Module):
        """Two-layer MLP over concatenated tactile and vision embeddings."""

        def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, num_classes: int = 2) -> None:
            super().__init__()
            hidden_dim = hidden_dim or input_dim // 2
            self.probe = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, fused_embedding: Any) -> Any:
            return self.probe(fused_embedding)

    class JointGraspDetector(nn.Module):
        def __init__(
            self,
            tactile_encoder: nn.Module,
            vision_encoder: nn.Module,
            head: nn.Module,
            train_encoders: bool = False,
        ) -> None:
            super().__init__()
            self.tactile_encoder = tactile_encoder
            self.vision_encoder = vision_encoder
            self.head = head
            self.train_encoders = train_encoders
            if not train_encoders:
                self.tactile_encoder.eval()
                self.vision_encoder.eval()
                for param in self.tactile_encoder.parameters():
                    param.requires_grad_(False)
                for param in self.vision_encoder.parameters():
                    param.requires_grad_(False)

        def _encode(self, batch: Dict[str, Any]) -> Any:
            tactile_tokens = self.tactile_encoder(batch["tactile_image"])
            tactile_embedding = tactile_tokens.mean(dim=1)
            vision_during = self.vision_encoder(batch["vision_during"])
            vision_before = self.vision_encoder(batch["vision_before"])
            return torch.cat([tactile_embedding, vision_during, vision_before], dim=1)

        def forward(self, batch: Dict[str, Any]) -> Any:
            if self.train_encoders:
                fused = self._encode(batch)
            else:
                with torch.no_grad():
                    fused = self._encode(batch)
            return self.head(fused)


def load_checkpoint_file(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def normalize_tactile_checkpoint_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    for prefix in (
        "teacher_encoder.backbone.",
        "target_encoder.backbone.",
        "encoder.backbone.",
        "backbone.",
        "model_encoder.",
        "encoder.",
    ):
        matching = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if matching:
            print(f"Using {len(matching)} tactile checkpoint keys with prefix {prefix!r}.")
            return matching
    return state_dict


def load_tactile_checkpoint(module: nn.Module, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        print("No tactile checkpoint provided; using randomly initialized tactile encoder.")
        return
    path = Path(checkpoint_path).expanduser().resolve()
    print(f"Loading tactile encoder checkpoint from {path}")
    state_dict = normalize_tactile_checkpoint_keys(load_checkpoint_file(path))
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"Loaded tactile checkpoint; missing={len(missing)} unexpected={len(unexpected)}")


def load_head_checkpoint(module: nn.Module, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        print("No joint head checkpoint provided; using randomly initialized head.")
        return
    path = Path(checkpoint_path).expanduser().resolve()
    print(f"Loading joint head checkpoint from {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("head_state_dict", checkpoint.get("state_dict", checkpoint))
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    print(f"Loaded joint head checkpoint; missing={len(missing)} unexpected={len(unexpected)}")


def build_joint_detector(
    tactile_model_size: Literal["tiny", "small", "base", "large"],
    tactile_checkpoint: Optional[str],
    vision_model_name: str,
    vision_pretrained: bool,
    head_checkpoint: Optional[str],
    train_encoders: bool,
) -> nn.Module:
    require_torch()
    sys.path.insert(0, str(SPARSH_ROOT))
    try:
        import tactile_ssl.model as sparsh_model
        from tactile_ssl.model import VIT_EMBED_DIMS
    except ModuleNotFoundError as exc:
        raise RuntimeError("Could not import local Sparsh model code.") from exc
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise RuntimeError("This script needs timm installed.") from exc

    tactile_factory = getattr(sparsh_model, f"vit_{tactile_model_size}")
    tactile_encoder = tactile_factory(
        img_size=SPARSH_GRASP_RESIZE_HW,
        in_chans=6,
        pos_embed_fn="sinusoidal",
        num_register_tokens=1,
    )
    load_tactile_checkpoint(tactile_encoder, tactile_checkpoint)

    print(f"Loading timm vision encoder {vision_model_name} pretrained={vision_pretrained}")
    vision_encoder = timm.create_model(vision_model_name, pretrained=vision_pretrained, num_classes=0)

    tactile_dim = VIT_EMBED_DIMS[f"vit_{tactile_model_size}"]
    vision_dim = vision_encoder.num_features
    head = JointMlpHead(input_dim=tactile_dim + 2 * vision_dim)
    load_head_checkpoint(head, head_checkpoint)
    return JointGraspDetector(tactile_encoder, vision_encoder, head, train_encoders=train_encoders)


def train_head(
    detector: nn.Module,
    dataloader: DataLoader,
    device: Any,
    steps: int,
    lr: float,
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
    if not detector.train_encoders:
        detector.tactile_encoder.eval()
        detector.vision_encoder.eval()

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
        logits = detector(batch)
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
            if not detector.train_encoders:
                detector.tactile_encoder.eval()
                detector.vision_encoder.eval()
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
            logits = detector(batch)
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
    torch.save({"state_dict": detector.head.state_dict(), "metrics": metrics}, path)
    print(f"Saved joint head checkpoint to {path}")


def run_predictions(detector: nn.Module, dataloader: DataLoader, device: Any, output_path: Path) -> None:
    detector.to(device)
    detector.eval()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            logits = detector(batch)
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


def describe_batch(batch: Dict[str, Any]) -> None:
    print(f"vision during shape: {tuple(batch['vision_during'].shape)}")
    print(f"vision before shape: {tuple(batch['vision_before'].shape)}")
    print(f"tactile image shape: {tuple(batch['tactile_image'].shape)}")
    print(f"labels: {batch['grasp_label'].tolist()}")
    print(f"sample ids: {batch['sample_id'][:3]}")
    print(f"sensors: {batch['sensor'][:3]}")
    print(f"pair modes: {batch['pair_mode'][:3]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "dataset-limited")
    parser.add_argument("--train", dest="train_csv", type=Path, default=None, metavar="CSV", help="Train CSV. When provided, train the joint MLP head.")
    parser.add_argument("--eval", dest="eval_csv", type=Path, default=None, metavar="CSV", help="Eval CSV. With --train, run periodic eval; without --train, run inference.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sensor", choices=["random", "gelsightA", "gelsightB"], default="random")
    parser.add_argument("--tactile-model-size", choices=["tiny", "small", "base", "large"], default="base")
    parser.add_argument("--tactile-checkpoint", type=str, default=None)
    parser.add_argument("--vision-model-name", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--vision-no-pretrained", action="store_true")
    parser.add_argument("--head-checkpoint", type=str, default=None)
    parser.add_argument("--train-encoders", action="store_true")
    parser.add_argument("--device", default="auto", help="Compute device: auto, cpu, mps, cuda, or cuda:N.")
    parser.add_argument("--smoke-batch", action="store_true")
    parser.add_argument("--smoke-model", action="store_true")
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
        default=REPO_ROOT / "runs" / "joint",
        help="Directory where training writes TensorBoard event files; point tensorboard --logdir here to watch loss/accuracy.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional subdirectory name for this run; set it to compare multiple runs in TensorBoard.",
    )
    parser.add_argument("--prediction-output", type=Path, default=REPO_ROOT / "joint-predictions.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_requested = args.train_csv is not None
    inference_requested = args.eval_csv is not None and not training_requested
    seed_everything(args.seed)
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    needs_data = args.smoke_batch or args.smoke_model or training_requested or inference_requested
    if needs_data:
        dataloader = build_dataloader(
            dataset_root=args.dataset_root,
            split="eval" if inference_requested else "train",
            csv_path=args.eval_csv if inference_requested else args.train_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sensor_policy=args.sensor,
            vision_model_name=args.vision_model_name,
            shuffle=training_requested,
            max_samples=args.max_train_samples if training_requested else None,
            pin_memory=device.type == "cuda",
            seed=args.seed,
        )
        batch = next(iter(dataloader))
        describe_batch(batch)

    if args.smoke_model or training_requested or inference_requested:
        detector = build_joint_detector(
            tactile_model_size=args.tactile_model_size,
            tactile_checkpoint=args.tactile_checkpoint,
            vision_model_name=args.vision_model_name,
            vision_pretrained=not args.vision_no_pretrained,
            head_checkpoint=args.head_checkpoint,
            train_encoders=args.train_encoders,
        )

    if args.smoke_model:
        detector.to(device)
        detector.eval()
        with torch.no_grad():
            logits = detector(move_batch_to_device(batch, device))
        print(f"logits shape: {tuple(logits.shape)}")

    if training_requested:
        eval_dataloader = None
        if args.eval_every_steps > 0 and args.eval_csv is not None and args.eval_csv.exists():
            eval_dataloader = build_dataloader(
                dataset_root=args.dataset_root,
                split="eval",
                csv_path=args.eval_csv,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                sensor_policy=args.sensor,
                vision_model_name=args.vision_model_name,
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
            args.train_steps,
            args.lr,
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
