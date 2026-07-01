# Tactile Detector CLI

This file documents the quick CLI workflows for `tactile-detector.py`.

## Environment

Use the `monty-tactile` conda environment. On this macOS setup, the OpenMP workaround is needed:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python tactile-detector.py ...
```

## Devices

Choose compute at runtime:

```bash
--device auto     # CUDA if available, else MPS, else CPU
--device cpu
--device mps
--device cuda
--device cuda:0
```

This machine currently has MPS built but not available in the active PyTorch runtime, so `--device auto` falls back to CPU.

## Explain Sparsh Tactile Handling

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python tactile-detector.py \
  --explain-sparsh
```

## Dataloader Smoke Test

Loads one batch from `dataset-limited/train.csv` and prints tensor shapes.

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python tactile-detector.py \
  --smoke-batch \
  --device cpu \
  --batch-size 2 \
  --sensor gelsightA \
  --pair-mode during-before
```

Expected image shape is `(batch, 6, 320, 240)`.

## Model Forward Smoke Test

Runs one batch through the Sparsh ViT encoder plus the MLP head.

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python tactile-detector.py \
  --smoke-model \
  --device cpu \
  --model-size base \
  --batch-size 1 \
  --sensor gelsightA \
  --pair-mode during-before
```

Expected logits shape is `(batch, 2)`.

To use the downloaded pretrained Sparsh DINO base encoder:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python tactile-detector.py \
  --smoke-model \
  --device cpu \
  --model-size base \
  --batch-size 1 \
  --sensor gelsightA \
  --pair-mode during-before \
  --checkpoint-encoder checkpoints/sparsh-dino-base/dino_vitbase.safetensors
```

## Train MLP Head

Trains only the custom MLP head by default. The Sparsh encoder is frozen unless `--train-encoder` is passed.

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python tactile-detector.py \
  --train feeling-of-success/train.csv \
  --eval feeling-of-success/eval.csv \
  --dataset-root feeling-of-success \
  --device cpu \
  --model-size base \
  --checkpoint-encoder checkpoints/sparsh-dino-base/dino_vitbase.safetensors \
  --batch-size 8 \
  --train-steps 5000 \
  --eval-every-steps 250 \
  --sensor random \
  --pair-mode during-before
```

Training writes TensorBoard scalars by default:

```text
runs/tactile/<run-name>/train/loss
runs/tactile/<run-name>/train/accuracy
runs/tactile/<run-name>/train/lr
```

Use `--run-name` to give a run a stable label, and `--tensorboard-logdir` to choose where event files are written:

```bash
  --run-name tactile-baseline \
  --tensorboard-logdir runs/tactile
```

Launch the web UI:

```bash
/Users/adwait/miniconda3/envs/monty-tactile/bin/tensorboard \
  --logdir /Users/adwait/workspace/touch-perception/runs \
  --host 0.0.0.0
```

## Automatic Training Checkpoints

During `--train`, checkpoints are written next to the TensorBoard run by default:

```text
runs/tactile/<run-name>/checkpoints/latest.pth
runs/tactile/<run-name>/checkpoints/best_loss.pth
runs/tactile/<run-name>/checkpoints/best_accuracy.pth
runs/tactile/<run-name>/checkpoints/best_eval_loss.pth
runs/tactile/<run-name>/checkpoints/best_eval_accuracy.pth
runs/tactile/<run-name>/checkpoints/step_000050.pth
```

Use `--checkpoint-every-steps N` to control periodic `step_*.pth` snapshots, or `--checkpoint-every-steps 0` to keep only `latest` and best checkpoints. Use `--checkpoint-dir /path/to/dir` to put them somewhere else. These checkpoints store the MLP head, optimizer state, metrics, and safe CLI metadata; pass `--save-full-checkpoints` only when you also need full detector weights.

## Save And Reload Head Checkpoint

Save a trained head:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python tactile-detector.py \
  --train dataset-limited/train.csv \
  --eval dataset-limited/eval.csv \
  --dataset-root dataset-limited \
  --device cpu \
  --model-size tiny \
  --batch-size 1 \
  --max-train-samples 5 \
  --train-steps 5 \
  --sensor gelsightA \
  --pair-mode during-before \
  --save-head-checkpoint /private/tmp/tactile-smoke-head.pth
```

Run inference on an eval CSV:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python tactile-detector.py \
  --eval dataset-limited/eval.csv \
  --dataset-root dataset-limited \
  --device cpu \
  --model-size tiny \
  --batch-size 4 \
  --sensor gelsightA \
  --pair-mode during-before \
  --checkpoint-head /private/tmp/tactile-smoke-head.pth \
  --prediction-output /private/tmp/tactile-eval-predictions.csv
```

## Checkpoints

The repo-local pretrained Sparsh DINO base checkpoint is:

```text
checkpoints/sparsh-dino-base/dino_vitbase.safetensors
```

By default, no Sparsh checkpoint is loaded unless `--checkpoint-encoder` or `--checkpoint-head` is passed. The script prints this explicitly:

```text
No encoder checkpoint provided; using randomly initialized weights.
No head checkpoint provided; using randomly initialized weights.
```

Pass checkpoints explicitly:

```bash
--checkpoint-encoder /path/to/encoder.ckpt
--checkpoint-encoder /path/to/encoder.safetensors
--checkpoint-head /path/to/head.pth
```

The script prints the resolved absolute checkpoint path before loading.

## Sensor And Frame Choices

Sensor options:

```bash
--sensor random
--sensor gelsightA
--sensor gelsightB
--sensor both
```

Frame pair options:

```bash
--pair-mode during-before  # default; safe for outcome prediction
```

The default grasp input is two RGB GelSight images concatenated into 6 channels. The script only supports `during-before` tactile evidence, so post-outcome `after` images are never used for training or inference.
