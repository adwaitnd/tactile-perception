# Vision Detector CLI

This file documents the quick CLI workflows for `vision-detector.py`.

## Model

Default encoder:

```text
vit_base_patch16_224.augreg_in21k
```

This is loaded with `timm.create_model(..., pretrained=True, num_classes=0)`, so it returns image embeddings instead of classification logits. The script freezes the ViT encoder by default and trains only the MLP head.

Hugging Face model page:

```text
https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k
```

To pre-download the checkpoint into the Hugging Face cache:

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python -c \
"import timm; timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=0)"
```

## Inputs

The script uses only:

```text
rgb/during.png
rgb/before.png
```

`rgb/after.png` is intentionally unsupported because it leaks post-outcome information.

The two frames are encoded independently by the same shared ViT, then their embeddings are concatenated:

```text
[during_embedding, before_embedding] -> MLP head -> 2 logits
```

## Resize Strategy

The script uses the pretrained timm evaluation transform for `vit_base_patch16_224.augreg_in21k`:

```text
Resize shortest side to 248 with bicubic interpolation
CenterCrop to 224x224
Normalize with mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
```

This keeps preprocessing aligned with the pretrained checkpoint. It may crop wide FoS RGB frames, so inspect transformed samples before trusting the vision baseline.

## Device

```bash
--device auto
--device cpu
--device mps
--device cuda
--device cuda:0
```

Inside Codex's sandbox MPS may appear unavailable; in a normal terminal on this machine, MPS works.

## Smoke Forward

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python vision-detector.py \
  --smoke-model \
  --device cpu \
  --batch-size 1
```

## Train Head

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python vision-detector.py \
  --train feeling-of-success/train.csv \
  --eval feeling-of-success/eval.csv \
  --dataset-root feeling-of-success \
  --device cpu \
  --batch-size 8 \
  --train-steps 5000 \
  --eval-every-steps 250
```

Training writes TensorBoard scalars by default:

```text
runs/vision/<run-name>/train/loss
runs/vision/<run-name>/train/accuracy
runs/vision/<run-name>/train/lr
```

Use `--run-name` to give a run a stable label, and `--tensorboard-logdir` to choose where event files are written:

```bash
  --run-name vision-baseline \
  --tensorboard-logdir runs/vision
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
runs/vision/<run-name>/checkpoints/latest.pth
runs/vision/<run-name>/checkpoints/best_loss.pth
runs/vision/<run-name>/checkpoints/best_accuracy.pth
runs/vision/<run-name>/checkpoints/best_eval_loss.pth
runs/vision/<run-name>/checkpoints/best_eval_accuracy.pth
runs/vision/<run-name>/checkpoints/step_000050.pth
```

Use `--checkpoint-every-steps N` to control periodic `step_*.pth` snapshots, or `--checkpoint-every-steps 0` to keep only `latest` and best checkpoints. Use `--checkpoint-dir /path/to/dir` to put them somewhere else. These checkpoints store the MLP head, optimizer state, metrics, and safe CLI metadata; pass `--save-full-checkpoints` only when you also need full detector weights.

## Save And Reload Head

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python vision-detector.py \
  --train dataset-limited/train.csv \
  --eval dataset-limited/eval.csv \
  --dataset-root dataset-limited \
  --device cpu \
  --batch-size 1 \
  --max-train-samples 5 \
  --train-steps 5 \
  --save-head-checkpoint /private/tmp/vision-smoke-head.pth
```

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python vision-detector.py \
  --smoke-model \
  --device cpu \
  --batch-size 1 \
  --checkpoint-head /private/tmp/vision-smoke-head.pth
```

## Infer On Eval CSV

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python vision-detector.py \
  --eval dataset-limited/eval.csv \
  --dataset-root dataset-limited \
  --device cpu \
  --checkpoint-head /private/tmp/vision-smoke-head.pth \
  --prediction-output /private/tmp/vision-eval-predictions.csv
```
