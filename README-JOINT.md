# Joint Detector CLI

`joint-detector.py` combines the tactile Sparsh encoder and the timm ViT vision encoder for a late-fusion grasp outcome classifier.

## Inputs

The script uses only pre-outcome frames:

```text
tactile/gelsight{A/B}_during.png
tactile/gelsight{A/B}_before.png
rgb/during.png
rgb/before.png
```

`after` images are not supported.

Tactile follows Sparsh ordering:

```text
[tactile_during_rgb, tactile_before_rgb] -> [6, 320, 240]
```

Vision uses the shared timm ViT on each RGB frame:

```text
vision_during -> embedding
vision_before -> embedding
```

The joint head receives:

```text
[tactile_embedding, vision_during_embedding, vision_before_embedding]
```

## Encoders

Tactile:

```text
Sparsh DINO base
checkpoints/sparsh-dino-base/dino_vitbase.safetensors
```

Vision:

```text
timm vit_base_patch16_224.augreg_in21k
```

Both encoders are frozen by default. Pass `--train-encoders` only for an explicit fine-tuning ablation.

## Smoke Forward

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python joint-detector.py \
  --smoke-model \
  --device cpu \
  --batch-size 1 \
  --sensor gelsightA \
  --tactile-model-size base \
  --tactile-checkpoint checkpoints/sparsh-dino-base/dino_vitbase.safetensors
```

Expected output:

```text
logits shape: (1, 2)
```

## Train Joint Head

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python joint-detector.py \
  --train feeling-of-success/train.csv \
  --eval feeling-of-success/eval.csv \
  --dataset-root feeling-of-success \
  --device cpu \
  --batch-size 8 \
  --train-steps 5000 \
  --eval-every-steps 250 \
  --sensor gelsightA \
  --tactile-model-size base \
  --tactile-checkpoint checkpoints/sparsh-dino-base/dino_vitbase.safetensors \
  --save-head-checkpoint /private/tmp/joint-smoke-head.pth
```

Training writes TensorBoard scalars by default:

```text
runs/joint/<run-name>/train/loss
runs/joint/<run-name>/train/accuracy
runs/joint/<run-name>/train/lr
```

Use `--run-name` to give a run a stable label, and `--tensorboard-logdir` to choose where event files are written:

```bash
  --run-name joint-baseline \
  --tensorboard-logdir runs/joint
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
runs/joint/<run-name>/checkpoints/latest.pth
runs/joint/<run-name>/checkpoints/best_loss.pth
runs/joint/<run-name>/checkpoints/best_accuracy.pth
runs/joint/<run-name>/checkpoints/best_eval_loss.pth
runs/joint/<run-name>/checkpoints/best_eval_accuracy.pth
runs/joint/<run-name>/checkpoints/step_000050.pth
```

Use `--checkpoint-every-steps N` to control periodic `step_*.pth` snapshots, or `--checkpoint-every-steps 0` to keep only `latest` and best checkpoints. Use `--checkpoint-dir /path/to/dir` to put them somewhere else. These checkpoints store the joint MLP head, optimizer state, metrics, and safe CLI metadata; pass `--save-full-checkpoints` only when you also need full detector weights.

## Reload Head And Infer

```bash
KMP_DUPLICATE_LIB_OK=TRUE MPLCONFIGDIR=/private/tmp/mplconfig-monty-tactile \
/Users/adwait/miniconda3/envs/monty-tactile/bin/python joint-detector.py \
  --eval dataset-limited/eval.csv \
  --dataset-root dataset-limited \
  --device cpu \
  --batch-size 4 \
  --sensor gelsightA \
  --tactile-model-size base \
  --tactile-checkpoint checkpoints/sparsh-dino-base/dino_vitbase.safetensors \
  --head-checkpoint /private/tmp/joint-smoke-head.pth \
  --prediction-output /private/tmp/joint-eval-predictions.csv
```

## Hardware

```bash
--device auto
--device cpu
--device mps
--device cuda
--device cuda:0
```

Inside Codex's sandbox, MPS may appear unavailable. In a normal terminal on this machine, MPS works.
