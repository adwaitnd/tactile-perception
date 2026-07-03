# Training and running tactile & vision based grasp prediction

**This study and code is owned by Adwait Dongare and may not be used for any purposes unless explicitly allowed.**

This document describes how to run the data preprocessing, training, evaluation, and results analysis code. For the study motivation, detector design, results, and interpretation, see [STUDY.md](STUDY.md).

## Environment

Use a conda environment with either CUDA or Apple Silicon MPS support.

```bash
conda create -n monty-tactile python=3.10 pip -y
conda activate monty-tactile

# Option 1: MacOS with MPS
python -m pip install -r requirements-macos.lock

# Option 2: CUDA-equipped system
python -m pip install -r requirements-cuda.lock
```

## Preprocess Data

Download the Feeling of Success `h5.zip` archive from the dataset page linked in [STUDY.md](STUDY.md). Direct archive link:
[h5.zip](https://opara.zih.tu-dresden.de/bitstreams/4ef8383c-bcae-4bf6-9bcd-1242488624b5/download).
Then convert it into per-sample folders.

```bash
conda activate monty-tactile

REPO=`pwd`
DATASET=$REPO/datasets/feeling-of-success
RAW_DATASET=~/Downloads/fos-unprocessed

unzip ~/Downloads/h5.zip -d $RAW_DATASET

python3 $REPO/scripts/convert_h5_to_png_samples.py \
  --dataset-dir $RAW_DATASET \
  --output-dir $DATASET

python3 $REPO/scripts/add_rgb_motion_diff_metadata.py \
  --dataset-root $DATASET
```

Generate deterministic train/eval/test split CSVs:

```bash
python3 $REPO/scripts/generate_feeling_success_splits.py \
  --manifest-csv $DATASET/manifest.csv \
  --visually-difficult-csv $REPO/possibly-visually-difficult.csv \
  --output-dir $DATASET \
  --seed 40
```

Record manifest and split hashes:

```bash
shasum -a 256 \
  $DATASET/manifest.csv \
  $DATASET/train.csv \
  $DATASET/eval.csv \
  $DATASET/test.csv
```

## Train Detectors

```bash
conda activate monty-tactile

REPO=`pwd`
DATASET=datasets/feeling-of-success
PROGRESS=progress

SPARSH_CHECKPOINT=$REPO/checkpoints/sparsh-dino-base/dino_vitbase.safetensors
if [ ! -f "$SPARSH_CHECKPOINT" ]; then
  hf download facebook/sparsh-dino-base dino_vitbase.safetensors \
    --local-dir $REPO/checkpoints/sparsh-dino-base
fi

python $REPO/tactile-detector.py \
  --train $DATASET/train.csv \
  --eval $DATASET/eval.csv \
  --dataset-root $DATASET \
  --device cuda \
  --checkpoint-encoder $SPARSH_CHECKPOINT \
  --sensor random \
  --batch-size 64 \
  --train-steps 5000 \
  --eval-every-steps 250 \
  --run-name tactile-extended \
  --tensorboard-logdir $PROGRESS/tactile \
  --checkpoint-every-steps 250

python3 $REPO/vision-detector.py \
  --train $DATASET/train.csv \
  --eval $DATASET/eval.csv \
  --dataset-root $DATASET \
  --device cuda \
  --batch-size 64 \
  --train-steps 5000 \
  --eval-every-steps 250 \
  --run-name vision-extended \
  --tensorboard-logdir $PROGRESS/vision \
  --checkpoint-every-steps 250

python3 $REPO/joint-detector.py \
  --train $DATASET/train.csv \
  --eval $DATASET/eval.csv \
  --dataset-root $DATASET \
  --device cuda \
  --sensor random \
  --tactile-checkpoint $SPARSH_CHECKPOINT \
  --batch-size 64 \
  --train-steps 5000 \
  --eval-every-steps 250 \
  --run-name joint-extended \
  --tensorboard-logdir $PROGRESS/joint \
  --checkpoint-every-steps 250
```

Optional fine-tuning from the best joint checkpoint:

```bash
python3 $REPO/joint-detector.py \
  --train $DATASET/train.csv \
  --eval $DATASET/eval.csv \
  --dataset-root $DATASET \
  --device cuda \
  --sensor random \
  --tactile-checkpoint $SPARSH_CHECKPOINT \
  --head-checkpoint $PROGRESS/joint/joint-extended/checkpoints/best_eval_accuracy.pth \
  --lr 1e-4 \
  --batch-size 64 \
  --train-steps 1000 \
  --eval-every-steps 100 \
  --run-name joint-finetuning \
  --tensorboard-logdir $PROGRESS/joint \
  --checkpoint-every-steps 100
```

## Evaluate Detectors

```bash
conda activate monty-tactile

REPO=`pwd`
DATASET=datasets/feeling-of-success
REPORTS=reports

SPARSH_CHECKPOINT=$REPO/checkpoints/sparsh-dino-base/dino_vitbase.safetensors
if [ ! -f "$SPARSH_CHECKPOINT" ]; then
  hf download facebook/sparsh-dino-base dino_vitbase.safetensors \
    --local-dir $REPO/checkpoints/sparsh-dino-base
fi

python $REPO/tactile-detector.py \
  --eval $DATASET/test.csv \
  --dataset-root $DATASET \
  --device cuda \
  --checkpoint-encoder $SPARSH_CHECKPOINT \
  --checkpoint-head $REPO/checkpoints/tactile/best_eval_accuracy.pth \
  --sensor gelsightA \
  --prediction-output $REPORTS/tactile/tactile-extended/test.csv

python3 $REPO/vision-detector.py \
  --eval $DATASET/test.csv \
  --dataset-root $DATASET \
  --device cuda \
  --checkpoint-head $REPO/checkpoints/vision/best_eval_accuracy.pth \
  --prediction-output $REPORTS/vision/vision-extended/test.csv

python3 $REPO/joint-detector.py \
  --eval $DATASET/test.csv \
  --dataset-root $DATASET \
  --device cuda \
  --sensor gelsightA \
  --tactile-checkpoint $SPARSH_CHECKPOINT \
  --head-checkpoint $REPO/checkpoints/joint/best_eval_accuracy.pth \
  --prediction-output $REPORTS/joint/joint-finetune/test.csv
```

## Analyze Results

Open or execute [results_analysis.ipynb](results_analysis.ipynb). The notebook computes accuracy, macro F1, AUROC, lifts, confusion matrices, object-level accuracy, confidence tables, and error slices.

The visually challenging subset is defined by object membership in [possibly-visually-difficult.csv](possibly-visually-difficult.csv).

Generated confusion matrix images are under:

```text
reports/results_analysis/confusion_matrices/
```
