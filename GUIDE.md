# Study: The value of tactile sensors in grasp prediction

## Step 1: Dataset and preprocessing

Primary Dataset from [Feeling of Success](https://sites.google.com/view/the-feeling-of-success). Data archive is [h5.zip](https://opara.zih.tu-dresden.de/bitstreams/4ef8383c-bcae-4bf6-9bcd-1242488624b5/download) 

Sparsh's tactile encoder already understands how to use GelSight images - not much preprocessing required besides rotation

The vision encoder vit_base_patch16_224 only operates on 224x224 image patches and I suspect maintaining aspect ratio is important. 

TODO: explain preprocessing steps and include a sample training script following a similar pattern as used later for other scripts in this document

## Detector setup

All three detectors use frozen pretrained encoders for the headline comparison and train only a small two-layer MLP classification head. The input frames are limited to the safe pre-outcome views: `before` and `during`. The `after` frames are intentionally excluded because they can leak the grasp outcome.

### Tactile-only detector

`tactile-detector.py` uses the Sparsh ViT encoder with the local `facebook/sparsh-dino-base` weights loaded from:

```text
checkpoints/sparsh-dino-base/dino_vitbase.safetensors
```

For each sample, the model reads GelSight RGB images from `before` and `during`, concatenated as a 6-channel tensor. In the main runs this is the Sparsh `base` model, whose token embedding width is 768. The encoder returns patch/register tokens, and the detector mean-pools tokens to one 768-dimensional tactile embedding.

The tactile MLP head is:

```text
768 -> 192 -> 2
```

where the final two logits represent `failure` and `success`.

### Vision-only detector

`vision-detector.py` uses the timm model:

```text
vit_base_patch16_224.augreg_in21k
```

The encoder is loaded with `num_classes=0`, so it returns a 768-dimensional image embedding instead of classification logits. The `during` and `before` RGB frames are encoded independently by the same frozen ViT. Their embeddings are concatenated into a 1536-dimensional vector.

The vision MLP head is:

```text
1536 -> 384 -> 2
```

### Fusion detector

`joint-detector.py` is a late-fusion model. It uses the same frozen Sparsh tactile encoder and the same frozen timm ViT vision encoder. The detector concatenates:

```text
tactile_embedding      768
vision_during          768
vision_before          768
--------------------------
fusion input          2304
```

The fusion MLP head is:

```text
2304 -> 1152 -> 2
```

This keeps the comparison fair: fusion receives both modalities, but it does not use extra labels, post-outcome frames, or a different pretraining source from the single-modality baselines.

## Training

```bash
# ideally run this from inside the repo's top-level directory
conda activate monty-tactile

REPO=`pwd`  # replace with actual repo location
DATASET=datasets/feeling-of-success.  # replace with actual location of pre-processed dataset
PROGRESS=progress  # replace with intended location for checkpoints and progress results

python $REPO/tactile-detector.py \
  --train $DATASET/train.csv \
  --eval $DATASET/eval.csv \
  --dataset-root $DATASET \
  --device cuda \
  --checkpoint-encoder $REPO/checkpoints/sparsh-dino-base/dino_vitbase.safetensors \
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
  --dataset-root f$DATASET/ \
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
  --tactile-checkpoint $REPO/checkpoints/sparsh-dino-base/dino_vitbase.safetensors \
  --batch-size 64 \
  --train-steps 5000 \
  --eval-every-steps 250 \
  --run-name joint-extended \
  --tensorboard-logdir $PROGRESS/joint \
  --checkpoint-every-steps 250

# fine-tune the joint detector for another 1000 steps
python3 $REPO/joint-detector.py \
  --train $DATASET/train.csv \
  --eval $DATASET/eval.csv \
  --dataset-root $DATASET \
  --device cuda \
  --sensor random \
  --tactile-checkpoint $REPO/checkpoints/sparsh-dino-base/dino_vitbase.safetensors \
  --head-checkpoint $PROGRESS/joint/joint-extended/checkpoints/best_eval_accuracy.pth \
  --lr 1e-4
  --batch-size 64 \
  --train-steps 1000 \
  --eval-every-steps 100 \
  --run-name joint-finetuning \
  --tensorboard-logdir $PROGRESS/joint \
  --checkpoint-every-steps 100
```

## Evaluation

```bash
# ideally run this from inside the repo's top-level directory
conda activate monty-tactile

REPO=`pwd`  # replace with actual repo location
DATASET=datasets/feeling-of-success.  # replace with actual location of pre-processed dataset
PROGRESS=progress  # replace with intended location for checkpoints and training products
REPORTS=reports  # replace with intended location for final results and reports

python $REPO/tactile-detector.py \
  --eval $DATASET/test.csv \
  --dataset-root $DATASET \
  --device cuda \
  --checkpoint-encoder $REPO/checkpoints/sparsh-dino-base/dino_vitbase.safetensors \
  --checkpoint-head $PROGRESS/tactile/tactile-extended/checkpoints/best_eval_accuracy.pth \
  --sensor gelsightA \
  --prediction-output $REPORTS/tactile/tactile-extended/test.csv

python3 $REPO/vision-detector.py \
  --eval $DATASET/test.csv \
  --dataset-root $DATASET/ \
  --device cuda \
  --checkpoint-head $PROGRESS/vision/vision-extended/checkpoints/best_eval_accuracy.pth \
  --prediction-output $REPORTS/vision/vision-extended/test.csv

python3 $REPO/joint-detector.py \
  --eval $DATASET/test.csv \
  --dataset-root $DATASET \
  --device cuda \
  --sensor gelsightA \
  --tactile-checkpoint $REPO/checkpoints/sparsh-dino-base/dino_vitbase.safetensors \
  --head-checkpoint $PROGRESS/joint/joint-finetuning/checkpoints/best_eval_accuracy.pth \
  --prediction-output reports/joint/joint-finetuning/test.csv
```

The final metrics were computed from the prediction CSVs in `reports/` using `results_analysis.ipynb`. The visually challenging subset is defined by object membership in `possibly-visually-difficult.csv`.

### Test split

| subset | detector | n | accuracy | macro F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| all | vision | 1185 | 0.7477 | 0.6731 | 0.6892 |
| all | tactile | 1185 | 0.8819 | 0.8554 | 0.9337 |
| all | fusion | 1185 | 0.8321 | 0.7991 | 0.8776 |
| visually challenging | vision | 99 | 0.7778 | 0.6265 | 0.7002 |
| visually challenging | tactile | 99 | 0.9091 | 0.8279 | 0.9866 |
| visually challenging | fusion | 99 | 0.8586 | 0.7514 | 0.8870 |

Headline accuracy lifts on the test split:

| subset | vision acc | tactile acc | fusion acc | tactile_lift | fusion_lift |
|---|---:|---:|---:|---:|---:|
| all | 0.7477 | 0.8819 | 0.8321 | 0.1342 | 0.0844 |
| visually challenging | 0.7778 | 0.9091 | 0.8586 | 0.1313 | 0.0808 |

Test split confusion matrices are ordered as rows = ground truth `[failure, success]`, columns = prediction `[failure, success]`.

| subset | detector | confusion matrix |
|---|---|---|
| all | vision | `[[160, 171], [128, 726]]` |
| all | tactile | `[[269, 62], [78, 776]]` |
| all | fusion | `[[253, 78], [121, 733]]` |
| visually challenging | vision | `[[7, 5], [17, 70]]` |
| visually challenging | tactile | `[[11, 1], [8, 79]]` |
| visually challenging | fusion | `[[10, 2], [12, 75]]` |

### Full dataset analysis

This combines train, eval, and test prediction CSVs in memory for analysis only; the CSV files are not merged on disk.

| subset | detector | n | accuracy | macro F1 | AUROC |
|---|---:|---:|---:|---:|---:|
| all | vision | 9296 | 0.7781 | 0.7566 | 0.8285 |
| all | tactile | 9296 | 0.8541 | 0.8366 | 0.9269 |
| all | fusion | 9296 | 0.8816 | 0.8705 | 0.9401 |
| visually challenging | vision | 2353 | 0.8062 | 0.7842 | 0.8699 |
| visually challenging | tactile | 2353 | 0.8819 | 0.8661 | 0.9469 |
| visually challenging | fusion | 2353 | 0.8993 | 0.8874 | 0.9517 |

Headline accuracy lifts on the full dataset analysis:

| subset | vision acc | tactile acc | fusion acc | tactile_lift | fusion_lift |
|---|---:|---:|---:|---:|---:|
| all | 0.7781 | 0.8541 | 0.8816 | 0.0761 | 0.1035 |
| visually challenging | 0.8062 | 0.8819 | 0.8993 | 0.0756 | 0.0931 |

The combined confusion matrix figures are available under:

```text
reports/results_analysis/confusion_matrices/
```

## Understanding results

Will be filled in manually
