**Monty Intelligence Platform**

**Founding Research Scientist - 1-Week Take-Home**

**Take-Home A - Does Touch Predict Grasp Outcome Better than Vision?**

Tasks #1 / #2 · grasp & lift stability and transparent/shiny grasping · classification

**Time:** ~1 week, part-time (~12-15 focused hours). **Compute:** any cloud GPU, reimbursed (Section 4). **Data:** open-source only. **What we're really assessing:** how you design a clean, honest, reproducible experiment and reason about where touch helps.

# 1\. Context - what we are testing

Monty is building an intelligence platform for robotic touch. The core bet is that **tactile sensing closes the failure gap that vision-only systems hit** on slip-prone, transparent, and deformable objects - and that a shared data/model/eval loop turns that into a product. This take-home is a scoped, one-week slice of our Week-1 plan. It is deliberately small so we can see how you set up a clean, honest, reproducible experiment - not how big a model you can train.

**The claim this assignment probes:** on physically/visually ambiguous contact-rich tasks, a **tactile-only** classifier should beat a **vision-only** classifier, and a **vision+tactile fusion** classifier should be best or close to best. You will demonstrate (or fail to demonstrate, honestly) this on public data only.

**This version focuses on the core thesis as a classification problem:** predicting whether a grasp will hold, from touch vs vision vs both, with special attention to visually hard objects (transparent, shiny, low-texture) where vision is expected to struggle most.

**Use only the open-source datasets named below.** We are still collecting our own data; do not ask for or use Monty's internal data. Where a public dataset is only an approximate proxy for our real task, say so explicitly - research honesty about proxy validity is part of what we evaluate.

# 2\. Your task

Build tactile-only, vision-only, and fusion classifiers that predict **grasp outcome** (and grasp condition) on a public grasping dataset, and quantify how much touch helps - especially on visually ambiguous grasps.

**Primary dataset - Feeling of Success** (paired GelSight tactile + RGB, with grasp success/failure over many objects). Project page: <https://sites.google.com/view/the-feeling-of-success> . Optionally use YCB object metadata for object_class. This is the best public proxy for \*"does touch predict grasp outcome better than vision."\* It is **not** our exact volunteer protocol - note that caveat.

**Labels:** primary \`result\` = success / failure; \`condition_class\` ∈ {stable, unstable, heavy, light, (transparent/opaque, shiny/matte if derivable)}. Map dataset's success→success, failure→failure; RGB→vision baseline, GelSight→tactile baseline, RGB+GelSight→fusion.

**The #2 twist:** identify the **visually hard subset** (transparent / shiny / low-texture objects) - via object metadata or your own grouping - and report tactile_lift **on that subset specifically**. The hypothesis is that touch's advantage is largest there.

## Models to build (three, sharing everything except the input modality)

- **Tactile-only:** frozen \`facebook/sparsh-dino-base\` encoder on the tactile image(s), + a 2-layer MLP head. If a stream is force/glove time-series rather than tactile images, use a TCN over the T×F series + MLP, and do not directly compare TCN-stream numbers against Sparsh-image numbers without flagging the modality difference.
- **Vision-only:** frozen \`vit_base_patch16_224\` (timm) on the RGB frame(s) + a 2-layer MLP head.
- **Late-fusion:** concatenate the two frozen embeddings + a 2-layer MLP head. Fusion must not use extra frames, extra labels, or extra pretraining unavailable to the single-modality baselines.

Keep encoders **frozen** for the headline result so it is a fair representation comparison. Unfreezing the last 2 blocks (lr 1e-5) is an **optional** stretch, reported separately.

# 3\. Setup - the shared experiment contract

These are fixed so results are comparable and reproducible. Treat them as hard requirements.

| **Item**         | **Value**                                                                      |
| ---------------- | ------------------------------------------------------------------------------ |
| Python           | 3.10                                                                           |
| PyTorch          | pin the exact version in requirements.lock                                     |
| Random seed      | 42 (set Python, NumPy, torch, cuda)                                            |
| Tactile encoder  | facebook/sparsh-dino-base (frozen)                                             |
| Vision encoder   | timm vit_base_patch16_224 (frozen)                                             |
| Fusion           | late fusion: concat frozen embeddings + 2-layer MLP head                       |
| Split            | grouped, leakage-safe (see below) - never row-level random                     |
| Metrics          | accuracy, macro F1, AUROC (binary), confusion matrix                           |
| Headline numbers | tactile_lift = tactile_acc − vision_acc; fusion_lift = fusion_acc − vision_acc |

## Environment

```python
conda create -n monty-tactile python=3.10 -y && conda activate monty-tactile

pip install --upgrade pip

pip install torch torchvision torchaudio

pip install transformers accelerate datasets huggingface_hub timm

pip install numpy pandas pyarrow scikit-learn scipy opencv-python pillow einops tqdm wandb matplotlib pyyaml

pip install git+<https://github.com/facebookresearch/sparsh.git>

pip freeze > requirements.lock # commit this
```

Verify the GPU and pin the seed:

```python
python -c "import torch;print('CUDA',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

def seed_everything(s=42):
import random,numpy as np,torch
random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
```

## Data manifest schema (every dataset is converted into this before training)

episode_id, dataset_name, task_id, task_name, object_id, object_class, condition_class, clip_type, result, failure_mode, rgb_path, tactile_path, force_path, metadata_path, split_group

\`split_group\` = object_id + session_id + collector/source - the key that prevents leakage.

## Leakage-safe split

**Never** use a random row-level split. Group by \`dataset_name + task_id + object_id + session/source\` and split the **groups** 70 / 15 / 15 into \`train.csv\`, \`val.csv\`, \`test_public.csv\`. Commit the split files and record their SHA so a run is fully identified:

sha256sum data/manifests/combined_manifest_v0.csv

sha256sum data/splits/v0/\*.csv # log these SHAs to W&B / the run report

## Result-validity checklist (satisfy before reporting any comparison)

- Same git commit, requirements.lock, seed, and model config across the three baselines
- Same manifest SHA and same split files for all three
- Same label target and same public test set
- No row-level leakage and no near-duplicate object/source leakage across splits
- Mistake/failure clips are labeled, not silently mixed into the normal class
- The vision-only model cannot cheat from visible labels, obvious props, or collection artifacts

# 4\. Compute - how to get a GPU (reimbursed)

**You do not pay for compute.** Monty reimburses your GPU costs for this assignment, so pick whatever cloud is fastest for you and don't let cost slow you down. The task is light - frozen encoders + a small head - and fits comfortably on a **single GPU with ≥24 GB** (e.g., RTX 4090 24 GB, L4 24 GB, A10 24 GB, or A100 40 GB). Expect only a few GPU-hours total.

## Reserve / launch a GPU (self-serve)

- Pick a provider you like - **RunPod, Lambda Cloud, Vast.ai**, or **Google Colab Pro+** (A100/L4) all work. Create an account.
- Launch **one** single-GPU instance (≥24 GB) using a recent PyTorch/CUDA image. Choose on-demand (not a long commitment).
- SSH in (or open the notebook), clone your repo, and run the environment setup above.
- Train and evaluate. **Stop or terminate the instance the moment you are idle** - this is the single biggest cost saver and keeps reimbursement small.
- No-cost fallback: **Google Colab** (Pro/Pro+) with an A100 or L4 is fine; export your notebook plus requirements.lock and note Colab's session limits.

## Cost expectation & cap

Typical rates are roughly **\$0.40-\$2.00 / GPU-hour**, so the whole assignment usually lands **under ~\$30-\$50**. You may spend **up to \$100 reimbursed without pre-approval**; if you expect to exceed that, email **<team@usemonty.com>** first.

## How to get reimbursed

Keep your provider's **invoices / usage receipts** (instance type, hours, total). At submission, email an itemized summary (provider, GPU, hours, total cost) with the receipts attached to **<team@usemonty.com>**, subject \`GPU reimbursement - &lt;your name&gt; - take-home\`. We reimburse within **14 business days**. Datasets here are only a few GB; use the instance's local disk or a small persistent volume.

# 5\. Deliverables

A small repo (\`monty-tactile-takehome\`) we can clone and reproduce. Include:

- \`README.md\` with the **exact** commands to reproduce, and \`requirements.lock\`.
- \`data/manifests/combined_manifest_v0.csv\` and \`data/splits/v0/{train,val,test_public}.csv\` (plus the logged SHAs).
- Three trained checkpoints - \`tactile*\*\`, \`vision*\*\`, \`fusion\_\*\` (or a download link if large).
- \`reports/\` with per-model metrics JSON (accuracy, macro F1, AUROC) and a **confusion matrix image per task/label**.
- A filled **results table**: for each label/test set, vision acc, tactile acc, fusion acc, **tactile_lift**, **fusion_lift**.
- Report tactile_lift and fusion_lift **both overall and on the visually-hard subset**, with the subset definition stated.
- A confusion matrix for grasp outcome per modality, and a short note on which object types vision misreads but touch catches.
- A **one-page writeup**: \*"Where tactile beats vision, where it does not, and why."\* Include your honest read on proxy-dataset validity and the next experiment you would run with real Monty tactile data.
- Your **GPU receipts** for reimbursement.

# 6\. How we evaluate your submission

We weight **rigor and research judgment over leaderboard numbers**. A smaller, correct, well-analyzed result beats a larger, messy one. Scored roughly as:

| **Criterion**                 | **Weight** | **What we look for**                                                                                                               |
| ----------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Reproducibility & rigor       | 25%        | One-command repro; pinned deps; seed 42; manifest/split SHAs logged; grouped leakage-safe split; validity checklist satisfied.     |
| Correctness of the comparison | 20%        | Identical setup across the three baselines; no leakage; vision can't exploit artifacts; honest modality/proxy caveats.             |
| Results                       | 20%        | Three baselines run and report acc / macro F1 / AUROC + confusion matrices; tactile_lift & fusion_lift computed; numbers are sane. |
| Analysis & research judgment  | 20%        | The one-pager: which cases tactile wins/loses and why; error analysis from confusion matrices; sensible next experiment.           |
| Code quality                  | 10%        | Clean, modular, config-driven (no hardcoded paths); readable.                                                                      |
| Communication                 | 5%         | Clear README + writeup; we can reproduce without back-and-forth.                                                                   |

**What a strong result looks like:** on the visually/physically ambiguous subset, **tactile-only ≥ vision-only** and **fusion is best or close**. But we care more that the comparison is \*valid\* and the analysis is \*honest\* than about the exact deltas - a clean null result with good reasoning is a pass.

# 7\. Rules, scope & tips

- **Open-source data only.** Do not request or use Monty internal data.
- **Encoders frozen** for the headline baselines; any fine-tuning is an optional, separately-reported stretch.
- **Same seed (42) and same split files** across all three baselines; **grouped** split, no row-level random.
- **AI coding assistants are allowed** (we use them too), but you must understand and be able to defend every line - we may walk through the code with you.
- **Time-box: this is scoped to roughly one week part-time; please don't sink more than ~12-15 focused hours into it.** If time is short, ship a correct, reproducible single-label result over broad-but-shaky coverage.
- Compute is reimbursed - don't let cost block you, but **stop idle instances**.

## Suggested day-by-day (light)

| **Day** | **Focus**                                                              |
| ------- | ---------------------------------------------------------------------- |
| 1       | Env + GPU; download data; build manifest; grouped splits; commit SHAs. |
| 2       | Dataset/dataloader; tactile-only baseline trains & evals.              |
| 3       | Vision-only baseline.                                                  |
| 4       | Fusion baseline; run all three on the same split.                      |
| 5       | Eval: confusion matrices, metrics table, tactile/fusion lift.          |
| 6       | Error analysis + one-page writeup.                                     |
| 7       | Reproducibility pass, polish, submit.                                  |

# 8\. Submission

Share a **private GitHub repo** (invite **SYLLL**) or a zip, containing everything in Section 5. Email submission + GPU receipts to **<team@usemonty.com>** with subject \`Take-home &lt;X&gt; - &lt;your name&gt;\`. **Deadline: one week from the day you start** - tell us your start date when you receive this. Expect a **30-45 minute follow-up** where you walk us through your results and decisions.

Questions about scope or blockers are welcome - email **<team@usemonty.com>**. Good luck, and have fun with it.