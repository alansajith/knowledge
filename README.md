# MT-LDI-MDS

**Multi-Teacher Lightweight Distilled Intelligence – Misbehavior Detection System**

A research project for IoV (Internet of Vehicles) security that improves upon single-teacher distillation by using **three specialized teacher LLMs** and a **learnable weighted aggregator** to fuse their knowledge into a lightweight BiLSTM student.

Hardware target: **NVIDIA GPU cluster (Google Cloud / University cluster — CUDA via PyTorch)**.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Setup (GPU Cluster)](#setup-gpu-cluster)
- [Project Structure](#project-structure)
- [Run Order](#run-order)
- [Multi-Teacher Loss Function](#multi-teacher-loss-function)
- [Interpreting the λ Weights](#interpreting-the-λ-weights)
- [Notes & Limitations](#notes--limitations)
- [Citations](#citations)

---

## Architecture Overview

```
VeReMi BSM Data
      │
      ├──► Teacher A (Qwen3-8B) ──► embeddings_A ──┐
      │      DoS + Sybil attacks                     │
      │                                              ▼
      ├──► Teacher B (Qwen3-8B) ──► embeddings_B ──► WeightedAggregator
      │      Position attacks                       (learnable λA, λB, λC)
      │                                              │
      ├──► Teacher C (Qwen3-8B) ──► embeddings_C ──┘
             Speed + Replay attacks
                                                   │
                                                   ▼
                                            BiLSTM Student
                                            ├── Embedding (input_dim → 128)
                                            ├── 2× Conv1D (64, k=3, same)
                                            ├── Permute
                                            ├── 3× BiLSTM (h=128, bi, dropout=0.3)
                                            ├── Multi-Head Attention (4 heads, 256)
                                            ├── Residual + LayerNorm
                                            ├── Global Average Pooling
                                            ├── FC(256→128) + BN + Dropout(0.3)
                                            └── Output(128 → 9)
```

### Teacher Specialisation

| Teacher | Attack labels | Attack families |
|---------|--------------|-----------------|
| **A** | 1, 2 | DoS, Sybil |
| **B** | 3, 4, 5 | Position spoofing (fixed, random, eventual stop) |
| **C** | 6, 7, 8 | Speed attacks + replay (fixed, random, data replay) |
| *(all)* | 0 | Benign is included in every subset for contrast |

---

## Setup (GPU Cluster)

> All commands assume you are in the repository root.

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r mt_ldi_mds/requirements.txt
```

### 2. Ensure CUDA is available

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True` on a CUDA-enabled machine.

### 3. Download the VeReMi Extension dataset

Place the CSV in the repo root or inside `mt_ldi_mds/data/`:

```bash
# If you already have Veremi_final_dataset.csv in the repo root:
cp Veremi_final_dataset.csv mt_ldi_mds/data/veremi.csv
```

If you don't have it yet, download from the official repository:  
https://github.com/josephkamel/VeReMi-Dataset

---

## Project Structure

```
mt_ldi_mds/
├── data/
│   └── split_dataset.py          # Preprocess, SCL, Z-score, NN search, split A/B/C
├── teachers/
│   └── finetune_teacher.py       # LoRA fine-tuning via peft + embedding extraction
├── aggregator/
│   └── weighted_aggregator.py    # Fixed / learnable λ fusion + LinearAdapter
├── student/
│   └── bilstm_student.py         # Exact BiLSTM paper architecture
├── training/
│   └── train_multiteacher.py     # Main multi-teacher distillation loop
├── evaluation/
│   └── evaluate.py               # Test evaluation + 3 comparison experiments
├── requirements.txt
└── README.md
```

## Run Order

### Step 1 – Data Preprocessing

```bash
cd mt_ldi_mds
python data/split_dataset.py --max-per-class 100000
```

This produces:
- `data/split_A.csv` (classes 0, 1, 2)
- `data/split_B.csv` (classes 0, 3, 4, 5)
- `data/split_C.csv` (classes 0, 6, 7, 8)

Each row includes a `nn_variability` column computed from the top-5 same-class nearest neighbors in the SCL embedding space.

### Step 2 – Teacher Fine-Tuning & Embedding Extraction

Repeat for each teacher:

```bash
python teachers/finetune_teacher.py --teacher A
python teachers/finetune_teacher.py --teacher B
python teachers/finetune_teacher.py --teacher C
```

What happens:
1. Converts the split CSV into instruction-tuning `train.jsonl` / `valid.jsonl`
2. Fine-tunes Qwen/Qwen3-8B with LoRA via `peft` (rank=16, alpha=32)
3. Saves adapters to `teachers/teacher_{A|B|C}_adapters/`
4. Extracts mean-pooled last-hidden-state embeddings and saves them as `teachers/embeddings_{A|B|C}.npy`

To skip LoRA and only run embedding extraction (e.g. for debugging):
```bash
python teachers/finetune_teacher.py --teacher A --skip-lora
```

### Step 3 – Multi-Teacher Distillation

```bash
CUDA_VISIBLE_DEVICES=0 python training/train_multiteacher.py --mode learnable --epochs 50
```

Arguments:
- `--mode {fixed,learnable}` – aggregator mode (default: `learnable`)
- `--kd-teachers ABC` – which teachers contribute to KD (default: `ABC`)
- `--epochs 50` – training epochs
- `--batch-size 32`
- `--lr 1e-3`
- `--checkpoint training/best_model.pt`

The script:
- Loads the full VeReMi dataset and does a stratified 70/15/15 split
- Loads teacher embeddings via **memory mapping** (avoids OOM on large files)
- Projects raw teacher embeddings (4096-dim) → 128-dim via lightweight `TeacherAligner` heads
- Trains the student with the multi-teacher loss
- Saves the best checkpoint by validation accuracy
- Logs metrics per epoch to `training/training_log.csv`

### Step 4 – Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate.py --auto-train
```

The `--auto-train` flag automatically trains missing baseline checkpoints:

1. **Single-Teacher Baseline** (`training/best_model_single.pt`) – only Teacher A KD
2. **Fixed Weights** (`training/best_model_fixed.pt`) – λ = (1/3, 1/3, 1/3)
3. **Learned Weights** (`training/best_model.pt`) – main contribution

After all checkpoints are ready it evaluates the held-out test set for each, prints a comparison summary table, and saves everything to `evaluation/results.json`.

---

## Multi-Teacher Loss Function

This is the **core contribution** of MT-LDI-MDS.

```
L_total = L_CE  +  λ₁ · L_KD_A  +  λ₂ · L_KD_B  +  λ₃ · L_KD_C
```

| Term | Definition |
|------|------------|
| **L_CE** | `CrossEntropyLoss(student_logits, true_labels)` |
| **L_KD_A** | `MSELoss(student_intermediate_features, projected_teacher_A_embedding)` |
| **L_KD_B** | `MSELoss(student_intermediate_features, projected_teacher_B_embedding)` |
| **L_KD_C** | `MSELoss(student_intermediate_features, projected_teacher_C_embedding)` |
| **λ₁, λ₂, λ₃** | Learnable softmax-normalised weights from `WeightedAggregator` |

Student intermediate features are the **128-dim vector after BatchNorm and Dropout**, just before the final classification layer. Teacher embeddings come from the last hidden layer of Qwen3-8B and are **projected from 4096-dim → 128-dim** by lightweight linear alignment heads (`TeacherAligner`) so the MSE is dimensionally valid.

The λ weights share the same optimisation step as the student and alignment heads, meaning the model **learns how much to trust each teacher automatically**.

---

## Interpreting the λ Weights

During training the script prints the current λ values every 10 epochs (and also on demand via `aggregator.print_weights()`).

Example output:

```
[WeightedAggregator] λ weights — A=0.5123, B=0.2876, C=0.2001
```

### What the numbers mean

- **λ_A ≈ 0.51** → Teacher A contributes about 51 % of the distillation weight. In this project Teacher A specialises in DoS and Sybil attacks, so a high λ_A suggests the student finds this teacher's embedding space most informative for the current batch of data.
- **λ_B ≈ 0.29** → Teacher B (position attacks) is the second-most trusted source.
- **λ_C ≈ 0.20** → Teacher C (speed / replay attacks) receives the lowest weight.

### Interpretation rules of thumb

| Scenario | Likely meaning |
|----------|----------------|
| One λ dominates (>0.70) | The student is essentially relying on a single teacher; the dataset may be heavily skewed toward that teacher's attack classes, or the other teachers produce noisier embeddings. |
| λ values are roughly uniform (~1/3 each) | All teachers contribute equally. This usually happens with `mode="fixed"` or when attack classes are perfectly balanced. |
| λ shifts across epochs | The student adapts its trust dynamically. Early epochs may favour one teacher; later epochs may rebalance as the student learns its own representation. |
| λ for a teacher stays near zero | That teacher's embeddings are not helpful for the student. Check whether the teacher was fine-tuned correctly or whether its specialisation overlaps poorly with the student's task. |

### Important caveats

- λ weights are **softmax-normalised** so they always sum to 1.0. An increase in one weight necessarily decreases the others.
- λ applies **globally** (per batch, during training), not per-sample. The current implementation does not sample-wise gating.
- If a teacher embedding file is missing, its KD term is skipped and the remaining λ weights still sum to 1.0 among the active teachers.

---

## Notes & Limitations

- **Embedding memory**: Raw Qwen3-8B embeddings are 4096-dim float32. For the full 22 M-row VeReMi dataset a single teacher embedding file would be ~351 GB.
  - **Practical workaround**: The training script automatically loads `.npy` files via `np.memmap`, avoiding full RAM residency.
  - **Recommended pipeline**: Extract teacher embeddings **only for the training set** (or downsample the dataset before training).
- **Teacher fine-tuning**: LoRA fine-tuning uses `peft` + `transformers` with CUDA. The fine-tuned adapter is saved separately and reloaded at embedding-extraction time.
- **Teacher alignment**: Because raw teacher hidden states (4096-dim) do not match the student's 128-dim intermediate features, `TeacherAligner` projections are added. This is standard practice in feature-based knowledge distillation.

---

## Citations

If you use this codebase, please cite the original VeReMi dataset and our multi-teacher distillation extension:

```bibtex
@dataset{veremi2019,
  title={VeReMi: A Dataset for Comparable Evaluation of Misbehavior Detection in VANETs},
  author={Kamel, Joseph and others},
  year={2019},
  url={https://github.com/josephkamel/VeReMi-Dataset}
}
```

---

*Built for NVIDIA GPU clusters — PyTorch CUDA backend.*