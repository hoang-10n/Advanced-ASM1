# RLHF Pipeline: GPT-2 with PPO

This repository implements a **full Reinforcement Learning from Human Feedback (RLHF) pipeline** using the Hugging Face `trl` (Transformer Reinforcement Learning) library. The project transitions a base model through Supervised Fine-Tuning (SFT), Reward Modeling (RM), and Proximal Policy Optimization (PPO) to align model outputs with human preferences.

## System Environment & Hardware

The training was conducted in a **Linux (Miniconda)** environment with the following specifications:

| Component           | Specification                                    |                                     |
| :------------------ | :----------------------------------------------- | ----------------------------------- |
| **GPU**             | NVIDIA GeForce RTX 3060 (12GB VRAM)              |                                     |
| **Driver / CUDA**   | Driver: 560.94                                   | CUDA: 12.6                          |
| **Memory**          | 16GB System RAM                                  | 20GB Total GPU Mem (12GB Dedicated) |
| **GPU Utilization** | ~10.6GB / 12GB used during PPO phase (100% Load) |                                     |

## Model Architectures

### 1. Policy & Reference Models (`gpt2`)

* **Source:** [Hugging Face `gpt2`](https://huggingface.co/gpt2)
* **Parameters:** ~124 Million
* **Architecture:** 12-layer, 768-hidden, 12-heads.
* **Role:** The **Policy Model** is trained via PPO to generate text. The **Reference Model** stays frozen to provide a KL-divergence anchor.

### 2. Reward Model (`gpt2-sequence-classification`)

* **Source:** Initialized from `gpt2` weights with a custom `SequenceClassification` head.
* **Parameters:** ~124 Million + 1 Linear Layer ($768 \times 1$).
* **Role:** Acts as a regression model that outputs a scalar value representing the "quality" or "human-likeness" of a generated response.

## Dataset: Human-Like DPO Dataset

The project uses the **Human-Like-DPO-Dataset** to provide preference pairs for alignment.

* **Source:** [`HumanLLMs/Human-Like-DPO-Dataset`](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)
* **Total Size:** ~12.5k preference pairs.
* **Project Subset:** * **Train:** 2,048 samples (sub-sampled for rapid iteration).

  * **Test:** 512 samples.
* **Structure:**

  * `prompt`: The user input query.
  * `chosen`: The preferred, high-quality, or more "human-like" response.
  * `rejected`: The lower-quality or less desirable response.

## Setup & Installation

```bash
# Create Environment
conda create -n rlhf python=3.11 -y
conda activate rlhf

# Install Specific Dependencies
pip install transformers==4.45.2 trl==0.11.3 datasets accelerate evaluate 
pip install rouge_score bert_score sacrebleu matplotlib pandas tensorboard
```

## RLHF Training Pipeline

### . Supervised Fine-Tuning (SFT)

The model is trained on "chosen" responses to establish a baseline of high-quality text generation.

* **Config:** `SFTConfig` with `max_steps=1000`, `learning_rate=2e-5`.

### . Reward Modeling (RM)

The model learns a Bradley-Terry ranking objective to score preferred responses higher than rejected ones.

* **Loss:** Log-sigmoid with **0.1 label smoothing** and **Weight Decay (0.01)** to prevent overfitting.
* **Stability:** Includes a loss threshold (0.01) early-stopping mechanism.

### . PPO Alignment

The Policy Model is optimized to maximize the Reward Model's output.

* **Objective:** $L = - \mathbb{E}[\min(r \cdot A, \text{clip}(r) \cdot A)] + \beta KL$
* **Stability Mechanisms:** * **Reward Normalization:** $r_{norm} = \frac{r - \mu}{\sigma}$ to prevent gradient explosion.

  * **KL Control:** High initial KL penalty (0.5) to keep the model grounded.
  * **Diversity:** Repetition penalty (1.3) and `no_repeat_ngram_size=3` to avoid reward hacking.

## Monitoring & Output

Results are stored in `runs/ppo_gpt2/`:

* **Checkpoints:** Saved for SFT, Reward Model (best/final), and PPO iterations.
* **Logs:** CSV/TensorBoard files tracking KL divergence, Reward Mean, and Entropy.
* **Metrics:** Final evaluation using **BLEU**, **ROUGE-L**, and **BERTScore** to provide a proxy for human alignment.

### Directory Structure

```text
runs/ppo_gpt2/
 ├── checkpoints/    # Model weights
 ├── logs/           # reward_training_log.csv, ppo_training_log.csv
 └── plots/          # reward_loss_curve.png, ppo_loss.png
```

## Known Limitations

* **Proxy Reward:** The model is trained on a relatively small subset (2k samples).
* **VRAM Constraints:** Limited to small-scale models (GPT-2) to fit all three models (Policy, Ref, Reward) on a single 12GB GPU.
* **Negative KL:** Occasional negative KL spikes suggest a need for tighter hyperparameter tuning on generation temperature and top-p values.
