# RLHF with TRL Library

## Project Description

This project implements a **full Reinforcement Learning with Human Feedback (RLHF) pipeline using the TRL library**.

This stage focuses on:

* Training stability
* Experiment tracking
* Performance comparison
* Research-grade logging
* Alignment metric evaluation

The pipeline includes:

* PPO alignment training
* DPO alignment training
* Reward-based optimisation
* Automatic alignment evaluation
* Experiment tracking using WandB and TensorBoard

## Experiment Overview

This experiment evaluates RLHF alignment performance using:

* Base model: GPT-2
* Preference dataset: Human-Like-DPO Dataset
* RLHF methods:

  * PPO (reward-based)
  * DPO (preference-based)

The goal is to analyse:

* Reward vs alignment correlation
* Training dynamics
* Policy improvement trends
* Stability differences between PPO and DPO

## Dataset

Dataset used:

```
HumanLLMs/Human-Like-DPO-Dataset
```

Data structure:

* prompt
* chosen response
* rejected response

Dataset split:

```
Train subset: 2048 samples
Validation subset: 512 samples
Split seed: 42
```

Dataset usage:

| Method     | Dataset Usage                 |
| ---------- | ----------------------------- |
| PPO        | prompts only                  |
| DPO        | full preference pairs         |
| Evaluation | chosen responses as reference |

## Models

### Policy Model

```
GPT-2
```

Used for:

* PPO training
* DPO training

### Reward Model

```
lvwerra/distilbert-imdb
```

Used to compute scalar reward during PPO rollouts.

This model outputs sentiment-style reward scores which serve as proxy preference signals.

## Required Libraries

Install dependencies:

```
pip install transformers==4.45.2 datasets accelerate trl==0.11.3 wandb evaluate tensorboard weave
```

### Library Purpose

| Library              | Role                           |
| -------------------- | ------------------------------ |
| transformers         | Model backbone                 |
| trl                  | PPO & DPO training             |
| datasets             | Preference dataset             |
| wandb                | Experiment tracking            |
| tensorboard          | Training visualisation         |
| evaluate             | Alignment metrics              |
| matplotlib / seaborn | Plotting                       |
| weave                | Experiment tracking (optional) |

## Hardware Requirements

Minimum:

* GPU with ≥ 12GB VRAM

Recommended:

* GPU ≥ 16GB
* CUDA environment
* Stable internet for WandB logging

Estimated runtime:

| Stage        | Time      |
| ------------ | --------- |
| PPO training | 1–2 hours |
| DPO training | 40–80 min |
| Evaluation   | 20 min    |

## RLHF Training Pipeline

Pipeline:

```
Preference Dataset
        ↓
PPO Training (Reward Model)
        ↓
Alignment Evaluation
        ↓
DPO Training
        ↓
Final Analysis
```

## PPO Objective

TRL PPO optimises:

```
L = − E[min(r * A, clip(r) * A)] + β KL
```

Where:

* r = policy probability ratio
* A = reward advantage
* KL = divergence penalty

### Reward Normalisation

Critical stability mechanism:

```
r_norm = (r − running_mean) / running_std
```

Prevents:

* Gradient explosion
* Reward collapse
* PPO instability

## DPO Objective

TRL DPO optimises:

```
L_DPO = − log σ( β[(π_chosen − π_rejected) − (π_ref_chosen − π_ref_rejected)] )
```

Advantages:

* No reward model needed
* More stable optimisation
* Faster convergence

## Alignment Metrics

Alignment is evaluated using:

* BLEU
* ROUGE-L
* BERTScore

Evaluation procedure:

1. Generate model responses
2. Compare with chosen responses
3. Compute metric averages

This provides automatic proxy for human alignment.

## Running the Experiment

Run full experiment:

```
python trl_rlhf_experiment.py
```

Execution stages:

1. Initialise logging (WandB + TensorBoard)
2. Load models
3. Load dataset
4. PPO training loop
5. Scheduled alignment evaluation
6. PPO checkpoint saving
7. DPO training
8. Plot generation
9. Experiment logging finalisation

## Experiment Logging

### WandB logs

Logged variables:

* reward
* BLEU
* ROUGE
* BERTScore
* PPO training stats

### TensorBoard logs

```
runs/<experiment_name>/logs
```

Logs include:

* reward curves
* alignment curves
* training statistics

## Output Directory Structure

```
runs/
 ├── checkpoints/
 │    ├── ppo_epoch_X
 │    └── dpo/
 ├── logs/
 │    └── reward_alignment.csv
 └── plots/
      ├── reward_alignment.png
      └── correlation.png
```

## Evaluation Procedure

Evaluation includes:

* Reward vs alignment trend analysis
* Metric correlation heatmap
* Training stability inspection
* Qualitative generation comparison

No human evaluation is conducted due to time constraints.

Dataset preference signals act as proxy.

## Key Experimental Features

This implementation includes:

* Scheduled alignment evaluation
* Reward normalisation for PPO stability
* PPO value-head training
* Reference policy freezing in DPO
* Automatic checkpoint saving
* Correlation analysis between reward and alignment
* Experiment tracking across multiple frameworks

## Known Limitations

* Reward model is proxy (sentiment-based)
* Small dataset subset used
* Single base model used
* No hyperparameter search
* No human evaluation
* Limited training steps

These constraints reflect computational limits.

## Purpose of Part 2

This stage aims to:

* Compare manual RLHF vs framework RLHF
* Analyse PPO vs DPO training behaviour
* Measure alignment improvements quantitatively
* Demonstrate research-grade experiment workflow