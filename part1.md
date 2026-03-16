# Manual RLHF Implementation

## Description

This project implements a **manual Reinforcement Learning with Human Feedback (RLHF) pipeline** using a decoder-only language model.

The goal of this part is to understand the internal mechanisms of RLHF training by implementing:

1. Supervised Fine-Tuning (SFT)
2. Reward Model training
3. Proximal Policy Optimisation (PPO)
4. Direct Preference Optimisation (DPO)

No specialised RLHF framework (e.g., TRL) is used in this stage.

## Dataset

Dataset used:

```
HumanLLMs/Human-Like-DPO-Dataset (Hugging Face)
```

Each sample contains:

* `prompt`
* `chosen`
* `rejected`

Dataset usage by stage:

| Stage        | Data Used                   |
| ------------ | --------------------------- |
| SFT          | prompt + chosen             |
| Reward Model | prompt + chosen vs rejected |
| PPO          | prompt only                 |
| DPO          | preference pairs            |

Dataset split:

- Train: 80% (512 data points)
- Validation: 20% (128 data points)
- Random seed: 42

## Model

Base model:

```
GPT-2
```

Reason:

* Computationally efficient
* Suitable for manual RL experimentation
* Fast convergence for small-scale experiments

## Required Libraries

Install all dependencies using:

```
pip install torch transformers datasets matplotlib accelerate
```

### Library Purpose

| Library      | Purpose                    |
| ------------ | -------------------------- |
| torch        | Neural network training    |
| transformers | Language model backbone    |
| datasets     | Preference dataset loading |
| matplotlib   | Training curve plotting    |
| accelerate   | Optional GPU optimisation  |

## Hardware Requirements

Minimum:

* GPU ≥ 8GB VRAM
* CUDA-enabled environment
* CPU not recommended

Approximate runtime (dataset subset):

| Stage        | Runtime   |
| ------------ | --------- |
| SFT          | 20–40 min |
| Reward Model | 30–60 min |
| PPO          | 40–80 min |
| DPO          | 30–60 min |

## RLHF Pipeline Overview

Pipeline:

```
Pretraining
      ↓
Supervised Fine-Tuning (SFT)
      ↓
Preference Dataset 
      ↓                              ↓
Reward Model Training
      ↓
Policy Optimization (PPO)
```

## Loss Functions

### Supervised Fine-Tuning Loss

Standard language modelling loss:

```
L_SFT = − E[ log P(chosen response | prompt) ]
```

This optimises next-token prediction likelihood.

### Reward Model Loss

Pairwise logistic preference loss:

```
L_reward = − log σ( R(chosen) − R(rejected) )
```

Where:

* R(x) is scalar reward output
* σ is sigmoid function

This trains the model to assign higher reward to preferred responses.

### PPO Objective

Simplified PPO objective:

```
L_PPO = − E[min(r * A, clip(r) * A)] + β KL(π || π_ref)
```

Where:

* r = probability ratio
* A = reward signal
* β = KL penalty coefficient

This stabilises policy updates while maximising reward.

### DPO Objective

Direct preference optimisation loss:

```
L_DPO = − log σ( β[(π_chosen − π_rejected) − (ref_chosen − ref_rejected)] )
```

This removes the need for explicit reward modelling.

## Running the Training Pipeline

Run full pipeline:

```
python manual_rlhf.py
```

Execution order:

1. Load dataset
2. Train SFT model
3. Train reward model
4. Train PPO policy
5. Train DPO policy
6. Save models
7. Plot training curves

## Model Outputs

All outputs saved to:

```
out_manual/
```

Saved models:

```
out_manual/sft/
out_manual/ppo/
out_manual/dpo/
out_manual/reward.pt
```

## Testing the Models

### Quick qualitative test

After training, test generation:

```
python test_generation.py
```

Example prompt:

```
What is the meaning of life?
```

Compare:

* SFT output
* PPO output
* DPO output

Expected observation:

* PPO → more aligned but sometimes unstable
* DPO → smoother preference alignment
* SFT → fluent but less aligned

## Evaluation Strategy

Since large-scale human evaluation is not feasible:

* Preference dataset used as proxy
* Evaluation includes:

  * Reward trends during PPO
  * DPO loss convergence
  * Qualitative generation comparison
  * Policy divergence monitoring

## Important Implementation Details

This implementation includes:

* Token-level masking for generated text
* Frozen reference policy
* KL regularisation
* Manual reward pooling
* Pairwise preference training
* Autoregressive sampling

## Known Limitations

* Small base model
* Limited training steps
* Simplified PPO implementation
* No reward normalisation
* No human evaluation
* No hyperparameter search
* No distributed training

These limitations are intentional for conceptual clarity.

## Reproducibility Notes

To reproduce results:

* Use identical dataset split seed
* Keep hyperparameters unchanged
* Use same tokenizer padding configuration
* Run pipeline sequentially

If memory errors occur:

* Reduce batch size
* Reduce max sequence length
* Use gradient accumulation

## Purpose of Manual Implementation

This stage aims to:

* Understand RLHF training dynamics
* Compare PPO vs DPO behaviour
* Identify stability challenges
* Build intuition before using RLHF frameworks