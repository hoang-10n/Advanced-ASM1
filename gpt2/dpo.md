# RLHF Pipeline: GPT-2 Alignment via DPO

This repository implements a **Direct Preference Optimization (DPO)** pipeline using the Hugging Face `trl` library. Unlike traditional RLHF, this approach skips the explicit reward modeling stage and optimizes the policy directly using preference pairs, offering a more stable and computationally efficient alignment process.


## System Environment & Hardware

The training was conducted in a **Linux (Miniconda)** environment with the following specifications:

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| **Driver / CUDA** | Driver: 560.94 / CUDA: 12.6 |
| **VRAM Usage** | \~10.5GB (Utilizing `fp16` and Gradient Checkpointing) |
| **Compute Device** | `cuda:0` (Configured via `CUDA_VISIBLE_DEVICES`) |

> **Note:** The implementation includes a `max_steps` constraint and gradient checkpointing specifically to allow `gpt2` training to fit within the 12GB VRAM limit of a 3060.


## Model Architectures

### . Policy & Reference Models (`gpt2`)

  * **Source:** [Hugging Face `gpt2`](https://www.google.com/search?q=%5Bhttps://huggingface.co/gpt2%5D\(https://huggingface.co/gpt2\))
  * **Parameters:** \~124 Million
  * **Architecture:** 12-layer, 768-hidden, 12-heads.
  * **Role:** The **Policy Model** is optimized directly via the DPO objective. The **Reference Model** (an implicit copy created by `DPOTrainer`) provides the baseline log-probabilities to ensure the model doesn't deviate too far from its original linguistic capabilities.


## Dataset: Human-Like DPO Dataset

The project utilizes preference pairs to guide the model toward human-like conversational styles.

  * **Source:** [`HumanLLMs/Human-Like-DPO-Dataset`](https://www.google.com/search?q=%5Bhttps://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset%5D\(https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset\))
  * **Subsetting:** \* **Train:** 2,048 samples (sub-sampled for efficiency).
      * **Test:** 512 samples.
  * **Structure:**
      * `prompt`: The context or question.
      * `chosen`: The preferred human-like response.
      * `rejected`: The less desirable or robotic response.


## Setup & Installation

```bash
# Create Environment
conda create -n rlhf_dpo python=3.11 -y
conda activate rlhf_dpo

# Install Dependencies
pip install transformers==4.45.2 trl==0.11.3 datasets accelerate evaluate 
pip install rouge_score bert_score sacrebleu matplotlib pandas scikit-learn
```


## DPO Training Pipeline

### . Direct Preference Optimization (DPO)

Instead of training a reward model, DPO uses the analytical relationship between the reward function and the optimal policy to optimize the model using a simple binary cross-entropy loss.

  * **Objective:** $L_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} [\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$
  * **Key Config:**
      * **Beta:** 0.005 (Controls the strength of the KL penalty).
      * **Optimizer:** AdamW with `learning_rate=5e-6`.
      * **Max Steps:** 300 (Overriding epochs for rapid alignment).

### . Preference Metrics

The trainer tracks how well the model distinguishes between "chosen" and "rejected" answers:

  * **Reward Margin:** The difference in log-likelihoods between chosen and rejected responses.
  * **Accuracy:** The frequency with which the model assigns a higher probability to the "chosen" response than the "rejected" one.


## Monitoring & Output

Results are stored in `runs/dpo_gpt2/`:

  * **Checkpoints:** Saved iterations of the aligned GPT-2 model.
  * **Plots:** \* `loss_curve.png`: Shows the DPO loss convergence.
      * `metrics_curve.png`: Visualizes **Reward Margin** and **Accuracy** over steps.
  * **Evaluation:** `generations_comparison.csv` provides a side-by-side look at Human targets vs. Model outputs.

### nal Generative Results

| Metric | Score |
| :--- | :--- |
| **BLEU** | 0.0014 |
| **ROUGE-L** | 0.0480 |
| **BERT\_F1** | 0.7892 |


## Known Limitations

  * **Proxy Reward:** The model is trained on a relatively small subset (2k samples).
  * **Metric Spikes:** With a very small $\beta$ (0.005), the reward margin can grow significantly; if the model collapses, it may generate repetitive phrases.
  * **BERTScore Stability:** Includes a "Critical Fix" to handle empty string generations, which can otherwise crash the `evaluate` library.