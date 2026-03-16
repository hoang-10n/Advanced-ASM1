# %%
!pip install transformers==4.36.2
!pip install datasets==2.15.0
!pip install peft==0.7.1
!pip install bitsandbytes==0.41.3
!pip install accelerate==0.25.0
!pip install trl==0.7.7
!pip install tqdm==4.66.1

# %% 
class CFG:
    is_training = False

# %%
import torch
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, Markdown
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

warnings.filterwarnings("ignore")
tqdm.pandas()

# %%
config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5,
)

sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 16}

# %%
def build_dataset(
    config,
    dataset_name="stanfordnlp/imdb",
    input_min_text_length=2,
    input_max_text_length=8,
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

# %%
dataset = build_dataset(config)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

# %%
if CFG.is_training:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
else:
    model_path = "/kaggle/input/pt2-imdb-pos-v2/gpt2-imdb-pos-v2"
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

# %%
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

# %%
ppo_trainer = PPOTrainer(
    config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
)

# %%
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

sentiment_pipe = pipeline(
    "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
)

# %%
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

# %%
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = gen_kwargs.copy()

if CFG.is_training:
    for epoch, batch in enumerate(tqdm(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            query_response = ppo_trainer.generate(query, **generation_kwargs).squeeze()
            response_len = len(query_response) - len(query)
            response_tensors.append(query_response[-response_len:])

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)

        positive_scores = [
            item["score"]
            for output in pipe_outputs
            for item in output
            if item["label"] == "POSITIVE"
        ]
        rewards = [torch.tensor(score) for score in positive_scores]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

# %%
if CFG.is_training:
    model.save_pretrained("gpt2-imdb-pos-v2")
    tokenizer.save_pretrained("gpt2-imdb-pos-v2")

# %%
bs = 16
game_data = {}

dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)

game_data["query"] = df_batch["query"].tolist()
query_tensors = df_batch["input_ids"].tolist()

response_tensors_ref, response_tensors = [], []

for i in range(bs):
    query = torch.tensor(query_tensors[i]).to(device)

    gen_len = output_length_sampler()
    query_response = ref_model.generate(
        query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()

    response_len = len(query_response) - len(query)
    response_tensors_ref.append(query_response[-response_len:])

    query_response = model.generate(
        query.unsqueeze(0), max_new_tokens=gen_len, **gen_kwargs
    ).squeeze()

    response_len = len(query_response) - len(query)
    response_tensors.append(query_response[-response_len:])

game_data["response (before)"] = [
    tokenizer.decode(response_tensors_ref[i]) for i in range(bs)
]
game_data["response (after)"] = [
    tokenizer.decode(response_tensors[i]) for i in range(bs)
]

texts = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
game_data["rewards (before)"] = [
    item["score"]
    for output in pipe_outputs
    for item in output
    if item["label"] == "POSITIVE"
]

texts = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
game_data["rewards (after)"] = [
    item["score"]
    for output in pipe_outputs
    for item in output
    if item["label"] == "POSITIVE"
]

df_results = pd.DataFrame(game_data)

# %%
def calculate_mean_rewards(df):
    mean_before = df["rewards (before)"].mean()
    mean_after = df["rewards (after)"].mean()
    return mean_before, mean_after, mean_after - mean_before

def display_mean_rewards(mean_before, mean_after, gain):
    display(
        Markdown(
            f"## Average Reward Improvement\n\n"
            f"Gain: {gain:.4f}\n\n"
            f"Before: {mean_before:.4f}\n"
            f"After: {mean_after:.4f}"
        )
    )

def plot_mean_rewards(mean_before, mean_after, gain):
    labels = ["Before", "After", "Gain"]
    values = [mean_before, mean_after, gain]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title("Mean Reward Comparison")
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")
    plt.show()

mean_before, mean_after, gain = calculate_mean_rewards(df_results)
display_mean_rewards(mean_before, mean_after, gain)
plot_mean_rewards(mean_before, mean_after, gain)
