from datasets import load_dataset
from transformers import AutoTokenizer
import os
from huggingface_hub import login

login("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")
# paths
data_path = "/home/work/datasets/fineweb-100B"
save_path = "/home/work/datasets/fineweb_llama_block_size"
num_available_cpus = len(os.sched_getaffinity(0))
block_size=2048
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.model_max_length = block_size

# load parquet dataset
dataset = load_dataset("parquet", data_files={"train": os.path.join(data_path, "*.parquet")}, split="train")

# tokenize function
def tokenize_function(examples):
    return tokenizer(examples["text"])

# tokenize in batched mode
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=num_available_cpus,      
    remove_columns=dataset.column_names,
    desc="Tokenizing with LLaMA tokenizer"
)

# group into fixed-length block_size chunks
def group_texts(examples):
    # concatenate all tokens
    concatenated = sum(examples["input_ids"], [])
    # drop remainder
    total_len = (len(concatenated) // block_size) * block_size
    concatenated = concatenated[:total_len]
    # split into chunks of block_size
    result = {
        "input_ids": [concatenated[i:i+block_size] for i in range(0, total_len, block_size)],
    }
    result["attention_mask"] = [[1]*block_size] * len(result["input_ids"])
    return result

lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    num_proc=num_available_cpus,
    desc=f"Grouping into {block_size}-token blocks"
)

# save to disk (huggingface arrow format)
lm_dataset.save_to_disk(save_path)
print(f"✅ Saved tokenized dataset at {save_path}")