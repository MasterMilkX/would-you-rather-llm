# %% [markdown]
# # Finetune ANY LLM MODEL to Generate Would You Rather Questions
# Fine-tunes any LLM model on a dataset of WYR questions so it can generate new ones in the same vibe.
# 

# %%
import random
import json
from tqdm import tqdm

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

import json
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import torch

from datetime import datetime
import os

# %%

# Check if CUDA is available
print(f"Is CUDA available? {torch.cuda.is_available()}")

# Get the name of the GPU
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")


# %%
from huggingface_hub import login
login()

# %%
# Load model directly

# list of models: google/gemma-3-1b-it, HuggingFaceTB/SmolLM2-1.7B-Instruct, TinyLlama/TinyLlama-1.1B-Chat-v1.0, Qwen/Qwen2.5-1.5B

MODEL_SET = {
    "gemma": "google/gemma-3-1b-it",
    "smol": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "llama": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "big-llama": "meta-llama/Llama-3.2-1B",
    "deepseek": "deepseek-ai/DeepSeek-V4-Pro",
    "qwen": "Qwen/Qwen2.5-1.5B",
}

EPOCHS = 5
TRAIN_UPVOTE = True
UPVOTE_Q = "finetuning-notebooks/upvote_questions.json"


import gc
import torch

def cleanup_gpu(*objs):
    """
    Delete large Python/PyTorch objects and clear CUDA cache.
    Pass in model, trainer, tokenizer, etc.
    """
    for obj in objs:
        try:
            del obj
        except Exception:
            pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    print("GPU cleanup complete.")

def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[GPU {label}] allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


def build_wyr_dataset(polls_path: str) -> Dataset:
    """
    Format each WYR poll as a plain generation string.
    The model learns to produce well-formed WYR questions by
    predicting the full text token-by-token.

    Two formats are included so the model learns both the
    full question and the two-option layout:

    Format A (plain from the title):
        Would you rather [optionA] or [optionB]?

    Format B (structured):
        Would you rather...
        A) [optionA]
        B) [optionB]
    """
    with open(polls_path, 'r') as f:
        polls = json.load(f)

    rows = []
    for poll in polls:
        ctx = poll['title'].strip().rstrip('.')
        a = poll['optionA'].strip().rstrip('.')
        b = poll['optionB'].strip().rstrip('.')

        # Randomly swap A/B so the model doesn't develop order bias
        if random.random() < 0.5:
            a, b = b, a
        
        # Format A — plain from the title
        rows.append({'text': f"{ctx}\nWould you rather...\nA) {a}\nB) {b}?"})

        # Format B — structured two-option layout
        rows.append({'text': f"Would you rather...\nA) {a}\nB) {b}"})

    random.shuffle(rows)

    # Save for inspection
    with open(JSONL_OUTPUT, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')

    print(f"Built {len(rows)} training examples ({len(polls)} polls x 2 formats) -> {JSONL_OUTPUT}")
    return Dataset.from_list(rows)


# %%

date = datetime.now().strftime("%Y-%m-%d")

POLLS_JSON    = "formatted_polls.json"   # <-- put your file here
JSONL_OUTPUT  = f"train-data/wyr_training_data-[{date}].jsonl"

os.makedirs("train-data", exist_ok=True)

train_dataset = build_wyr_dataset(POLLS_JSON)

if TRAIN_UPVOTE and os.path.exists(UPVOTE_Q):
    print(f"\nTRAIN_UPVOTE=True — appending upvoted questions from {UPVOTE_Q}")
    upvote_dataset = build_wyr_dataset(UPVOTE_Q)
    train_dataset = concatenate_datasets([train_dataset, upvote_dataset]).shuffle(seed=42)
    print(f"Combined dataset size: {len(train_dataset)}")
elif TRAIN_UPVOTE:
    print(f"TRAIN_UPVOTE=True but {UPVOTE_Q} not found — skipping")

print(train_dataset)
print("\nSample entries:")
for i in range(4):
    print(f"\n--- {i} ---")
    print(train_dataset[i]['text'])

for MODEL_NAME, BASE_MODEL_ID in MODEL_SET.items():
    print(f"======   {MODEL_NAME}: {BASE_MODEL_ID}  =======")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype="auto"
    )

    # %%
    # set padding token
    if MODEL_NAME != "gemma":
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    else:
        tokenizer.eos_token_id = 106
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(106)
        model.config.eos_token_id = 106
        model.config.eos_token = tokenizer.eos_token


    # %%
    # Output directories
    LORA_OUTPUT_DIR   = f"../model_outputs/{MODEL_NAME}-wyr-lora"
    MERGED_OUTPUT_DIR = f"../model_outputs/{MODEL_NAME}-wyr-merged"

    MAX_SEQ_LEN = 128  # WYR questions are short


    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=LORA_OUTPUT_DIR,
        num_train_epochs=EPOCHS,          # small dataset — multiple passes help
        dataloader_num_workers=4,
        # dataloader_pin_memory=4,
        # per_device_train_batch_size=4,
        # gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        dataloader_pin_memory=False,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=True,
        bf16=(
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 8
        ),
        fp16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()


    # %%
    # 7) Save LoRA adapter + tokenizer locally
    os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)
    trainer.model.save_pretrained(LORA_OUTPUT_DIR)
    tokenizer.save_pretrained(LORA_OUTPUT_DIR)
    print(f"Saved LoRA adapter to {LORA_OUTPUT_DIR}")
    cleanup_gpu(trainer, model)

    # 8) Merge LoRA weights into base model and save a standalone model
    print("Merging LoRA adapter into base model...")

    # Reload base model on CPU (or cuda if you want)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cpu",   # merge on CPU to avoid GPU OOM
    )

    lora_model = PeftModel.from_pretrained(base_model, LORA_OUTPUT_DIR)
    merged_model = lora_model.merge_and_unload()  # apply LoRA weights into base

    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    print(f"Saved merged full model to {MERGED_OUTPUT_DIR}")
    cleanup_gpu(base_model, lora_model, merged_model, tokenizer)

    # %% [markdown]
    # ## Generate new Would You Rather questions
    # 

    # %%
    # Quick generation test — prime the model with the WYR opener
    # and let it complete the question.

    model.eval()

    prompts = [
        "Would you rather...",
        #"Would you rather...",
    ]
    prompts *= 10

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"PROMPT: {prompt!r}")
        print(f"OUTPUT: {generated}")
        print()


print("All done!")

