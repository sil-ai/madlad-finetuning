import os
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import numpy as np
import evaluate
from clearml import Dataset as ClearMLDataset
from clearml.config import config_obj 
from dotenv import load_dotenv
import random

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, help="Path to the source file")
parser.add_argument("--target", type=Path, help="Path to the target file")
parser.add_argument("--source-lang", type=str, help="Source language ISO code")
parser.add_argument("--target-lang", type=str, help="Target language ISO code")
parser.add_argument("--HF-TOKEN", type=str, help="Hugging Face API token")
parser.add_argument("--dataset-id", type=str, help="ClearML dataset ID")
parser.add_argument("--lora_r", type=int, help="LoRA r value", default=8)
parser.add_argument("--lora-alpha", type=int, help="LoRA alpha value", default=32)
parser.add_argument("--lora-dropout", type=float, help="LoRA dropout value", default=0.05)

args = parser.parse_args()

dataset = ClearMLDataset.get(dataset_id=args.dataset_id)
base_path = dataset.get_local_copy()

source_lang = args.source_lang if args.source_lang else args.source.stem.split("-")[0]
target_lang = args.target_lang if args.target_lang else args.target.stem.split("-")[0]

source_file = f"{base_path}/{args.source.stem}.txt"
target_file = f"{base_path}/{args.target.stem}.txt"


# Read the source and target files
with open(source_file, "r", encoding="utf-8") as f:
    source_sentences = f.readlines()

with open(target_file, "r", encoding="utf-8") as f:
    target_sentences = f.readlines()

# Ensure both files have the same number of lines
assert len(source_sentences) == len(
    target_sentences
), "Source and target files must have the same number of lines."

# Create a DataFrame
df = pd.DataFrame(
    {
        "source": [line.strip() for line in source_sentences],
        "target": [line.strip() for line in target_sentences],
    }
)

# Remove rows with empty source or target
df = df[(df["source"] != "") & (df["target"] != "")]

# Find indices where either source or target is "<range>"
to_drop = df[(df['source'] == '<range>') | (df['target'] == '<range>')].index

# Include the row above for each match (if it exists)
to_drop = to_drop.union(to_drop - 1)

print(f"Drop {len(to_drop)} rows.")

# Drop the rows
df = df.drop(to_drop).reset_index(drop=True)

# Shuffle df
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split index for 90% training data
split_idx = int(0.9 * len(df))

# Split df into training and evaluation sets
train_df = df.iloc[:split_idx].reset_index(drop=True)
eval_df = df.iloc[split_idx:].reset_index(drop=True)

# Load the Word Correspondences dataset
wc_df = pd.read_csv(f"{base_path}/en-NASB-nih-NIH_top_source_scores_filtered.csv")

# Append wc_df rows only to the training dataframe
# train_df = pd.concat([train_df, wc_df[['source', 'target']]], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Convert dataframes to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# model_name = "jbochi/madlad400-3b-mt"
model_name = "facebook/nllb-200-3.3B"

tokenizer = NllbTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, max_length=256)

def text_is_in_vocab(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    return len(tokens) == 2 and tokens[1] == text

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    modules_to_save=["embed_tokens", "lm_head"],
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Unfreeze additional parameters
for name, param in model.named_parameters():
    print(name)
    if "layer_norm" in name:
        param.requires_grad = True

model.print_trainable_parameters()

chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    # top_predictions = np.argmax(predictions[0], axis=-1)
    
    # Decode the generated predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as they are used to mask tokens during loss computation
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode the labels
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute CHRF score
    result = chrf.compute(predictions=decoded_preds, references=decoded_labels)

    # Print first ten sample sentences
    sample_size = min(10, len(decoded_preds))
    
    print("\n----- First 10 Sample Translations -----")
    for idx in range(sample_size):
        print(f"Source:      {tokenized_eval_dataset[idx]['source']}")
        print(f"Target:      {decoded_labels[idx]}")
        print(f"Prediction:  {decoded_preds[idx]}")
        print("---")
    
    return {"chrf": result["score"]}


def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=256,
        truncation=True,
        padding='longest',
    )

    return model_inputs


tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
print(f'{tokenized_train_dataset[0]=}')

print(f'{tokenized_train_dataset=}')
print(f'{tokenized_eval_dataset=}')
HF_TOKEN = args.HF_TOKEN

training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-finetuned-lora",
    evaluation_strategy="steps",
    eval_steps=400,
    save_strategy="steps",
    save_steps=400,
    learning_rate=5e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    weight_decay=1e-5,
    warmup_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=256,
    metric_for_best_model='chrf',
    greater_is_better=True,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
    fp16=False,
    gradient_accumulation_steps=8,
    push_to_hub=True,
    push_to_hub_model_id=f"nllb-finetuned-{source_lang}-{target_lang}",
    push_to_hub_organization="sil-ai",
    push_to_hub_token=HF_TOKEN,
    hub_private_repo=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
# trainer.save_model(f"./madlad400-finetuned-{source_lang}-{target_lang}")
# trainer.evaluate()
# trainer.push_to_hub(f"sil-ai/madlad400-finetuned-{source_lang}-{target_lang}", private=True, token=HF_TOKEN)
