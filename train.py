import json
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
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
parser.add_argument("--source-2", type=Path, help="Path to the second source file")
parser.add_argument("--source-lang", type=str, help="Source language ISO code")
parser.add_argument("--target-lang", type=str, help="Target language ISO code")
parser.add_argument("--HF-TOKEN", type=str, help="Hugging Face API token")
parser.add_argument("--dataset-id", type=str, help="ClearML dataset ID")
parser.add_argument("--lora_r", type=int, help="LoRA r value", default=8)
parser.add_argument("--lora-alpha", type=int, help="LoRA alpha value", default=32)
parser.add_argument("--lora-dropout", type=float, help="LoRA dropout value", default=0.05)
parser.add_argument("--data-aug", action="store_true", help="Augment with Word Correspondences dataset")
parser.add_argument("--tokenize", type=str, choices=["source", "target", "both"], help="Tokenize the dataset")
parser.add_argument("--num-tokens", type=int, help="Number of tokens to add to the tokenizer", default=1000)
parser.add_argument("--rslora", action="store_true", help="Use RSLora")


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

if args.source_2:
    source_file_2 = f"{base_path}/{args.source_2.stem}.txt"
    with open(source_file_2, "r", encoding="utf-8") as f:
        source_sentences_2 = f.readlines()

with open(f'{base_path}/vref.txt', 'r', encoding='utf-8') as f:
    vrefs = f.readlines()

# Ensure both files have the same number of lines
assert len(source_sentences) == len(
    target_sentences
), "Source and target files must have the same number of lines."

# Create a DataFrame
df = pd.DataFrame(
    {
        "source": [line.strip() for line in source_sentences],
        "target": [line.strip() for line in target_sentences],
        "index": [vref.strip() for vref in vrefs],
    }
)

if args.source_2:
    df['source_2'] = [line.strip() for line in source_sentences_2]

    df = pd.DataFrame(
        {
            "source": df["source"].tolist() + df["source_2"].tolist(),
            "target": df["target"].tolist() * 2,
            "index": df['index'].tolist() * 2,
        }
    )

# Find indices where either source or target is "<range>"
to_drop = df[(df['source'] == '<range>') | (df['target'] == '<range>')].index

# Include the row above for each match (if it exists)
to_drop = to_drop.union(to_drop - 1)

print(f"Drop {len(to_drop)} rows.")
df = df.drop(to_drop)

# Remove rows with empty source or target
df = df[(df["source"] != "") & (df["target"] != "") & (df["source"] != '...') & (df["target"] != '...')]

# Shuffle the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Drop the rows
unique_indices = df["index"].unique()
random.Random(42).shuffle(unique_indices)  # Shuffle the indices for splitting

# Split by index: 90% train, 10% eval
split_idx = int(0.9 * len(unique_indices))
train_indices = unique_indices[:split_idx]
eval_indices = unique_indices[split_idx:]

# Use these indices to filter the DataFrame into train and eval sets, then shuffle
train_df = df[df["index"].isin(train_indices)].reset_index(drop=True)
eval_df = df[df["index"].isin(eval_indices)].reset_index(drop=True)

print(f'{train_df.sort_values("index").head(50)=}')
print(f'{eval_df.sort_values("index").head(50)=}')

# Load the Word Correspondences dataset
wc_df = pd.read_csv(f"{base_path}/en-NASB-nih-NIH_top_source_scores_filtered.csv")

if args.data_aug:
    # Append wc_df rows only to the training dataframe
    train_df = pd.concat([train_df, wc_df[['source', 'target']]], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Convert dataframes to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

model_name = "jbochi/madlad400-3b-mt"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, max_length=128)

if args.tokenize:
    tokenization_train_dataset = []
    if args.tokenize in ["source", "both"]:
        tokenization_train_dataset.append(source_file)
    if args.tokenize in ["target", "both"]:
        tokenization_train_dataset.append(target_file)
    
    bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=args.num_tokens)
    bpe_tokenizer.pre_tokenizer = Whitespace()
    bpe_tokenizer.train(files=tokenization_train_dataset, trainer=trainer)
    bpe_tokenizer.save(f"{base_path}/bpe_tokenizer.json")

    with open(f"{base_path}/bpe_tokenizer.json", "r", encoding="utf-8") as f:
        bpe_tokenizer_data = json.load(f)
    
    new_tokens = bpe_tokenizer_data["model"]["vocab"].keys()
    existing_tokens = tokenizer.get_vocab().keys()
    unique_new_tokens = [token for token in new_tokens if token not in existing_tokens]

    tokenizer.add_tokens(unique_new_tokens)


def text_is_in_vocab(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    return len(tokens) == 2 and tokens[1] == text

madlad_language_codes = {
    'eng': 'en',
    'swh': 'sw',
}

language_token = f"<2{madlad_language_codes.get(target_lang, target_lang)}>"

if not text_is_in_vocab(tokenizer, language_token):
    print(f"Adding {language_token} to the vocabulary.")
    tokenizer.add_tokens([language_token])

model.resize_token_embeddings(len(tokenizer))

# Sample tokenization from first training example
print(tokenizer(train_dataset[0]["source"]))
print(tokenizer(train_dataset[0]["target"]))

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    target_modules=["q", "v", "k", "o", "wi_0", "wi_1", "wo"],
    modules_to_save=["embed_tokens", "lm_head"],
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias="none",
    use_rslora=args.rslora,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Unfreeze additional parameters
for name, param in model.named_parameters():
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
    inputs = [f"{language_token} {src}" for src in examples["source"]]    
    targets = examples["target"]
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=128,
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
    output_dir="./madlad400-finetuned-lora",
    evaluation_strategy="steps",
    eval_steps=400,
    save_strategy="steps",
    save_steps=1600,
    learning_rate=5e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    num_train_epochs=15,
    weight_decay=1e-5,
    warmup_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=128,
    metric_for_best_model='chrf',
    greater_is_better=True,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
    fp16=False,
    gradient_accumulation_steps=8,
    push_to_hub=True,
    push_to_hub_model_id=f"madlad400-finetuned-{source_lang}-{target_lang}",
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
