import os
from datasets import load_dataset, Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import numpy as np
import evaluate
from clearml import Dataset as ClearMLDataset
from dotenv import load_dotenv

load_dotenv()

dataset = ClearMLDataset.get(dataset_id="85c436bb386847e29fe72e8449814b11")
base_path = dataset.get_local_copy()
source_file = f"{base_path}/en-NASB.txt"
target_file = f"{base_path}/nih-NIH.txt"


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

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv("data/data.csv", index=False)

# Load dataset
dataset = Dataset.from_pandas(df, split="train")

model_name = "jbochi/madlad400-3b-mt"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    # inference_mode=False,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

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
    
    return {"chrf": result["score"]}

source_lang = "en"
target_lang = "sw"
task_prefix = f"<2{target_lang}>"

def preprocess_function(examples):
    inputs = [task_prefix + src for src in examples["source"]]
    targets = examples["target"]
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=256,
        truncation=True,
        padding='longest',
    )
    # Tokenize targets using text_target
    # labels = tokenizer(
    #     text_target=targets, max_length=256, truncation=True, padding="max_length"
    # )
    # # Assign labels for loss computation
    # model_inputs["labels"] = labels["input_ids"]

    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

print(f'{tokenized_dataset[0]=}')

split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f'{train_dataset=}')
print(f'{eval_dataset=}')

training_args = Seq2SeqTrainingArguments(
    output_dir="./madlad400-finetuned-lora",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-3,  # Adjusted learning rate
    warmup_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    metric_for_best_model='chrf',
    greater_is_better=True,
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=10,
    fp16=False,
    gradient_accumulation_steps=2,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./madlad400-finetuned")
trainer.evaluate()
trainer.push_to_hub("sil-ai/madlad400-finetuned-engNASB-swhONEN")
