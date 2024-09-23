from datasets import load_dataset, Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import pandas as pd
import torch
import evaluate
from clearml import Task

task = Task.init(
    project_name="IDX Translation Fine-tuning/IDX MADLAD Exp",
    task_name="madlad-finetuning",
)

# Read the source and target files
with open("data/source.txt", "r", encoding="utf-8") as f:
    source_sentences = f.readlines()

with open("data/target.txt", "r", encoding="utf-8") as f:
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

# Save to CSV
df.to_csv("data/data.csv", index=False)


# dataset = load_dataset('csv', data_files={'train': 'data/data.csv'}, download_mode='force_redownload')
dataset = Dataset.from_pandas(df, split="train")


model_name = "jbochi/madlad400-3b-mt"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
)

# Set number of beams to 1 to avoid multiple predictions per input
model.config.num_beams = 1

chrf = evaluate.load("chrf")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print("Type of predictions:", type(predictions))
    if hasattr(predictions, "shape"):
        print("Shape of predictions:", predictions.shape)
    print("Sample predictions:", predictions[:1])

    # # Check if predictions contain multiple beams
    # if isinstance(predictions, list) and isinstance(predictions[0], list) and isinstance(predictions[0][0], list):
    #     # If multiple beams, select the first beam
    #     predictions = [pred[0] for pred in predictions]
    # elif isinstance(predictions, torch.Tensor):
    #     # If predictions are tensors, convert to list
    #     predictions = predictions.tolist()

    # # Ensure predictions are a list of lists
    # if isinstance(predictions, list) and all(isinstance(p, list) for p in predictions):
    #     pass
    # else:
    #     print("Unexpected predictions format:", type(predictions))
    #     return {"chrf": float('nan')}

    # # # Check if predictions contain multiple beams
    # # if isinstance(predictions[0], list) and isinstance(predictions[0][0], list):
    # #     # If multiple beams, select the first beam
    # #     predictions = [pred[0] for pred in predictions]

    # # Decode the predictions
    # decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # # Replace -100 in the labels as they are used to mask tokens during loss computation
    # labels = [
    #     [(label if label != -100 else tokenizer.pad_token_id) for label in label_seq]
    #     for label_seq in labels
    # ]
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # # Compute CHRF
    # result = chrf.compute(predictions=decoded_preds, references=decoded_labels)

    return {"chrf": 0}  # result["score"]}


source_lang = "en"
target_lang = "sw"
task_prefix = f"<2{target_lang}>"


def preprocess_function(examples):
    inputs = [task_prefix + src for src in examples["source"]]
    targets = examples["target"]
    model_inputs = tokenizer(
        inputs,
        max_length=256,  # Reduced max_length for memory efficiency
        truncation=True,
        padding=False,
    )

    # Tokenize targets using text_target
    labels = tokenizer(
        text_target=targets, max_length=256, truncation=True, padding=False
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

training_args = Seq2SeqTrainingArguments(
    output_dir="./madlad400-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,  # Adjust based on GPU memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,  # Use mixed precision if supported
)


data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding=True, return_tensors="pt"
)

# bleu = load_metric("bleu")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     # BLEU expects a list of references for each prediction
#     references = [[label] for label in decoded_labels]
#     result = bleu.compute(predictions=decoded_preds, references=references)
#     return {"bleu": result["bleu"]}

trainer = Trainer(
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
