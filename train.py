from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import pandas as pd
import torch


# Read the source and target files
with open('data/source.txt', 'r', encoding='utf-8') as f:
    source_sentences = f.readlines()

with open('data/target.txt', 'r', encoding='utf-8') as f:
    target_sentences = f.readlines()

# Ensure both files have the same number of lines
assert len(source_sentences) == len(target_sentences), "Source and target files must have the same number of lines."

# Create a DataFrame
df = pd.DataFrame({
    'source': [line.strip() for line in source_sentences],
    'target': [line.strip() for line in target_sentences]
})

# Save to CSV
df.to_csv('data/data.csv', index=False)



dataset = load_dataset('csv', data_files={'train': 'data/data.csv'})

model_name = 'jbochi/madlad400-3b-mt'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)

source_lang = "en"
target_lang = "sw"
task_prefix = f"<2{target_lang}>"

def preprocess_function(examples):
    inputs = [task_prefix + src for src in examples['source']]
    targets = examples['target']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

split_dataset = tokenized_dataset['train'].train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

training_args = TrainingArguments(
    output_dir="./madlad400-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    # predict_with_generate=True,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,  # Use mixed precision if supported
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

bleu = load_metric("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # BLEU expects a list of references for each prediction
    references = [[label] for label in decoded_labels]
    result = bleu.compute(predictions=decoded_preds, references=references)
    return {"bleu": result["bleu"]}

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

