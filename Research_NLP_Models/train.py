import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import Qwen2Tokenizer, AutoModelForCausalLM

torch.cuda.set_device(0)  # GPU 2 now maps to index 0 due to CUDA_VISIBLE_DEVICES

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.gradient_checkpointing_enable()

# Load and preprocess dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
def preprocess(example):
    inputs = tokenizer(example["article"], max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(example["highlights"], max_length=1024, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True).with_format("torch")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True,
    dataloader_pin_memory=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
