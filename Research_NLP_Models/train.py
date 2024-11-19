import tensorflow as tf
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import Qwen2Tokenizer
from transformers import AutoModelForCausalLM

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[2], True)
        print("TensorFlow configurat pentru a folosi GPU 2")
    except RuntimeError as e:
        print("Eroare la configurarea TensorFlow pentru GPU:", e)


if torch.cuda.is_available():
    device = torch.device("cuda:2")
    print("Torch configurat pentru a folosi GPU 2")
else:
    device = torch.device("cpu")
    print("Torch configurat să folosească CPU")

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and preprocess dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
def preprocess(example):
    inputs = tokenizer(example["article"], max_length=32768, truncation=True, padding="max_length")
    labels = tokenizer(example["highlights"], max_length=1024, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True,
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

