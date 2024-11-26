from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os

def train_and_save_model():
    model_name = "distilgpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Setăm un `pad_token` pentru tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    train_data_path = "data.txt"
    if os.path.getsize(train_data_path) == 0:
        raise ValueError("Training file is empty!")

    # Încarcă și preprocesează dataset-ul
    dataset = load_dataset("text", data_files=train_data_path)["train"]

    # Tokenizează dataset-ul
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    trainer.save_model("./model")
    tokenizer.save_pretrained("./model")

if __name__ == "__main__":
    # Creează fișierul de date
    with open("data.txt", "w") as f:
        f.write("Hello world. This is a test dataset. Fine-tune GPT-2 for text generation.\n")
        f.write("Another example line to ensure enough data for training.\n")

    train_and_save_model()
