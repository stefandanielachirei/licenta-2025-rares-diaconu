import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import evaluate

# 1. Load model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# 3. Preprocessing
def preprocess(example):
    inputs = tokenizer(
        example["article"], max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )
    labels = tokenizer(
        example["highlights"], max_length=150, truncation=True, padding="max_length", return_tensors="pt"
    )
    inputs["labels"] = labels["input_ids"]
    return {k: v.squeeze() for k, v in inputs.items()}

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# 4. DataLoader and DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=4, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=4, collate_fn=data_collator)

# 5. Initialize Accelerator
accelerator = Accelerator()
model, train_dataloader, eval_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader)

# 6. Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# 7. Early Stopping Configuration
patience = 3  # Number of epochs to wait before stopping
best_val_loss = float('inf')
early_stop_counter = 0

# Metric for evaluation
rouge = evaluate.load("rouge")

# 8. Training Loop
num_epochs = 10  # Maximum number of epochs
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()
    
    # Average training loss for the epoch
    train_loss /= len(train_dataloader)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

    # Validation
    model.eval()
    val_loss = 0
    all_predictions = []
    all_references = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            # Generate summaries and collect for metric evaluation
            generated = model.generate(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Average validation loss for the epoch
    val_loss /= len(eval_dataloader)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

    # Compute ROUGE score
    rouge_score = rouge.compute(predictions=all_predictions, references=all_references)
    print(f"Epoch {epoch + 1}, ROUGE Score: {rouge_score}")

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        # Save the best model
        model.save_pretrained("./best_model")
        tokenizer.save_pretrained("./best_model")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"No improvement for {patience} epochs. Stopping early.")
            break

# Final save
output_dir = "./t5_final_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
