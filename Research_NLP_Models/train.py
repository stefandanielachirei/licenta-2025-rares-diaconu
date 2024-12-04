import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import evaluate

# 2. Set device to use a single GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 3. Load model and tokenizer with 4-bit quantization
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    } 
).to(device)

# 4. Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

train_subset_size = 50000
dataset["train"] = dataset["train"].select(range(train_subset_size))

# 5. Preprocessing
def preprocess(example):
    inputs = tokenizer(
        example["article"], max_length=1024, truncation=True, padding="max_length", return_tensors="pt"
    )
    labels = tokenizer(
        example["highlights"], max_length=1024, truncation=True, padding="max_length", return_tensors="pt"
    )
    inputs["labels"] = labels["input_ids"]
    return {k: v.squeeze() for k, v in inputs.items()}

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# 6. DataLoader and DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=4, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=4, collate_fn=data_collator)

# 7. Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# 8. Metric for evaluation
rouge = evaluate.load("rouge")

# 9. Early Stopping Configuration
patience = 3  # Number of epochs to wait before stopping
best_val_loss = float('inf')
early_stop_counter = 0

# 10. Training Loop
num_epochs = 10 
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()
    train_loss = 0
    train_progress = tqdm(train_dataloader, desc="Training", leave=False)
    for batch in train_progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        train_progress.set_postfix({"loss": loss.item()})
    
    train_loss /= len(train_dataloader)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")

    model.eval()
    val_loss = 0
    all_predictions = []
    all_references = []
    val_progress = tqdm(eval_dataloader, desc="Validation", leave=False)
    for batch in val_progress:
        batch = {k: v.to(device) for k, v in batch.items()}
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

            val_progress.set_postfix({"val_loss": outputs.loss.item()})
    
    val_loss /= len(eval_dataloader)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

    # Compute ROUGE score
    rouge_score = rouge.compute(predictions=all_predictions, references=all_references)
    print(f"Epoch {epoch + 1}, ROUGE Score: {rouge_score}")

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        
        model.save_pretrained("./best_model")
        tokenizer.save_pretrained("./best_model")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"No improvement for {patience} epochs. Stopping early.")
            break

# 11. Final save
output_dir = "./llama_final_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
