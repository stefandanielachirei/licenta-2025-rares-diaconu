import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import evaluate
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.tensorboard import SummaryWriter

# 2. Set device to use a single GPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# 3. Load model and tokenizer with 4-bit quantization
model_name = "meta-llama/Llama-3.2-1B"
model_dir = "./best_model_4_training"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

writer = SummaryWriter(log_dir="./tensorboard_logs_6_training")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"

base_model.resize_token_embeddings(len(tokenizer))

adapter_model = PeftModel.from_pretrained(base_model, model_dir)

print(f"Dropout rate in the model: {adapter_model.config.hidden_dropout_prob if hasattr(adapter_model.config,'hidden_dropout_prob') else 'Not specified, likely default.'}")

lora_config = LoraConfig(
    r=16,  # Dimensiunea rank-ului (valoare tipică: 8-16)
    lora_alpha=32,  # Factor de scalare
    target_modules=["q_proj", "v_proj"],  # Modulele vizate (de exemplu, proiecțiile atenției)
    lora_dropout=0.3,  # Dropout pentru regularizare
    bias="none"  # Nu adaptăm biasele
)

model = get_peft_model(adapter_model, lora_config)
model.print_trainable_parameters()

model = model.to(device)

# 4. Load dataset
#dataset = load_dataset("cnn_dailymail", "3.0.0")

#start_idx = 0
#end_idx = 100000
#dataset["train"] = dataset["train"].select(range(start_idx, end_idx))
dataset = load_dataset("multi_news", trust_remote_code=True)

# 5. Preprocessing
def preprocess(example):
    inputs = tokenizer(
        example["document"], max_length=256, truncation=True, padding="max_length", return_tensors="pt"
    )
    labels = tokenizer(
        example["summary"], max_length=256, truncation=True, padding="max_length", return_tensors="pt"
    )
    inputs["labels"] = labels["input_ids"]
    if torch.all(inputs["input_ids"] == tokenizer.pad_token_id) or torch.all(labels["input_ids"] == tokenizer.pad_token_id):
        print(f"Skipped invalid example: {example}")
        return None
    return {k: v.squeeze() for k, v in inputs.items()}

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# 6. DataLoader and DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=8, collate_fn=data_collator)

# 7. Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)

# 8. Metric for evaluation
rouge = evaluate.load("rouge")


# 9. Early Stopping Configuration
patience = 3  # Number of epochs to wait before stopping
best_val_loss = float('inf')
early_stop_counter = 0
gradient_accumulation_steps = 4

# 10. Training Loop
num_epochs = 20 

total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps,  # 10% warmup steps
    num_training_steps=total_steps
)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()
    train_loss = 0
    optimizer.zero_grad()  # Inițializezi gradientul la începutul epocii
    train_progress = tqdm(train_dataloader, desc="Training", leave=False)
    
    for step, batch in enumerate(train_progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps  # Împarte loss-ul
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("Invalid loss detected, skipping batch.")
            continue
        
        loss.backward()  # Backpropagation pentru acumularea gradientului
        train_loss += loss.item() * gradient_accumulation_steps  # Re-scalează loss-ul pentru totalizare
        
        if (step + 1) % gradient_accumulation_steps == 0:  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradient
            optimizer.step()  # Actualizează parametrii
            scheduler.step()
            optimizer.zero_grad()  # Resetează gradientul
        
        train_progress.set_postfix({"loss": loss.item() * gradient_accumulation_steps})  # Log loss-ul real
    
    train_loss /= len(train_dataloader)
    print(f"Epoch {epoch + 1}, Training Loss: {train_loss}")
    writer.add_scalar("Loss/Training", train_loss, epoch)

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
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=150,  # Specify how many tokens to generate
		num_beams=4,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            all_predictions.extend(predictions)
            all_references.extend(references)

            val_progress.set_postfix({"val_loss": outputs.loss.item()})
    
    val_loss /= len(eval_dataloader)
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")
    writer.add_scalar("Loss/Validation", val_loss, epoch)

    # Compute ROUGE score
    rouge_score = rouge.compute(predictions=all_predictions, references=all_references)
    print(f"Epoch {epoch + 1}, ROUGE Score: {rouge_score}")

    writer.add_scalar("ROUGE/ROUGE-1", rouge_score["rouge1"], epoch)
    writer.add_scalar("ROUGE/ROUGE-2", rouge_score["rouge2"], epoch)
    writer.add_scalar("ROUGE/ROUGE-L", rouge_score["rougeL"], epoch)
    writer.add_scalar("ROUGE/ROUGE-Lsum", rouge_score["rougeLsum"], epoch)

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        
        model.save_pretrained("./best_model_6_training")
        tokenizer.save_pretrained("./best_model_6_training")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"No improvement for {patience} epochs. Stopping early.")
            break
        
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, f"./checkpoint_epoch_{epoch + 1}.pth")

# 11. Final save
output_dir = "./llama_final_model_6_training"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
writer.close()
