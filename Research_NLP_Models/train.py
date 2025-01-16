import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, get_cosine_with_hard_restarts_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import evaluate
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

cache_dir = 'K:/Work/Rares/cache2'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
os.environ['HF_DATASETS_CACHE'] = cache_dir

model_name = "meta-llama/Llama-3.2-1B"
model_dir = "./llama_final_model_big_patent_5_training"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

writer = SummaryWriter(log_dir="./tensorboard_logs_big_patent_7_training")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = "left"

base_model.resize_token_embeddings(len(tokenizer))

adapter_model = PeftModel.from_pretrained(base_model, model_dir)

print(f"Dropout rate in the model: {adapter_model.config.hidden_dropout_prob if hasattr(adapter_model.config,'hidden_dropout_prob') else 'Not specified, likely default.'}")

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Adăugat k_proj și o_proj
    lora_dropout=0.6,
    bias="none"
)

model = get_peft_model(adapter_model, lora_config)
model.print_trainable_parameters()

model = model.to(device)

dataset = load_dataset("big_patent", trust_remote_code=True) 

start_idx = 800000
end_idx = 900000
dataset["train"] = dataset["train"].select(range(start_idx, end_idx))

def preprocess(example):
    prompt = f"Summarize the following text in maximum 3 important sentences:\n{example['description']}\n\nConcise summary:"
    inputs = tokenizer(
        prompt, 
        max_length=256,
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    labels = tokenizer(
        example["abstract"], 
        max_length=256,
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    inputs["labels"] = labels["input_ids"]
    if torch.all(inputs["input_ids"] == tokenizer.pad_token_id) or torch.all(labels["input_ids"] == tokenizer.pad_token_id):
        print(f"Skipped invalid example: {example}")
        return None
    return {k: v.squeeze() for k, v in inputs.items()}

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=8, shuffle=True, collate_fn=data_collator)
dataset["validation"] = dataset["validation"].select(range(0, 5000))
eval_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=16, collate_fn=data_collator)

# optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.25)

rouge = evaluate.load("rouge")

# early stopping configuration
patience = 3
best_val_loss = float('inf')
early_stop_counter = 0
gradient_accumulation_steps = 2  # Redus la 2
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
num_epochs = 30

total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0.1 * total_steps,
    num_training_steps=total_steps,
    num_cycles = 3
)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()
    train_loss = 0
    optimizer.zero_grad()
    train_progress = tqdm(train_dataloader, desc="Training", leave=False)
    
    for step, batch in enumerate(train_progress):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels = batch["labels"].view(-1)
        loss = loss_fn(logits, labels) / gradient_accumulation_steps
        #loss = outputs.loss / gradient_accumulation_steps
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("Invalid loss detected, skipping batch.")
            continue
        
        loss.backward()
        train_loss += loss.item() * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        train_progress.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
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
            
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=100,
                num_beams=3,
                temperature=0.6,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.3,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
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

    # ROUGE scores
    rouge_score = rouge.compute(predictions=all_predictions, references=all_references)
    print(f"Epoch {epoch + 1}, ROUGE Score: {rouge_score}")

    writer.add_scalar("ROUGE/ROUGE-1", rouge_score["rouge1"], epoch)
    writer.add_scalar("ROUGE/ROUGE-2", rouge_score["rouge2"], epoch)
    writer.add_scalar("ROUGE/ROUGE-L", rouge_score["rougeL"], epoch)
    writer.add_scalar("ROUGE/ROUGE-Lsum", rouge_score["rougeLsum"], epoch)

    # early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        model.save_pretrained("./best_model_big_patent_7_training")
        tokenizer.save_pretrained("./best_model_big_patent_7_training")
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

# final save
output_dir = "./llama_final_model_big_patent_7_training"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
writer.close()