import tensorflow as tf
import torch
import os
import random
import json
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
max_length_allowed = 512

def preprocess_function(examples):
    inputs = tokenizer(
        examples["input_text"],
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    targets = tokenizer(
        examples["summary_text"],
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    inputs["labels"] = targets["input_ids"]

    inputs["labels"] = inputs["labels"].masked_fill(inputs["labels"] == tokenizer.pad_token_id, -100)

    if inputs["input_ids"].shape != inputs["labels"].shape:
        print(f"Dimensiuni incompatibile: input_ids {inputs['input_ids'].shape} vs labels {inputs['labels'].shape}")
    
    print(f"input_ids shape: {inputs['input_ids'].shape}, labels shape: {inputs['labels'].shape}")
    
    return inputs

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


dataset_path = r"K:\Work\Rares\all_chapterized_books"

data = []
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")  # Inițializează tokenizerul pentru modelul folosit

for book_folder in os.listdir(dataset_path):
    book_path = os.path.join(dataset_path, book_folder)
    
    if os.path.isdir(book_path):
        chapters = []
        
        for chapter_file in sorted(os.listdir(book_path)):
            chapter_path = os.path.join(book_path, chapter_file)
            
            if chapter_file.endswith(".txt") and not chapter_file.startswith("toc"):
                with open(chapter_path, "r", encoding="utf-8", errors="replace") as f:
                    chapter_text = f.read()
                    chapters.append(chapter_text)
        
        input_text = "\n\n".join(chapters)
        summary_text = "Acesta este sumarul cărții."

        # Tokenizează textele pentru a verifica numărul de tokeni
        input_tokens = tokenizer(input_text, truncation=False)["input_ids"]
        summary_tokens = tokenizer(summary_text, truncation=False)["input_ids"]

        # Verifică dacă depășesc numărul maxim permis de tokeni
        if len(input_tokens) > max_length_allowed or len(summary_tokens) > max_length_allowed:
            print("Exemplu exclus: lungime excesivă a textului.")
            continue

        # Adaugă exemplul la dataset dacă este valid
        data.append({
            "input_text": input_text,
            "summary_text": summary_text
        })

random.shuffle(data)
train_data = data[:int(0.8 * len(data))]
validation_data = data[int(0.8 * len(data)):int(0.9 * len(data))]
test_data = data[int(0.9 * len(data)):]

output_dir = "./dataset_json"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

with open(os.path.join(output_dir, "validation.json"), "w", encoding="utf-8") as f:
    json.dump(validation_data, f, indent=4, ensure_ascii=False)

with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)

print("Fișierele JSON au fost create cu succes!")

cache_directory = "./huggingface_cache"
dataset_path = "./dataset_json"
booksum = load_dataset("json", data_files={"train": f"{dataset_path}/train.json",
                                           "validation": f"{dataset_path}/validation.json",
                                           "test": f"{dataset_path}/test.json"},
                                cache_dir= cache_directory)



model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="balanced")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", padding="max_length")
tokenized_datasets = booksum.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    load_best_model_at_end=True,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

trainer.train()

#trainer.evaluate(tokenized_datasets["test"])