import tensorflow as tf
import torch
import os
import random
import json
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

def preprocess_function(examples):
    inputs = [ex["input_text"] for ex in examples["text"]]
    targets = [ex["summary_text"] for ex in examples["summary"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
        print("TensorFlow configurat pentru a folosi GPU 1")
    except RuntimeError as e:
        print("Eroare la configurarea TensorFlow pentru GPU:", e)


if torch.cuda.is_available():
    device = torch.device("cuda:1")
    print("Torch configurat pentru a folosi GPU 1")
else:
    device = torch.device("cpu")
    print("Torch configurat să folosească CPU")


dataset_path = r"K:\Work\Rares\all_chapterized_books"

data = []

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

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenized_datasets = booksum.map(preprocess_function, batched=True)

#model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    load_best_model_at_end=True,
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