from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import PeftModel
from schemas import TextRequest
import torch
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.getenv("HF_TOKEN")

base_model_path = "Meta-LLaMA/LLaMA-3.2-1B"
lora_model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, token = HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
) if device == "cuda" else None

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    quantization_config=bnb_config,
    token=HF_TOKEN
)

model = PeftModel.from_pretrained(base_model, lora_model_path)
model.to(device)
model.eval()
torch.cuda.empty_cache()

@app.post("/generate")
async def generate_text(request: TextRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}
