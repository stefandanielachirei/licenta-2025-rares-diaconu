from fastapi import FastAPI
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    prompt: str

model_path = "./model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

@app.post("/generate")
async def generate_text(request: TextRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=50,  # Numai 50 de token-uri noi
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}

