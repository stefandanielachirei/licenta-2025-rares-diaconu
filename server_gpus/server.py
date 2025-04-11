from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from schemas import ReviewInput, SentimentRequest, TextsRequest
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import JSONResponse
import torch
import os
import numpy as np

app = FastAPI()

device = 0 if torch.cuda.is_available() else -1

HF_TOKEN = os.getenv("HF_TOKEN")

base_model_path = "Meta-LLaMA/LLaMA-3.2-1B"
lora_model_path = "./model_trained"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, token=HF_TOKEN)
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

llama_model = PeftModel.from_pretrained(base_model, lora_model_path)
llama_model.to(device)
llama_model.eval()
torch.cuda.empty_cache()

def llama_summarize(text: str) -> str:
    prompt = f"Summarize the following text in a clear and concise sentence:\n\n{text.strip()}"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = llama_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=40,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.replace(prompt, "").strip()

def label_from_score(label, score):
    if label == "POSITIVE":
        if score > 0.9:
            return "very_positive"
        elif score > 0.7:
            return "positive"
        else:
            return "neutral"
    else:
        if score > 0.9:
            return "very_negative"
        elif score > 0.7:
            return "negative"
        else:
            return "neutral"

long_summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

@app.post("/summarize_reviews")
def summarize_reviews(payload: ReviewInput):
    results = []

    for text in payload.texts:
        token_count = len(text.split())

        if token_count < payload.max_tokens_threshold:
            summary = llama_summarize(text)
            method = "llama_custom"
        else:
            summary = long_summary_pipeline(text, max_length=100, min_length=40, do_sample=False)[0]["summary_text"]
            method = "bart"

        results.append({
            "original": text,
            "summary": summary,
            "method": method
        })

    return JSONResponse(
        status_code=201,
        content={"summaries": results}
    )

model_name_sentiment = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name_sentiment,
    device=device,
    torch_dtype=torch.float16 if device == 0 else None
)

@app.post("/analyze-sentiment")
def analyze_sentiment(payload: SentimentRequest):
    try:
        results = sentiment_pipeline(payload.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    response = []
    for text, result in zip(payload.texts, results):
        label = result["label"]
        score = result["score"]
        fine_label = label_from_score(label, score)
        response.append({
            "text": text,
            "label": label,
            "score": score,
            "fine_label": fine_label
        })

    return JSONResponse(
        status_code=201,
        content={"sentiments": response}
    )
    
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.post("/most-dissimilar")
def most_dissimilar_reviews(request: TextsRequest):
    texts = request.texts

    if len(texts) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 reviews")

    embeddings = similarity_model.encode(texts)

    sim_matrix = cosine_similarity(embeddings)

    dissim_matrix = 1 - sim_matrix
    np.fill_diagonal(dissim_matrix, 0)

    scores = dissim_matrix.mean(axis=1)

    top_indices = np.argsort(scores)[-5:][::-1]

    return JSONResponse(
        status_code=201,
        content={"indices": top_indices.tolist()}
    )

# Rulează serverul: uvicorn server:app --host 0.0.0.0 --port 8000
