from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

device = 0 if torch.cuda.is_available() else -1

model_name_sentiment = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model_name_sentiment,
    device=device,
    torch_dtype=torch.float16 if device == 0 else None
)

class ReviewRequest(BaseModel):
    review: str

@app.post("/analyze-sentiment")
async def analyze_sentiment(request: ReviewRequest):
    result = sentiment_pipeline(request.review)
    return {"label": result[0]["label"], "score": result[0]["score"]}

# Rulează serverul: uvicorn server:app --host 0.0.0.0 --port 8000
