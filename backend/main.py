from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from database import get_db, engine
from models import Base, Book, Review
import os
import requests

DATABASE_URL = f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@postgres:5432/{os.getenv("POSTGRES_DB")}'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

Base.metadata.create_all(bind=engine)

app = FastAPI()

NLP_URL = "http://nlp_model:8001/generate"

class TestResponse(BaseModel):
    text: str

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate_text")
def generate_text(request: PromptRequest) :
    prompt = request.prompt
    response = requests.post(NLP_URL, json={"prompt":prompt})
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail="Error generating text")
    
@app.get("/books")
def get_books(db: Session = Depends(get_db)):
    return db.query(Book).limit(10).all()

@app.get("/reviews")
def get_reviews(db: Session = Depends(get_db)):
    reviews = db.query(Review).limit(10).all()
    return reviews
