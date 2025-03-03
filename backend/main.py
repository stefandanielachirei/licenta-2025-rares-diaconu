from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import requests

DATABASE_URL = f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@postgres:5432/{os.getenv("POSTGRES_DB")}'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Test(Base):
    __tablename__ = 'test'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(80), unique=True, nullable=False)

Base.metadata.create_all(bind=engine)

app = FastAPI()

NLP_URL = "http://nlp_model:8001/generate"

class TestResponse(BaseModel):
    text: str

class PromptRequest(BaseModel):
    prompt: str

@app.get("/api/text", response_model=TestResponse)
def get_text():
    db = SessionLocal()
    first_entry = db.query(Test).first()
    db.close()
    if first_entry:
        return {"text": first_entry.text}
    else:
        raise HTTPException(status_code=404, detail="No entries found in the database!")

@app.post("/generate_text")
def generate_text(request: PromptRequest) :
    prompt = request.prompt
    response = requests.post(NLP_URL, json={"prompt":prompt})
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=response.status_code, detail="Error generating text")
