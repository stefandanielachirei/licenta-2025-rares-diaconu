from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Configurarea conexiunii la PostgreSQL
DATABASE_URL = f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@postgres:5432/{os.getenv("POSTGRES_DB")}'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Definirea modelului pentru baza de date
class Test(Base):
    __tablename__ = 'test'
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(80), unique=True, nullable=False)

# Creăm tabelele în baza de date
Base.metadata.create_all(bind=engine)

# Inițializarea aplicației FastAPI
app = FastAPI()

# Schema pentru datele de răspuns
class TestResponse(BaseModel):
    text: str

# Ruta principală
@app.get("/api/text", response_model=TestResponse)
def get_text():
    db = SessionLocal()
    first_entry = db.query(Test).first()
    db.close()
    if first_entry:
        return {"text": first_entry.text}
    else:
        raise HTTPException(status_code=404, detail="No entries found in the database!")
