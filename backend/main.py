from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, engine
import models
import schemas
import requests
from fastapi.responses import JSONResponse
from models import User, Book, UserBook, Review
from schemas import PromptRequest, UserCreateRequest

app = FastAPI(
    title="Web application similar with goodreads using natural language processing(NLP) and RESTFul APIs",
    description="""This is a web application where there are many similarities with the goodreads app from Amazon"
    "with the help of NLP and RESTFul APIs""",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models.Base.metadata.create_all(bind = engine)

@app.post("/save-user")
def save_user(user_data: UserCreateRequest, db: Session = Depends(get_db)):

    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = User(email=user_data.email, role=user_data.role)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    response_data = {
        "message": "User registered successfully",
        "email": new_user.email, 
        "role": new_user.role
    }
    return JSONResponse(status_code=201, content=response_data)

@app.post("/generate_text")
def generate_text(request: PromptRequest) :
    prompt = request.prompt
    response = requests.post("http://nlp_model:8001/generate", json={"prompt":prompt})
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
