from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, engine
import models
import requests
from fastapi.responses import JSONResponse
from models import User, Book, Review, UserBook
from schemas import PromptRequest, UserCreateRequest, BookCreate, BookUpdate, StatusUpdate
from middleware import TokenValidationMiddleware
from pydantic import BaseModel
from typing import Optional

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

app.add_middleware(TokenValidationMiddleware)
models.Base.metadata.create_all(bind = engine)

SERVER_URL = "http://192.168.251.219:8000/analyze-sentiment"

class ReviewRequest(BaseModel):
    review: str

@app.post("/sentiment_analysis")
async def sentiment_analysis(request: ReviewRequest):
    payload = {"review": request.review}
    response = requests.post(SERVER_URL, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.text, "status_code": response.status_code}

def sanitize_dict(obj):
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

def validate_admin_role(request: Request):
    user = request.scope.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized: User not authenticated")
    if user is None or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: insufficient permissions")
    
def validate_user_role(request: Request):
    user = request.scope.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized: User not authenticated")
    if user is None or user.get("role") != "user":
        raise HTTPException(status_code=403, detail="Forbidden: insufficient permissions")

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

@app.post("/books")
def create_book(request: Request, book : BookCreate, db: Session = Depends(get_db)):

    validate_admin_role(request=request)

    new_book = Book(**book.model_dump())
    db.add(new_book)
    db.commit()
    db.refresh(new_book)
    response_data = {
        "message": "Book created successfully",
        "id": new_book.id,
        "title": new_book.title,
        "author": new_book.author,
        "isbn": new_book.isbn,
        "goodreads_id": new_book.goodreads_id,
        "image_url": new_book.image_url,
    }
    return JSONResponse(status_code=201, content=response_data)

@app.get("/books/{book_id}")
def get_book(request: Request, book_id: int, db: Session = Depends(get_db)):

    validate_admin_role(request=request)

    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    return sanitize_dict(book)

@app.get("/books")
def get_books(user_email: str = Query(..., alias="user_email"),
              title : Optional[str] = Query(None, alias="title"),
              author : Optional[str] = Query(None, alias="author"),
              isbn : Optional[str] = Query(None, alias="isbn"),
              page: int = 1,
              items_per_page : int = 10,
              db: Session = Depends(get_db)):

    if page < 1 or items_per_page <= 0:
        raise HTTPException(status_code=422, detail="Page must be >= 1 and items_per_page > 0")
    
    query = db.query(Book, UserBook.status).outerjoin(
        UserBook, (UserBook.book_id == Book.id) & (UserBook.user_email == user_email)
    )

    if title:
        query = query.filter(Book.title.ilike(f"%{title}%"))

    if author:
        query = query.filter(Book.author.ilike(f"%{author}%"))

    if isbn:
        query = query.filter(Book.isbn == isbn)
    
    total_books = query.count()
    skip = (page - 1) * items_per_page
    books = query.offset(skip).limit(items_per_page).all()

    if skip >= total_books and total_books > 0:
        raise HTTPException(status_code=416, detail="Page out of range")

    return {
        "total_books": total_books,
        "page_number" : page,
        "books": [
            {
                "id": book.id,
                "title": book.title,
                "author": book.author,
                "isbn": book.isbn,
                "image_url": book.image_url,
                "status": status if status else "none"
            }
            for book, status in books
        ],
    }

@app.put("/books/{book_id}")
def update_book(request: Request, book_id: int, book_update : BookUpdate, db: Session = Depends(get_db)):

    validate_admin_role(request=request)

    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    for key, value in book_update.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(book, key, value)

    db.commit()
    response_data = {
        "message": "Book updated successfully",
        "id": book.id,
        "title": book.title,
        "author": book.author,
        "isbn": book.isbn,
        "goodreads_id": book.goodreads_id,
        "image_url": book.image_url,
    }
    return JSONResponse(status_code=200, content=response_data)

@app.delete("/books/{book_id}")
def delete_book(request: Request, book_id: int, db: Session = Depends(get_db)):

    validate_admin_role(request=request)

    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    db.delete(book)
    db.commit()
    response_data = {
        "message": f"Book {book_id} deleted successfully"
    }
    return JSONResponse(status_code=200, content=response_data)

@app.get("/reviews/{review_id}")
def get_review(request: Request, review_id: int, db: Session = Depends(get_db)):

    validate_admin_role(request=request)

    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    return sanitize_dict(review)

@app.get("/reviews")
def get_reviews(request: Request, 
                page: int = 1,
                items_per_page: int = 10,
                db: Session = Depends(get_db)):
    
    validate_admin_role(request=request)

    if page < 1 or items_per_page <= 0:
        raise HTTPException(status_code=422, detail="Page must be >= 1 and items_per_page > 0")
        
    total_reviews = db.query(Review).count()
    skip = (page - 1) * items_per_page
    reviews = db.query(Review).offset(skip).limit(items_per_page).all()

    if skip >= total_reviews and total_reviews > 0:
        raise HTTPException(status_code=416, detail="Page out of range")

    return {
        "total_reviews": total_reviews,
        "page_number": page,
        "reviews": [sanitize_dict(review) for review in reviews],
    }

@app.get("/users")
def get_all_users(request: Request,
                  page: int = 1,
                  items_per_page: int = 10,
                  db: Session = Depends(get_db)):

    validate_admin_role(request=request)

    if page < 1 or items_per_page <= 0:
        raise HTTPException(status_code=422, detail="Page must be >= 1 and items_per_page > 0")
    
    total_users = db.query(User).count()
    skip = (page - 1) * items_per_page
    users = db.query(User).offset(skip).limit(items_per_page).all()

    if skip >= total_users and total_users > 0:
        raise HTTPException(status_code=416, detail="Page out of range")

    return {
        "total_users": total_users,
        "page_number": page,
        "users": [sanitize_dict(user) for user in users],
    }

@app.delete("/deleteUser")
def delete_user(email: str = Query(...), db: Session = Depends(get_db)):

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(user)
    db.commit()
    response_data = {
        "message": f"User {email} deleted successfully"
    }
    return JSONResponse(status_code=200, content=response_data)

@app.delete("/reviews/{review_id}")
def delete_review(request: Request, review_id: int, db: Session = Depends(get_db)):

    validate_admin_role(request=request)

    review = db.query(Review).filter(Review.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    db.delete(review)
    db.commit()
    response_data = {
        "message": "Review deleted successfully"
    }
    return JSONResponse(status_code=200, content=response_data)

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

@app.post("/update_status")
def update_status(request: Request, data: StatusUpdate, db: Session = Depends(get_db)):

    validate_user_role(request=request)
    
    book = db.query(Book).filter(Book.id == data.book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    user_book = db.query(UserBook).filter(
        UserBook.user_email == data.user_email,
        UserBook.book_id == data.book_id
    ).first()

    if user_book:
        if data.status == "none":
            db.delete(user_book)
        else:
            user_book.status = data.status
    
    else:
        if data.status != "none":
            new_user_book = UserBook(
                user_email=data.user_email,
                book_id=data.book_id,
                status=data.status
            )
            db.add(new_user_book)

    db.commit()
    response_data = {
        "message": "Status updated successfully"
    }

    return JSONResponse(status_code=201, content=response_data)

@app.get("/to_read_books/{user_email}")
def get_to_read_books(request: Request, user_email: str, page: int = 1, items_per_page: int = 4, db: Session = Depends(get_db)):

    validate_user_role(request=request)

    if page < 1 or items_per_page <= 0:
        raise HTTPException(status_code=422, detail="Page must be >= 1 and items_per_page > 0")
    
    books_query = (
        db.query(Book)
        .join(UserBook, Book.id == UserBook.book_id)
        .filter(UserBook.user_email == user_email, UserBook.status == "to_read")
    )

    total_books = books_query.count()
    skip = (page - 1) * items_per_page
    books = books_query.offset(skip).limit(items_per_page).all()

    if skip >= total_books and total_books > 0:
        raise HTTPException(status_code=416, detail="Page out of range")

    return {"books": [sanitize_dict(book) for book in books], "total_books": total_books}

