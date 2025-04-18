from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import get_db, engine
from fastapi.responses import JSONResponse
from models import User, Book, Review, UserBook
from schemas import PromptRequest, UserCreateRequest, BookCreate, BookUpdate, StatusUpdate, ReviewCreate, ReviewDelete
from middleware import TokenValidationMiddleware
from typing import Optional
from dotenv import load_dotenv
import models
import requests
import os

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

load_dotenv()

app.add_middleware(TokenValidationMiddleware)
models.Base.metadata.create_all(bind = engine)

SENTIMENT_URL = os.getenv("SENTIMENT_URL")
SUMMARIZATION_SERVER_URL = os.getenv("SUMMARIZATION_SERVER_URL")
SIMILARITY_SERVER_URL = os.getenv("SIMILARITY_SERVER_URL")

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

    validate_user_role(request=request)

    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    return sanitize_dict(book)

@app.get("/books_admin")
def get_books(request: Request, 
              page: int = 1,
              items_per_page : int = 10,
              db: Session = Depends(get_db)):
    
    validate_admin_role(request=request)

    if page < 1 or items_per_page <= 0:
        raise HTTPException(status_code=422, detail="Page must be >= 1 and items_per_page > 0")
    
    total_books = db.query(Book).count()
    skip = (page - 1) * items_per_page
    books = db.query(Book).offset(skip).limit(items_per_page).all()

    if skip >= total_books and total_books > 0:
        raise HTTPException(status_code=416, detail="Page out of range")

    return {
        "total_books": total_books,
        "page_number" : page,
        "books": [sanitize_dict(book) for book in books]
    }

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

@app.get("/read_books/{user_email}")
def get_read_books(request: Request, user_email: str, page: int = 1, items_per_page: int = 4, db: Session = Depends(get_db)):

    validate_user_role(request=request)

    if page < 1 or items_per_page <= 0:
        raise HTTPException(status_code=422, detail="Page must be >= 1 and items_per_page > 0")
    
    books_query = (
        db.query(Book)
        .join(UserBook, Book.id == UserBook.book_id)
        .filter(UserBook.user_email == user_email, UserBook.status == "read")
    )

    total_books = books_query.count()
    skip = (page - 1) * items_per_page
    books = books_query.offset(skip).limit(items_per_page).all()

    if skip >= total_books and total_books > 0:
        raise HTTPException(status_code=416, detail="Page out of range")

    return {"books": [sanitize_dict(book) for book in books], "total_books": total_books}

@app.post("/add_review")
def add_review(request: Request, review:ReviewCreate , db: Session = Depends(get_db)):

    validate_user_role(request=request)

    book = db.query(Book).filter(Book.id == review.book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    user = db.query(User).filter(User.email == review.user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    existing_review = db.query(Review).filter(
        Review.book_id == review.book_id,
        Review.user_email == review.user_email
    ).first()

    if existing_review:
        raise HTTPException(status_code=409, detail="You have already reviewed this book, please update your existing review")
    
    new_review = Review(
        book_id=review.book_id,
        user_email=review.user_email,
        summary = review.summary,
        review_text = review.review_text
    )

    db.add(new_review)
    db.commit()
    db.refresh(new_review)

    response_data = {
        "message": "Review added successfully", 
        "review_id": new_review.id
    }

    return JSONResponse(status_code=201, content=response_data)

@app.put("/update_review")
def update_review(request: Request, review: ReviewCreate, db: Session = Depends(get_db)):

    validate_user_role(request=request)

    existing_review = db.query(Review).filter(
        Review.book_id == review.book_id,
        Review.user_email == review.user_email
    ).first()

    if not existing_review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    existing_review.review_text = review.review_text
    db.commit()

    response_data = {
        "message": "Review updated successfully",
        "review_id": existing_review.id
    }

    return JSONResponse(status_code=200, content=response_data)

@app.delete("/delete_review")
def delete_review(request: Request, review: ReviewDelete, db: Session = Depends(get_db)):
    
    validate_user_role(request=request)

    review_to_delete = db.query(Review).filter(
        Review.book_id == review.book_id,
        Review.user_email == review.user_email
    ).first()

    if not review_to_delete:
        raise HTTPException(status_code=404, detail="Review not found")
    
    db.delete(review_to_delete)
    db.commit()

    response_data = {
        "message": "Review deleted successfully",
        "review_id": review_to_delete.id
    }

    return JSONResponse(status_code=200, content=response_data)
                  
@app.get("/reviews")
def get_reviews_by_user(request: Request, email: str = Query(...), db: Session = Depends(get_db)):

    validate_user_role(request=request)
    reviews = db.query(Review).filter(Review.user_email == email).all()

    if not reviews:
        raise HTTPException(status_code=404, detail="No reviews found for this user")

    return reviews

@app.get("/books/{book_id}/summaries_live")
def live_summaries_for_book(
    request : Request,
    book_id: int,
    user_email: str = Query(..., alias="user_email"),
    page: int = 1, 
    items_per_page: int = 4,
    db: Session = Depends(get_db)
):
    
    validate_user_role(request=request)

    if page < 1 or items_per_page <= 0:
        raise HTTPException(status_code=422, detail="Page must be >= 1 and items_per_page > 0")

    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    
    user_review = (
        db.query(Review)
        .filter(Review.book_id == book_id, Review.user_email == user_email)
        .first()
    )

    reviews_query = db.query(Review).filter(
        Review.book_id == book_id,
        Review.user_email != user_email if user_review else True
    )

    total = reviews_query.count() + (1 if user_review else 0)
    skip = (page - 1) * items_per_page

    reviews = []
    if page == 1:
        if user_review:
            reviews.append(user_review)
            remaining_limit = items_per_page - 1
        else:
            remaining_limit = items_per_page

        additional_reviews = reviews_query.limit(remaining_limit).all()
        reviews.extend(additional_reviews)
    else:
        offset_value = skip - (1 if user_review and page > 1 else 0)
        reviews = reviews_query.offset(offset_value).limit(items_per_page).all()


    texts = [r.review_text for r in reviews if r.review_text]

    payload = {
        "texts": texts,
        "max_tokens_threshold": 40
    }

    try:
        response = requests.post(SUMMARIZATION_SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Summarization error: {e}")

    summaries = response.json()["summaries"]

    return {
        "total": total,
        "page": page,
        "items_per_page": items_per_page,
        "summaries": [
            {
                "review_id": r.id,
                "original": s["original"],
                "summary": s["summary"],
                "method": s["method"]
            }
            for r, s in zip(reviews, summaries)
        ]
    }

@app.get("/books/{book_id}/reviews_sentiment")
def reviews_sentiment_analysis(
    request: Request,
    book_id: int,
    user_email: str = Query(..., alias="user_email"),
    page: int = 1,
    items_per_page: int = 4,
    db: Session = Depends(get_db)
):
    validate_user_role(request=request)

    if page < 1 or items_per_page <= 0:
        raise HTTPException(status_code=422, detail="Invalid pagination")

    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    user_review = (
        db.query(Review)
        .filter(Review.book_id == book_id, Review.user_email == user_email)
        .first()
    )

    reviews_query = db.query(Review).filter(
        Review.book_id == book_id,
        Review.user_email != user_email if user_review else True
    )

    total = reviews_query.count() + (1 if user_review else 0)
    skip = (page - 1) * items_per_page

    reviews = []
    if page == 1:
        if user_review:
            reviews.append(user_review)
            remaining_limit = items_per_page - 1
        else:
            remaining_limit = items_per_page

        additional_reviews = reviews_query.limit(remaining_limit).all()
        reviews.extend(additional_reviews)
    else:
        offset_value = skip - (1 if user_review and page > 1 else 0)
        reviews = reviews_query.offset(offset_value).limit(items_per_page).all()

    texts = [r.review_text.strip() for r in reviews if r.review_text and r.review_text.strip()]

    payload = {
        "texts": texts
    }

    try:
        response = requests.post(SENTIMENT_URL, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {e}")

    sentiments = response.json()["sentiments"]

    return {
        "total": total,
        "page": page,
        "items_per_page": items_per_page,
        "reviews": [
            {
                "review_id": r.id,
                "text": s["text"],
                "label": s["label"],
                "score": round(s["score"], 4),
                "fine_label": s["fine_label"]
            }
            for r, s in zip(reviews, sentiments)
        ]
    }

@app.get("/books/{book_id}/top-dissimilar")
def get_top_dissimilar_reviews_for_book(request: Request, book_id: int, db: Session = Depends(get_db)):

    validate_user_role(request=request)

    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    

    reviews = db.query(Review).filter(Review.book_id == book_id).all()

    if not reviews:
        raise HTTPException(status_code=404, detail="No reviews found for this book")

    texts = [review.review_text for review in reviews]

    try:
        response = requests.post(SIMILARITY_SERVER_URL, json={"texts": texts}, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Similarity server error: {e}")

    data = response.json()
    top_indices = data.get("indices", [])

    top_dissimilar = []
    seen_ids = set()

    for idx in top_indices:
        if 0 <= idx < len(reviews):
            review = reviews[idx]
            if review.id not in seen_ids:
                top_dissimilar.append({
                    "review_id": review.id,
                    "text": review.review_text
                })
                seen_ids.add(review.id)

    return {"top_dissimilar_reviews": top_dissimilar}

@app.get("/books/{book_id}/user-review")
def get_user_review_for_book(
    request: Request,
    book_id: int,
    user_email: str = Query(..., alias="user_email"),
    db: Session = Depends(get_db)
):
    validate_user_role(request=request)

    book = db.query(Book).filter(Book.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    review = db.query(Review).filter(
        Review.book_id == book_id,
        Review.user_email == user_email
    ).first()

    if not review:
        raise HTTPException(status_code=404, detail="Review not found")

    return sanitize_dict(review)



