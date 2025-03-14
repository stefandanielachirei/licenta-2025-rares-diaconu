import pandas as pd
import json
import requests
import gc
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Book, Review, User

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Sau load_in_4bit=True pentru și mai puțină memorie
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_threshold=6.0
)

model_id = "sshleifer/distilbart-cnn-12-6"
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"  # Plasează automat pe GPU dacă este disponibil
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

def create_default_user():
    db = SessionLocal()
    existing_user = db.query(User).filter(User.email == "anonim@gmail.com").first()
    
    if not existing_user:
        user = User(email="anonim@gmail.com", role="user")
        db.add(user)
        db.commit()
        print("Default user 'anonim@gmail.com' created.")
    
    db.close()

def fetch_cover_url(isbn):
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    response = requests.get(url)
    return url if response.status_code == 200 else None

def import_books():
    """Importă cărțile din CSV și salvează goodreads_id în baza de date."""
    db = SessionLocal()

    if db.query(Book).count() > 0:
        print("The books are already in the database.")
        db.close()
        return
    
    books_df = pd.read_csv("books.csv", nrows=3)

    for _, row in books_df.iterrows():
        isbn = str(row['isbn']) if 'isbn' in row and pd.notna(row['isbn']) else None
        goodreads_id = str(row['goodreads_book_id']) if 'goodreads_book_id' in row and pd.notna(row['goodreads_book_id']) else None
        image_url = fetch_cover_url(isbn) if isbn else None

        book = Book(
            title=row['book_id'],  # Trebuie să verifici dacă 'book_id' este corect pentru titlu
            author="anonim@gmail.com",  # Modifică dacă ai coloana cu autori
            isbn=isbn,
            goodreads_id=goodreads_id,  # Folosim goodreads_id
            image_url=image_url
        )
        db.add(book)

    db.commit()
    db.close()
    print("The books have been imported successfully!")

def import_reviews():
    db: Session = SessionLocal()

    if db.query(Review).count() > 0:
        print("The reviews are already in the database.")
        db.close()
        return
    
    existing_books = {str(book.goodreads_id): book.id for book in db.query(Book).all() if book.goodreads_id}
    
    with open("goodreads_reviews_spoiler_raw.json", "r", encoding="utf-8-sig") as f:
        for line in f:
            try:
                review_data = json.loads(line)
                book_id = str(review_data.get("book_id")).strip()
                review_text = review_data.get("review_text", "").strip()
                
                if book_id not in existing_books or not review_text or len(review_text) < 10:
                    continue
                
                review = Review(
                    book_id=existing_books[book_id],
                    user_email="anonim@gmail.com",
                    review_text=review_text
                )
                db.add(review)
                db.commit()
            
            except Exception as e:
                print(f"Skipping review due to error: {e}")
                db.rollback()
                continue
    
    db.close()
    print("All matching reviews were imported!")


if __name__ == "__main__":
    create_default_user()
    import_books()
    import_reviews()
