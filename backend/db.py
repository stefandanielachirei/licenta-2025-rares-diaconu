import pandas as pd
import json
import requests
from transformers import pipeline
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Book, Review

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

Base.metadata.create_all(bind=engine)

def fetch_cover_url(isbn):
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    response = requests.get(url)
    return url if response.status_code == 200 else None

def summarize_text(text):
    if len(text) < 50:
        return text
    try:
        summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Failed summarization: {e}")
        return text

def import_books():
    db = SessionLocal()
    
    if db.query(Book).count() > 0:
        print("The books are already in the database.")
        db.close()
        return
    
    books_df = pd.read_csv("books.csv", nrows=30)

    for _, row in books_df.iterrows():
        isbn = str(row['isbn']) if 'isbn' in row else None
        image_url = fetch_cover_url(isbn) if isbn else None
        
        book = Book(
            title=row['title'],
            author=row['authors'].split(",")[0] if 'authors' in row and pd.notna(row['authors']) else "Unknown",
            isbn=isbn,
            image_url=image_url
        )
        db.add(book)

    db.commit()
    db.close()
    print("The books have been imported with success!")

def import_reviews():
    """Importă și rezumă un review pentru fiecare carte din baza de date (maxim 30)."""
    db: Session = SessionLocal()

    if db.query(Review).count() > 0:
        print("The reviews are already in the database.")
        db.close()
        return

    # Obținem toate ID-urile cărților existente în baza de date
    existing_books = {str(book.id): book for book in db.query(Book).all()}
    
    count = 0
    used_books = set()

    with open("goodreads_reviews_spoiler.json", "r") as f:
        for line in f:
            review_data = json.loads(line)
            book_id = str(review_data.get("book_id")).strip()

            if book_id not in existing_books or book_id in used_books:
                continue

            review_text = review_data.get("review_text", "").strip()

            if not review_text or len(review_text) < 10:
                continue

            summary = summarize_text(review_text)

            review = Review(
                book_id=book_id,
                user_email="anonim@gmail.com",
                review_text=review_text,
                summary=summary
            )
            db.add(review)
            count += 1
            used_books.add(book_id)

            if count >= 30:
                break

    db.commit()
    db.close()
    print(f"{count} reviews were imported and summarized!")


if __name__ == "__main__":
    import_books()
    import_reviews()
