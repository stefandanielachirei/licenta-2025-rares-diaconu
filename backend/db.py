import pandas as pd
import json
import requests
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Book, Review, User
from PIL import Image
import io

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
    isbn = str(isbn).replace('.', '').split('e')[0]
    
    ol_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    response = requests.get(ol_url)
    
    if response.status_code == 200:
        try:
            image = Image.open(io.BytesIO(response.content))
            if image.size[0] > 1 and image.size[1] > 1:
                return ol_url
        except Exception as e:
            print(f"Invalid image from Open Library for ISBN {isbn}: {e}")
    
    gb_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
    response = requests.get(gb_url)
    data = response.json()
    
    if "items" in data:
        google_cover = data["items"][0]["volumeInfo"].get("imageLinks", {}).get("thumbnail")
        if google_cover:
            return google_cover
    
    return None

def import_books():
    db = SessionLocal()

    if db.query(Book).count() > 0:
        print("The books are already in the database.")
        db.close()
        return
    
    books_df = pd.read_csv("books.csv", dtype={'isbn': str}, nrows=20)

    imported_books = 0

    for _, row in books_df.iterrows():
        isbn = str(row['isbn']).replace('.', '').split('e')[0] if 'isbn' in row and pd.notna(row['isbn']) else None
        if not isbn:
            continue

        image_url = fetch_cover_url(isbn)
        if not image_url:
            print(f"Skipping book with ISBN {isbn}: No cover found.")
            continue

        goodreads_id = str(row['goodreads_book_id']) if 'goodreads_book_id' in row and pd.notna(row['goodreads_book_id']) else None

        book = Book(
            title=row['original_title'],
            author=row['authors'].split(",")[0] if 'authors' in row and pd.notna(row['authors']) else "Unknown",
            isbn=isbn,
            goodreads_id=goodreads_id,
            image_url=image_url
        )
        db.add(book)
        imported_books += 1

    db.commit()
    db.close()
    print(f"Successfully imported {imported_books} books!")

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
