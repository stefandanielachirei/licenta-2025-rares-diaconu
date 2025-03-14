from sqlalchemy import Column, Integer, String, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from database import Base

class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    author = Column(String)
    isbn = Column(String, unique=True)
    goodreads_id = Column(String, unique=True, nullable=False)
    image_url = Column(Text, nullable=True)

    reviews = relationship("Review", back_populates="book", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = "users"

    email = Column(String, primary_key=True)
    role = Column(String, nullable=False)

class UserBook(Base):
    __tablename__ = "user_books"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, ForeignKey("users.email"))
    book_id = Column(Integer, ForeignKey("books.id"))
    status = Column(Enum("to_read", "read", name="status_enum"))

    user = relationship("User", backref="user_books")
    book = relationship("Book", backref="user_books")

class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    book_id = Column(Integer, ForeignKey("books.id"))
    user_email = Column(String, ForeignKey("users.email"))
    review_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)

    book = relationship("Book", back_populates="reviews")
    user = relationship("User", backref="reviews")

