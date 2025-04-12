'use client';

import { useEffect, useState } from "react";
import { useParams } from 'next/navigation';

export default function BookPage() {
  const { id } = useParams();
  const bookId = id;

  interface Book {
    title: string;
    author: string;
    isbn: string;
    image_url: string;
  }

  const [book, setBook] = useState<Book | null>(null);
  const [reviews, setReviews] = useState([]);
  const [page, setPage] = useState(1);
  const [topDissimilar, setTopDissimilar] = useState([]);
  const itemsPerPage = 4;

  const fetchBook = async () => {
    try {
      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token is missing");
      }
  
      const validateResponse = await fetch("http://localhost:8080/validate", {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
  
      if (!validateResponse.ok) {
        throw new Error("Token validation failed");
      }
  
      const userInfo = await validateResponse.json();
      const userEmail = userInfo.username;
  
      const bookResponse = await fetch(`http://localhost:8000/books/${bookId}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });
  
      if (!bookResponse.ok) {
        throw new Error("Failed to fetch book details");
      }
  
      const bookData = await bookResponse.json();
      setBook(bookData);
  
      const reviewParams = new URLSearchParams({
        user_email: userEmail,
        page: page.toString(),
        items_per_page: itemsPerPage.toString(),
      });
  
      const reviewResponse = await fetch(
        `http://localhost:8000/books/${bookId}/reviews_sentiment?${reviewParams.toString()}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
        }
      );
  
      if (!reviewResponse.ok) {
        throw new Error("Failed to fetch reviews");
      }
  
      const reviewsData = await reviewResponse.json();
      setReviews(reviewsData.reviews);
  
    } catch (error) {
      console.error("Fetch error:", error);
    }
  };

  const fetchDissimilarReviews = async () => {
    try {
      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token is missing");
      }
  
      const response = await fetch(`http://localhost:8000/books/${bookId}/top-dissimilar`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      });
  
      if (!response.ok) {
        throw new Error("Failed to fetch dissimilar reviews");
      }
  
      const data = await response.json();
      setTopDissimilar(data.top_dissimilar_reviews || []);
    } catch (error) {
      console.error("Fetch dissimilar reviews error:", error);
    }
  };

  useEffect(() => {
    fetchDissimilarReviews();
  }, [bookId]);
  
  useEffect(() => {
    fetchBook();
  }, [bookId, page]);  

  return (
    <div className="min-h-screen flex gap-6 p-6 bg-gradient-to-br from-purple-300 via-purple-100 to-white">
      <div className="w-1/2 space-y-6">
        {book && (
          <div className="flex bg-white p-6 rounded-lg shadow-md items-center ml-4">
            <img
              src={book.image_url || 'https://via.placeholder.com/200x300'}
              alt={book.title}
              className="w-40 h-60 object-cover rounded-lg"
            />
            <div className="flex-1">
              <h2 className="text-xl font-bold ml-4">{book.title}</h2>
              <p className="text-gray-600 ml-4">Author: {book.author}</p>
              <p className="text-gray-600 ml-4">ISBN: {book.isbn}</p>
            </div>
          </div>
        )}
        
        <div className="bg-white p-6 rounded-lg shadow-md ml-4">
          <h3 className="text-lg font-semibold text-purple-700 mb-4">
            Top 5 Most Dissimilar Reviews
          </h3>
          <div className="space-y-4">
            {topDissimilar.length > 0 ? (
              topDissimilar.map((review: any) => (
                <div
                  key={review.review_id}
                  className="bg-gray-100 p-4 rounded shadow-sm text-gray-800 text-sm italic"
                >
                  "{review.text.length > 250 ? review.text.slice(0, 250) + '…' : review.text}"
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-400">No data available.</p>
            )}
          </div>
        </div>
      </div>
  
      <div className="w-1/2 bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-lg font-semibold text-purple-700 mb-4">All Reviews</h2>
        <div className="space-y-4">
          {reviews.map((r: any) => (
            <div key={r.review_id} className="bg-gray-100 p-4 rounded shadow-sm">
              <div className="flex justify-between items-start">
                <p className="text-gray-800 text-sm italic max-w-[85%]">"{r.text}"</p>
                <span className="text-xs font-semibold px-2 py-1 bg-purple-200 text-purple-800 rounded-full shadow-sm">
                  {r.fine_label}
                </span>
              </div>
            </div>
          ))}
        </div>
  
        <div className="flex justify-end gap-2 mt-6">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="bg-purple-500 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
          >
            Prev
          </button>
          <button
            onClick={() => setPage((p) => p + 1)}
            className="bg-purple-500 text-white px-4 py-2 rounded-lg text-sm"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
