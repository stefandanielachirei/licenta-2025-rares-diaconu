'use client';

import { useEffect, useState } from "react";
import { useParams } from 'next/navigation';

export default function BookPage() {
  const params = useParams();
  const bookId = params?.book?.[0];
  interface Book {
    title: string;
    author: string;
    isbn: string;
    image_url: string;
  }
  const [book, setBook] = useState<Book | null>(null);
  const [reviews, setReviews] = useState([]);
  const [page, setPage] = useState(1);
  const itemsPerPage = 4;
  var email = '';

  const fetchBook = async () => {
    try {
      const token = window.localStorage.getItem("token");
      if (!token) throw new Error("Authentication token is missing");

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
      email = userInfo.username;

      const bookResponse = await fetch(`http://localhost:8000/books/${bookId}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      if (!bookResponse.ok) throw new Error("Failed to fetch book details");
      const bookData = await bookResponse.json();
      setBook(bookData);

      const reviewsParams = new URLSearchParams({
        user_email: email,
        page: page.toString(),
        items_per_page: itemsPerPage.toString(),
      });

      const reviewsResponse = await fetch(
        `http://localhost:8000/books/${bookId}/reviews_sentiment?${reviewsParams.toString()}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      if (!reviewsResponse.ok) throw new Error("Failed to fetch reviews");

      const reviewsData = await reviewsResponse.json();
      setReviews(reviewsData.reviews);
    } catch (error) {
      console.error(error);
    }
  };

  useEffect(() => {
    if (bookId && email) {
      fetchBook();
    }
  }, [bookId, email, page]);

  return (
    <div className="flex gap-6 p-6">
      <div className="w-3/4 space-y-6">
        {book && (
          <div className="flex bg-white p-6 rounded-lg shadow-md items-center max-w-[900px] gap-6">
            <img
              src={book.image_url || 'https://via.placeholder.com/200x300'}
              alt={book.title}
              className="w-40 h-60 object-cover rounded-lg"
            />
            <div className="flex-1">
              <h2 className="text-xl font-bold">{book.title}</h2>
              <p className="text-gray-600">Author: {book.author}</p>
              <p className="text-gray-600">ISBN: {book.isbn}</p>
            </div>
          </div>
        )}

        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-lg font-semibold text-purple-700 mb-4">All Reviews</h2>
          <div className="space-y-4">
            {reviews.map((r: any) => (
              <div key={r.review_id} className="bg-gray-100 p-4 rounded shadow-sm">
                <div className="flex justify-between items-start">
                  <p className="text-gray-800 text-sm italic max-w-[85%]">“{r.text}”</p>
                  <span className="text-xs px-2 py-1 bg-purple-100 text-purple-700 rounded-full">
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

      <div className="w-1/4 bg-white p-4 rounded-lg shadow-md h-fit">
        <p className="text-sm text-gray-500">Placeholder for extra content</p>
      </div>
    </div>
  );
}
