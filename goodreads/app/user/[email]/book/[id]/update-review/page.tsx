"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";

export default function AddReviewPage() {
  const [formData, setFormData] = useState({
    summary: "",
    review_text: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [review, setReview] = useState<any>(null);
  const router = useRouter();
  const params = useParams();
  const bookId = params.id;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const fetchReview = async () => {
    setLoading(true);
    try {
        const token = window.localStorage.getItem("token");
        if (!token) {
          throw new Error("No token found. Please log in again");
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


        const response = await fetch(`http://localhost:8000/books/${bookId}/user-review?user_email=${userInfo.username}`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${token}`,
            },
        });
        if (!response.ok) {
          if (response.status === 404) {
            setError("You haven't written a review yet, nothing to edit.");
            return;
          } else {
            throw new Error(`Failed to fetch review: ${response.statusText}`);
          }
        }
        
        const data = await response.json();
        setReview(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReview();
  }, [bookId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("No token found. Please log in again");
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

      const res = await fetch("http://localhost:8000/update_review", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          book_id: parseInt(bookId as string),
          user_email: userInfo.username,
          review_text: formData.review_text,
          summary: formData.summary.trim() === "" ? null : formData.summary,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to add review");
      }

      setSuccessMessage(`${data.message}`);
      setFormData({ summary: "", review_text: "" });

      setTimeout(() => {
        router.push(`/user/${userInfo.username}`);
      }, 2000);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-screen bg-gradient-to-b from-blue-100 to-indigo-100 flex flex-col items-center justify-center px-4 py-12">
      <div
        onClick={() => router.back()}
        className="absolute top-6 left-6 cursor-pointer flex items-center space-x-2 bg-white px-4 py-2 rounded-lg shadow hover:bg-gray-100 transition"
      >
        <span className="text-xl">←</span>
        <span className="text-indigo-700 font-semibold">Back</span>
      </div>

      <div className="flex flex-col lg:flex-row gap-10">
        <div className="bg-white rounded-lg shadow-lg p-8 w-[500px]">
          <h2 className="text-indigo-700 text-center text-2xl font-bold mb-4">
            Current Review
          </h2>
          <div className="text-gray-700 whitespace-pre-wrap">
            {loading
              ? "Loading..."
              : review
              ? (
                  <>
                    <p className="mt-2"><strong>Review:</strong> {review.review_text}</p>
                  </>
                )
              : "No review available"}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-8 w-[500px]">
          <h2 className="text-indigo-700 text-center text-2xl font-bold mb-4">
            Update Review
          </h2>

          {successMessage && (
            <p className="text-green-600 text-center mb-4">{successMessage}</p>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-base font-medium text-gray-700 mb-1">
                Review Text
              </label>
              <textarea
                name="review_text"
                value={formData.review_text}
                onChange={handleInputChange}
                required
                className="w-full border border-gray-300 rounded px-4 py-2"
                rows={6}
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`w-full bg-indigo-700 text-white py-2 px-4 rounded hover:bg-indigo-800 transition ${
                loading && "opacity-50 cursor-not-allowed"
              }`}
            >
              {loading ? "Saving..." : "Update Review"}
            </button>

            {error && <p className="text-red-500 text-center">{error}</p>}
          </form>
        </div>
      </div>
    </div>
  );
}
