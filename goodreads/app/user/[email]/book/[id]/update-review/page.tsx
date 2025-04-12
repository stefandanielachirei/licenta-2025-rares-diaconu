"use client";

import { useState } from "react";
import { useRouter, useParams } from "next/navigation";

export default function AddReviewPage() {
  const [formData, setFormData] = useState({
    summary: "",
    review_text: "",
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const router = useRouter();
  const params = useParams();
  const bookId = params.id;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

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
    <div className="min-h-screen w-screen bg-gradient-to-b from-blue-100 to-indigo-100 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-lg p-10 w-[500px]">
            <h2 className="text-indigo-700 text-center text-3xl font-bold mb-8">Update Review</h2>
            {successMessage && (
                <p className="text-green-600 text-center mb-4 font-medium">{successMessage}</p>
            )}
            <form onSubmit={handleSubmit} className="space-y-6">
            <div>
                <label className="block text-base font-medium text-gray-700 mb-1">Review Text</label>
                <textarea
                name="review_text"
                value={formData.review_text}
                onChange={handleInputChange}
                required
                className="w-full border-2 border-gray-300 rounded-lg px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-indigo-600 transition"
                rows={6}
                />
            </div>
            <button
                type="submit"
                disabled={loading}
                className={`w-full bg-indigo-700 text-white rounded-lg px-6 py-3 font-medium text-lg hover:bg-indigo-800 transition ${
                loading ? "opacity-50 cursor-not-allowed" : ""
                }`}
            >
                {loading ? "Saving..." : "Update Review"}
            </button>
            {error && <p className="text-red-500 mt-2 text-center">{error}</p>}
            </form>
        </div>
    </div>
  );
}
