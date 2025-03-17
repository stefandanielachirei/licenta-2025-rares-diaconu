"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { useSearchParams } from "next/navigation";
import "./styles.css";

const UpdateBookPage = () => {
  const [formData, setFormData] = useState({
    title: "",
    author: "",
    isbn: "",
    goodreads_id: "",
    image_url: ""
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();
  const bookId = searchParams.get('id');

  const handleInputChange = (e:any) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const isValidISBN:any = (isbn: string) => {
    const pattern = /^[0-9Xx-]+$/;
    return pattern.test(isbn);
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    if (!formData.title.trim()) {
        setError("Title is required.");
        setLoading(false);
        return;
    }
  
    if (!formData.author.trim()) {
        setError("Author is required.");
        setLoading(false);
        return;
    }

    if (!formData.isbn.trim() || !isValidISBN(formData.isbn)) {
        setError("Invalid ISBN format.");
        setLoading(false);
        return;
    }

    if (!formData.goodreads_id.trim()) {
        setError("Goodreads ID is required.");
        setLoading(false);
        return;
    }

    try {
        const token = window.localStorage.getItem("token");
        if (!token) {
          throw new Error("Authentication token is missing");
        }
  
        const response = await fetch(`http://localhost:8000/books/${bookId}`, {
            method: "GET",
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });
  
        if (!response.ok) {
          throw new Error("Failed to update book.");
        }
  
        const existingData = await response.json();
  
        const isDataUnchanged =
          formData.title === existingData.title &&
          formData.author === existingData.author &&
          Number(formData.isbn) === Number(existingData.isbn) &&
          Number(formData.goodreads_id) === Number(existingData.goodreads_id) &&
          formData.image_url === existingData.image_url
  
        if (isDataUnchanged) {
          alert("No changes detected. Update not performed.");
          setLoading(false);
          return;
        }
  
        const putResponse = await fetch(`http://localhost:8000/books/${bookId}`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(formData),
        });
  
        if (!putResponse.ok) {
          throw new Error(`Failed to update book: ${putResponse.statusText}`);
        }
  
        alert("Book updated successfully!");
        router.push("/admin");
      } catch (err:any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
  

  return (
    <div className="min-h-screen w-screen bg-gradient-to-b from-blue-100 to-indigo-100 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-lg p-6 w-80">
        <h2 className="text-indigo-700 text-center text-2xl font-bold mb-6">Update Book</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Title</label>
            <input type="text" name="title" value={formData.title} onChange={handleInputChange} className="w-full border-2 border-gray-300 rounded-lg px-3 py-2 text-base focus:outline-none focus:ring-2 focus:ring-indigo-600 transition" required />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Author</label>
            <input type="text" name="author" value={formData.author} onChange={handleInputChange} className="w-full border-2 border-gray-300 rounded-lg px-3 py-2 text-base focus:outline-none focus:ring-2 focus:ring-indigo-600 transition" required />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">ISBN</label>
            <input type="text" name="isbn" value={formData.isbn} onChange={handleInputChange} className="w-full border-2 border-gray-300 rounded-lg px-3 py-2 text-base focus:outline-none focus:ring-2 focus:ring-indigo-600 transition" required />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Goodreads ID</label>
            <input type="text" name="goodreads_id" value={formData.goodreads_id} onChange={handleInputChange} className="w-full border-2 border-gray-300 rounded-lg px-3 py-2 text-base focus:outline-none focus:ring-2 focus:ring-indigo-600 transition" required />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Image URL</label>
            <input type="text" name="image_url" value={formData.image_url} onChange={handleInputChange} className="w-full border-2 border-gray-300 rounded-lg px-3 py-2 text-base focus:outline-none focus:ring-2 focus:ring-indigo-600 transition" required />
          </div>
          <button type="submit" className={`w-full bg-indigo-700 text-white rounded-lg px-4 py-2 font-medium hover:bg-indigo-800 transition ${loading ? "opacity-50 cursor-not-allowed" : ""}`} disabled={loading}>
            {loading ? "Updating..." : "Update Book"}
          </button>
          {error && <p className="text-red-500 mt-4 text-center">{error}</p>}
        </form>
      </div>
    </div>
  );
};

export default UpdateBookPage;
