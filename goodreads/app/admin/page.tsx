"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import "./styles.css";

const AdminDashboard = () => {
  const [activePage, setActivePage] = useState("books");
  const router = useRouter();
  const [books, setBooks] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(2);

  const changePage = (page: string) => {
    setActivePage(page);
  };

  useEffect(() => {
    if (activePage === "books") {
      fetchBooks();
    }
  }, [activePage, currentPage, itemsPerPage]);

  const fetchBooks = async () => {
    setLoading(true);
    setError(null);
    try {
        const token = window.localStorage.getItem("token");
        if(!token){
            throw new Error("Authentication token is missing");
        }
        const response = await fetch(`http://localhost:8000/books?page=${currentPage}&items_per_page=${itemsPerPage}`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${token}`,
            },
        });
        if(!response.ok){
            throw new Error(`Failed to fetch students: ${response.statusText}`);
        }
        const data = await response.json();
        setBooks(data.books);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const deleteBook = async(bookId: number) => {
    try{
      const token = window.localStorage.getItem("token");
      if(!token){
        throw new Error("Authentication token is missing");
      }

      const professor = books.find((book: any) => book.id === bookId);
      if (!professor) {
        throw new Error("Book not found");
      }

      const response = await fetch(`http://localhost:8000/books/${bookId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if(!response.ok){
        throw new Error(`Failed to delete book: ${response.statusText}`);
      }

      setBooks((prev) => prev.filter((book:any) => book.id !== bookId));
      
    }catch (err: any) {
      alert(err.message);
    }
  }

  const handleLogout = async () => {
    try {
      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("No token found. Please log in again");
      }

      const response = await fetch(`http://localhost:8080/logout`, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
        },
      });

      if(!response.ok){
        throw new Error("Failed to log out");
      }


      const data = await response.json();
      alert(data.message || "Successfully logged out.");

      window.localStorage.removeItem("token");
      router.push(`/`);

    }catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  const renderContent = () => {
    switch (activePage) {
        case "books":
            return (
              <div className="p-8">
                <h1 className="text-2xl font-bold mb-4">Manage Books</h1>
                {loading && <p className="text-blue-500">Loading...</p>}
                {error && <p className="text-red-500">Error: {error}</p>}
                {!loading && books.length > 0 && (
                  <>
                    <table className="table-auto w-full border-collapse border border-gray-300">
                      <thead>
                        <tr>
                          <th className="border border-gray-300 px-4 py-2">ID</th>
                          <th className="border border-gray-300 px-4 py-2">Title</th>
                          <th className="border border-gray-300 px-4 py-2">Author</th>
                          <th className="border border-gray-300 px-4 py-2">ISBN</th>
                          <th className="border border-gray-300 px-4 py-2">Image_URL</th>
                          <th className="border border-gray-300 px-4 py-2">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {books.map((book: any, index) => (
                          <tr key={index}>
                            <td className="border border-gray-300 px-4 py-2">{book.id}</td>
                            <td className="border border-gray-300 px-4 py-2">{book.title}</td>
                            <td className="border border-gray-300 px-4 py-2">{book.author}</td>
                            <td className="border border-gray-300 px-4 py-2">{book.isbn}</td>
                            <td className="border border-gray-300 px-4 py-2">{book.image_url}</td>
                            <td className="border border-gray-300 px-4 py-2">
                                <div className="flex gap-2">
                                    <button 
                                        onClick={() => router.push(`admin/book_update?id=${book.id}`)}
                                        className="bg-blue-500 text-white px-2 py-1 rounded hover:bg-blue-700 transition">
                                        Edit
                                    </button>
                                    <button
                                        onClick={() => deleteBook(book.id)}
                                        className="bg-red-500 text-white px-2 py-1 rounded hover:bg-red-700 transition"
                                        >
                                        Delete
                                    </button>
                                </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
          
                    <div className="flex items-center justify-center gap-4 mt-4">
                        <button
                        onClick={() => router.push(`admin/book_create`)}
                        className="bg-yellow-500 px-6 py-2 rounded-lg flex items-center hover:bg-purple-200 transition"
                      >
                        Create
                      </button>
                      <button
                        onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
                        className="border border-black px-6 py-2 rounded-lg flex items-center hover:bg-gray-200 transition"
                      >
                        ← PREV
                      </button>
          
                      <span className="text-lg font-semibold">Page {currentPage}</span>
          
                      <button
                        onClick={() => setCurrentPage((prev) => prev + 1)}
                        className="border border-black px-6 py-2 rounded-lg flex items-center hover:bg-pink-200 transition"
                      >
                        NEXT →
                      </button>
          
                      <div className="flex items-center border border-black px-4 py-2 rounded-lg">
                        <span className="mr-2">Items/Page:</span>
                        <select
                          value={itemsPerPage}
                          onChange={(e) => setItemsPerPage(Number(e.target.value))}
                          className="bg-transparent focus:outline-none"
                        >
                          <option value={2}>2</option>
                          <option value={4}>4</option>
                          <option value={6}>6</option>
                        </select>
                      </div>
                    </div>
                  </>
                )}
              </div>
            );          
      case "reviews":
        return <h1 className="text-2xl font-bold">Manage Reviews</h1>;
      case "deleteAccount":
        return <h1 className="text-2xl font-bold">Delete Account</h1>;
      case "logout":
        return (
          <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-b from-blue-100 to-indigo-100">
            <p className="text-lg font-semibold mb-4">Are you sure you want to log out?</p>
            <button
              onClick={handleLogout}
              className="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition"
            >
              Logout
            </button>
          </div>
        );
      default:
        return <h1 className="text-2xl font-bold">Select a page from the sidebar.</h1>;
    }
  };

  return (
    <div className="flex h-screen w-screen bg-gradient-to-b from-purple-700 to-blue-500">
      <div className="w-64 bg-white bg-opacity-90 rounded-xl p-6 shadow-xl m-4 flex flex-col justify-between">
        <div>
          <h2 className="text-2xl font-bold text-purple-700 text-center mb-6">Admin Panel</h2>
          <ul className="space-y-4">
            <li
              onClick={() => setActivePage("books")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "books" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
              }`}
            >
              Books
            </li>
            <li
              onClick={() => setActivePage("reviews")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "reviews" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
              }`}
            >
              Reviews
            </li>
            <li
              onClick={() => setActivePage("deleteAccount")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "deleteAccount"
                  ? "bg-purple-200 text-purple-700"
                  : "hover:bg-purple-200 hover:text-purple-700"
              }`}
            >
              Delete Account
            </li>
          </ul>
        </div>

        <div>
          <ul>
            <li
              onClick={() => changePage("logout")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "logout" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
              }`}
            >
              Log Out
            </li>
          </ul>
        </div>
      </div>

      <div className="flex-1 bg-white bg-opacity-90 rounded-xl p-6 shadow-xl m-4 overflow-auto">
        {renderContent()}
      </div>
    </div>
  );
};

export default AdminDashboard;
