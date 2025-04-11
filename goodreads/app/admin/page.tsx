"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import "./styles.css";

const AdminDashboard = () => {
  const [activePage, setActivePage] = useState("books");
  const [books, setBooks] = useState<any[]>([]);
  const [reviews, setReviews] = useState<any[]>([]);
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPageBooks, setCurrentPageBooks] = useState(1);
  const [itemsPerPageBooks, setItemsPerPageBooks] = useState(2);
  const [currentPageReviews, setCurrentPageReviews] = useState(1);
  const [itemsPerPageReviews, setItemsPerPageReviews] = useState(2);
  const [currentPageUsers, setCurrentPageUsers] = useState(1);
  const [itemsPerPageUsers, setItemsPerPageUsers] = useState(2);
  const router = useRouter();

  const changePage = (page: string) => {
    setActivePage(page);
  };

  const getActivePageFromQuery = () => {
    if (typeof window !== "undefined") {
      const urlParams = new URLSearchParams(window.location.search);
      return urlParams.get("activePage") || localStorage.getItem("activePage") || "books";
    }
    return "books";
  };

  useEffect(() => {
    const page = getActivePageFromQuery();
    setActivePage(page);
    window.localStorage.setItem("activePage", page);
  }, []);
  
  useEffect(() => {
    if (activePage === "books") {
      fetchBooks();
    }
  }, [activePage, currentPageBooks, itemsPerPageBooks]);
  
  useEffect(() => {
    if (activePage === "reviews") {
      fetchReviews();
    }
  }, [activePage, currentPageReviews, itemsPerPageReviews]);
  
  useEffect(() => {
    if (activePage === "deleteAccount") {
      fetchUsers();
    }
  }, [activePage, currentPageUsers, itemsPerPageUsers]);

  const fetchBooks = async () => {
    setLoading(true);
    setError(null);
    try {
        const token = window.localStorage.getItem("token");
        if(!token){
            throw new Error("Authentication token is missing");
        }
        const response = await fetch(`http://localhost:8000/books_admin?page=${currentPageBooks}&items_per_page=${itemsPerPageBooks}`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${token}`,
            },
        });
        if(!response.ok){
            throw new Error(`Failed to fetch books: ${response.statusText}`);
        }
        const data = await response.json();
        setBooks(data.books);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchReviews = async () => {
    setLoading(true);
    setError(null);
    try {
      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token is missing");
      }
      const response = await fetch(`http://localhost:8000/reviews?page=${currentPageReviews}&items_per_page=${itemsPerPageReviews}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch reviews: ${response.statusText}`);
      }
      const data = await response.json();
      setReviews(data.reviews);
    } catch (err : any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchUsers = async () => {
    setLoading(true);
    setError(null);
    try {
        const token = window.localStorage.getItem("token");
        if(!token){
            throw new Error("Authentication token is missing");
        }
        const response = await fetch(`http://localhost:8000/users?page=${currentPageUsers}&items_per_page=${itemsPerPageUsers}`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${token}`,
            },
        });
        if(!response.ok){
            throw new Error(`Failed to fetch users: ${response.statusText}`);
        }
        const data = await response.json();
        setUsers(data.users);
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

      const book = books.find((book: any) => book.id === bookId);
      if (!book) {
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

  const deleteReview = async(reviewId: number) => {
    try{
      const token = window.localStorage.getItem("token");
      if(!token){
        throw new Error("Authentication token is missing");
      }

      const review = books.find((review: any) => review.id === reviewId);
      if (!review) {
        throw new Error("Review not found");
      }

      const response = await fetch(`http://localhost:8000/reviews/${reviewId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if(!response.ok){
        throw new Error(`Failed to delete review: ${response.statusText}`);
      }

      setReviews((prev) => prev.filter((review:any) => review.id !== reviewId));
      
    }catch (err: any) {
      alert(err.message);
    }
  }

  const deleteUser = async(email: string) => {

    if (email === "anonim@gmail.com") {
      alert("This user cannot be deleted!");
      return;
    }

    const confirmDelete = window.confirm(`Are you sure you want to delete ${email}?`);
    if (!confirmDelete) return;

    try{
      const token = window.localStorage.getItem("token");
      if(!token){
        throw new Error("Authentication token is missing");
      }

      const user = users.find((user: any) => user.email === email);
      if (!user) {
        throw new Error("User not found");
      }

      const response = await fetch(`http://localhost:8000/deleteUser?email=${email}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if(!response.ok){
        throw new Error(`Failed to delete user: ${response.statusText}`);
      }

      const responseIDM = await fetch("http://localhost:8080/deleteUser", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          username: user.email
        }),
      });

      if(!responseIDM.ok){
        throw new Error(`Failed to delete IDM user: ${responseIDM.statusText}`);
      }

      setUsers((prev) => prev.filter((user:any) => user.email !== email));
      
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
      window.localStorage.removeItem("activePage");
      setActivePage("books");
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
                        onClick={() => setCurrentPageBooks((prev) => Math.max(prev - 1, 1))}
                        className="border border-black px-6 py-2 rounded-lg flex items-center hover:bg-gray-200 transition"
                      >
                        ← PREV
                      </button>
          
                      <span className="text-lg font-semibold">Page {currentPageBooks}</span>
          
                      <button
                        onClick={() => setCurrentPageBooks((prev) => prev + 1)}
                        className="border border-black px-6 py-2 rounded-lg flex items-center hover:bg-pink-200 transition"
                      >
                        NEXT →
                      </button>
          
                      <div className="flex items-center border border-black px-4 py-2 rounded-lg">
                        <span className="mr-2">Items/Page:</span>
                        <select
                          value={itemsPerPageBooks}
                          onChange={(e) => setItemsPerPageBooks(Number(e.target.value))}
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
        return (
          <div className="p-8">
            <h1 className="text-2xl font-bold mb-4">Manage Reviews</h1>
            {loading && <p className="text-blue-500">Loading...</p>}
            {error && <p className="text-red-500">Error: {error}</p>}
            {!loading && reviews.length > 0 && (
              <>
                <table className="table-auto w-full border-collapse border border-gray-300">
                  <thead>
                    <tr>
                      <th className="border border-gray-300 px-4 py-2">ID</th>
                      <th className="border border-gray-300 px-4 py-2">Book ID</th>
                      <th className="border border-gray-300 px-4 py-2">User Email</th>
                      <th className="border border-gray-300 px-4 py-2">Review Text</th>
                      <th className="border border-gray-300 px-4 py-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reviews.map((review : any, index) => (
                      <tr key={index}>
                        <td className="border border-gray-300 px-4 py-2">{review.id}</td>
                        <td className="border border-gray-300 px-4 py-2">{review.book_id}</td>
                        <td className="border border-gray-300 px-4 py-2">{review.user_email}</td>
                        <td className="border border-gray-300 px-4 py-2">{review.review_text}</td>
                        <td className="border border-gray-300 px-4 py-2">
                          <div className="flex gap-2">
                            <button
                              onClick={() => deleteReview(review.id)}
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
                    onClick={() => setCurrentPageReviews((prev) => Math.max(prev - 1, 1))}
                    className="border border-black px-6 py-2 rounded-lg flex items-center hover:bg-gray-200 transition"
                  >
                    ← PREV
                  </button>
                  <span className="text-lg font-semibold">Page {currentPageReviews}</span>
                  <button
                    onClick={() => setCurrentPageReviews((prev) => prev + 1)}
                    className="border border-black px-6 py-2 rounded-lg flex items-center hover:bg-pink-200 transition"
                  >
                    NEXT →
                  </button>
                  <div className="flex items-center border border-black px-4 py-2 rounded-lg">
                    <span className="mr-2">Items/Page:</span>
                    <select
                      value={itemsPerPageReviews}
                      onChange={(e) => setItemsPerPageReviews(Number(e.target.value))}
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
      case "deleteAccount":
        return (
          <div className="p-8">
            <h1 className="text-2xl font-bold mb-4">Manage Users</h1>
            {loading && <p className="text-blue-500">Loading...</p>}
            {error && <p className="text-red-500">Error: {error}</p>}
            {!loading && users.length > 0 && (
              <>
                <table className="table-auto w-full border-collapse border border-gray-300">
                  <thead>
                    <tr>
                      <th className="border border-gray-300 px-4 py-2">Email</th>
                      <th className="border border-gray-300 px-4 py-2">Role</th>
                      <th className="border border-gray-300 px-4 py-2">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {users.map((user: any, index) => (
                      <tr key={index}>
                        <td className="border border-gray-300 px-4 py-2">{user.email}</td>
                        <td className="border border-gray-300 px-4 py-2">{user.role}</td>
                        <td className="border border-gray-300 px-4 py-2">
                          <div className="flex gap-2">
                            {user.email !== "anonim@gmail.com" ? (
                              <button
                                onClick={() => deleteUser(user.email)}
                                className="bg-red-500 text-white px-2 py-1 rounded hover:bg-red-700 transition"
                              >
                                Delete
                              </button>
                            ) : (
                              <span className="text-gray-500">Cannot Delete</span>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="flex items-center justify-center gap-4 mt-4">
                <button
                    onClick={() => setCurrentPageUsers((prev) => Math.max(prev - 1, 1))}
                    className="border border-black px-6 py-2 rounded-lg flex items-center hover:bg-gray-200 transition"
                  >
                    ← PREV
                  </button>
                  <span className="text-lg font-semibold">Page {currentPageUsers}</span>
                  <button
                    onClick={() => setCurrentPageUsers((prev) => prev + 1)}
                    className="border border-black px-6 py-2 rounded-lg flex items-center hover:bg-pink-200 transition"
                  >
                    NEXT →
                  </button>
                  <div className="flex items-center border border-black px-4 py-2 rounded-lg">
                    <span className="mr-2">Items/Page:</span>
                    <select
                      value={itemsPerPageUsers}
                      onChange={(e) => setItemsPerPageUsers(Number(e.target.value))}
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
