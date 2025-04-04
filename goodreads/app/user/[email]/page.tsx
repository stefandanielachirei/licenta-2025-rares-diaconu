"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import "./styles.css";

const UserDashboard = () => {
  type BookType = {
      id: number;
      title: string;
      author: string;
      isbn: string;
      image_url: string;
    };
  const [activePage, setActivePage] = useState("all_books");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessageChangePassword, setSuccessChangePassword] = useState<string | null>(null);
  const [formDataChangePassword, setFormDataChangePassword] = useState({
    currentPassword: "",
    newPassword: "",
  });
  const [books, setBooks] = useState([]);
  const router = useRouter();
  const [title, setTitle] = useState("");
  const [author, setAuthor] = useState("");
  const [isbn, setIsbn] = useState("");
  const [pageAllBooks, setPageAllBooks] = useState(1);
  const [pageToReadBooks, setPageToReadBooks] = useState(1);
  const [pageReadBooks, setPageReadBooks] = useState(1);
  const [totalAllBooks, setTotalAllBooks] = useState(0);
  const [totalToReadBooks, setTotalToReadBooks] = useState(0);
  const [totalReadBooks, setTotalReadBooks] = useState(0);
  const [toReadBooks, setToReadBooks] = useState<BookType[]>([]);
  const [ReadBooks, setReadBooks] = useState<BookType[]>([]);
  const itemsPerPageAllBooks = 2;
  const itemsPerPageToReadBooks = 4;
  const itemsPerPageReadBooks = 4;

  const handleInputChangePassword = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormDataChangePassword({ ...formDataChangePassword, [e.target.name]: e.target.value });
  };

  const changePage = (page: string) => {
    setActivePage(page);
  };

  useEffect(() => {
    localStorage.setItem("activePage", activePage);
  }, [activePage]);

  const fetchBooks = async () => {
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
  
      const params = new URLSearchParams({
        user_email: userInfo.username,
        page: pageAllBooks.toString(),
        items_per_page: itemsPerPageAllBooks.toString(),
      });
  
      if (title) params.append("title", title);
      if (author) params.append("author", author);
      if (isbn) params.append("isbn", isbn);
  
      const response = await fetch(`http://localhost:8000/books?${params.toString()}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`,
        },
      });
  
      if (!response.ok) throw new Error("Failed to fetch books");
  
      const data = await response.json();
      setBooks(data.books);
      setTotalAllBooks(data.total_books);
    } catch (error) {
      console.error(error);
    }
  };

  const handleStatusChange = async (bookId: number, newStatus: string) => {
    try {

      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token is missing.");
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
      const response = await fetch("http://localhost:8000/update_status", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json" ,
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
          user_email: userInfo.username,
          book_id: bookId,
          status: newStatus,
        }),
      });
  
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Error updating status");
  
      console.log("Status updated:", data.message);

      setBooks((prevBooks : any) =>
        prevBooks.map((book : any) =>
          book.id === bookId ? { ...book, status: newStatus } : book
        )
      );
    } catch (error) {
      console.error("Error updating status:", error);
    }
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
      if (!page) {
        setActivePage("all_books");
        localStorage.setItem("activePage", "all_books");
      } else {
        setActivePage(page);
      }
    }, []);

  useEffect(() => {
    if(activePage == "all_books"){
      fetchBooks();
    }
  }, [activePage, pageAllBooks]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPageAllBooks(1);
    fetchBooks();
  };

  useEffect(() => {
    if(activePage == "to_read"){
      fetchToReadBooks();
    } 
  }, [activePage, pageToReadBooks]);

  const fetchToReadBooks = async () => {
    try {
      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token is missing.");
      }

      const validateResponse = await fetch("http://localhost:8080/validate", {
        method: "GET",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!validateResponse.ok) {
        throw new Error("Token validation failed");
      }

      const userInfo = await validateResponse.json();

      const response = await fetch(`http://localhost:8000/to_read_books/${userInfo.username}?page=${pageToReadBooks}&per_page=${itemsPerPageToReadBooks}`, {
        method: "GET",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!response.ok) throw new Error("Failed to fetch To Read books");

      const data = await response.json();
      setToReadBooks(data.books);
      setTotalToReadBooks(data.total_books);
    } catch (error) {
      console.error("Error fetching To Read books:", error);
    }
  };

  useEffect(() => {
    if(activePage == "to_read"){
      fetchReadBooks();
    } 
  }, [activePage, pageToReadBooks]);

  const fetchReadBooks = async () => {
    try {
      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token is missing.");
      }

      const validateResponse = await fetch("http://localhost:8080/validate", {
        method: "GET",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!validateResponse.ok) {
        throw new Error("Token validation failed");
      }

      const userInfo = await validateResponse.json();

      const response = await fetch(`http://localhost:8000/read_books/${userInfo.username}?page=${pageReadBooks}&per_page=${itemsPerPageReadBooks}`, {
        method: "GET",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!response.ok) throw new Error("Failed to fetch Read books");

      const data = await response.json();
      setReadBooks(data.books);
      setTotalReadBooks(data.total_books);
    } catch (error) {
      console.error("Error fetching Read books:", error);
    }
  };

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formDataChangePassword.newPassword || formDataChangePassword.newPassword.length < 8) {
      setError("Password must be at least 8 characters long.");
      setLoading(false);
      return;
    }
  
    try {
      const token = window.localStorage.getItem("token");
      if (!token) {
        throw new Error("Authentication token is missing.");
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
  
      const changePasswordResponse = await fetch(
        "http://localhost:8080/changePassword",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({
            username: userInfo.username,
            current_password: formDataChangePassword.currentPassword,
            new_password: formDataChangePassword.newPassword,
          }),
        }
      );
  
      if (!changePasswordResponse.ok) {
        const errorData = await changePasswordResponse.json();
        throw new Error(errorData.detail || "Failed to change password.");
      }
  
      setSuccessChangePassword("Password changed successfully!");
      setFormDataChangePassword({ currentPassword: "", newPassword: "" });
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    }
  };

  const deleteAccount = async() => {

    try{
      const token = window.localStorage.getItem("token");
      if(!token){
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

      const response = await fetch(`http://localhost:8000/deleteUser?email=${userInfo.username}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if(!response.ok){
        throw new Error(`Failed to delete account: ${response.statusText}`);
      }

      const responseIDM = await fetch("http://localhost:8080/deleteUser", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          username: userInfo.username
        }),
      });

      if(!responseIDM.ok){
        throw new Error(`Failed to delete IDM user: ${responseIDM.statusText}`);
      }
      
      alert("Account deleted successfully!");
      window.localStorage.removeItem("token");
      router.push("/");

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
      case "all_books":
        return (
          <div>
            <form onSubmit={handleSearch} className="mb-4 flex flex-wrap gap-2">
              <input
                type="text"
                placeholder="Title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="px-3 py-2 border rounded-lg"
              />
              <input
                type="text"
                placeholder="Author"
                value={author}
                onChange={(e) => setAuthor(e.target.value)}
                className="px-3 py-2 border rounded-lg"
              />
              <input
                type="text"
                placeholder="ISBN"
                value={isbn}
                onChange={(e) => setIsbn(e.target.value)}
                className="px-3 py-2 border rounded-lg"
              />
              <button
                type="submit"
                className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600"
              >
                Search
              </button>
            </form>

            <div className="flex flex-col gap-6">
              {books.map((book: any) => (
                <div
                  key={book.id}
                  className="flex bg-white p-6 rounded-lg shadow-md items-center w-full justify-between"
                >
                  <div className="flex items-center gap-6 w-full">
                    <img
                      src={book.image_url || "https://via.placeholder.com/150x200"}
                      alt={book.title}
                      className="w-32 h-48 object-cover rounded-lg"
                    />
                    <div className="flex-1">
                    <div className="flex items-center gap-4">
                      <h2 className="text-xl font-bold">{book.title}</h2>
                      <select
                        value={book.status}
                        onChange={(e) => handleStatusChange(book.id, e.target.value)}
                        className="px-3 py-2 border rounded-lg bg-gray-100 text-sm"
                      >
                        <option value="none">None</option>
                        <option value="read">Read</option>
                        <option value="to_read">To Read</option>
                      </select>
                    </div>
                      <p className="text-gray-600">Author: {book.author}</p>
                      <p className="text-gray-600">ISBN: {book.isbn}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 flex justify-between">
              <button
                onClick={() => setPageAllBooks((prev) => Math.max(1, prev - 1))}
                disabled={pageAllBooks === 1}
                className="bg-purple-500 text-white px-4 py-2 rounded-lg disabled:opacity-50"
              >
                Prev
              </button>
              <button
                onClick={() =>
                  setPageAllBooks((prev) => (prev * itemsPerPageAllBooks < totalAllBooks ? prev + 1 : prev))
                }
                disabled={pageAllBooks * itemsPerPageAllBooks >= totalAllBooks}
                className="bg-purple-500 text-white px-4 py-2 rounded-lg disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )     
      case "to_read":
        return (
          <div>
            <div className="flex flex-col">
              <div className="flex flex-wrap gap-6 justify-center">
                {toReadBooks.map((book: any) => (
                  <div
                    key={book.id}
                    className="flex bg-white p-6 rounded-lg shadow-md items-center w-full max-w-[600px] justify-between"
                  >
                    <img
                      src={book.image_url || "https://via.placeholder.com/200x300"}
                      alt={book.title}
                      className="w-40 h-60 object-cover rounded-lg"
                    />
                    <div className="flex-1 ml-4">
                      <h2 className="text-xl font-bold">{book.title}</h2>
                      <p className="text-gray-600">Author: {book.author}</p>
                      <p className="text-gray-600">ISBN: {book.isbn}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="flex justify-between p-4 mt-4">
                <button
                  onClick={() => setPageToReadBooks((prev) => Math.max(1, prev - 1))}
                  disabled={pageToReadBooks === 1}
                  className="bg-purple-500 text-white px-4 py-2 rounded-lg disabled:opacity-50"
                >
                  Prev
                </button>
                <button
                  onClick={() =>
                    setPageToReadBooks((prev) => (prev * itemsPerPageToReadBooks < totalToReadBooks ? prev + 1 : prev))
                  }
                  disabled={pageToReadBooks * itemsPerPageToReadBooks >= totalToReadBooks}
                  className="bg-purple-500 text-white px-4 py-2 rounded-lg disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        )
      case "read":
        return (
          <div>
            <div className="flex flex-col">
              <div className="flex flex-wrap gap-6 justify-center">
                {ReadBooks.map((book: any) => (
                  <div
                    key={book.id}
                    className="flex flex-col bg-white p-6 rounded-lg shadow-md items-center w-full max-w-[400px]"
                  >
                    <img
                      src={book.image_url || "https://via.placeholder.com/200x300"}
                      alt={book.title}
                      className="w-40 h-60 object-cover rounded-lg mb-4"
                    />
                    <div className="text-center mb-4">
                      <h2 className="text-xl font-bold">{book.title}</h2>
                      <p className="text-gray-600">Author: {book.author}</p>
                      <p className="text-gray-600">ISBN: {book.isbn}</p>
                    </div>
                    <button
                      onClick={() => router.push(`user/add-review/${book.id}`)}
                      className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg"
                    >
                      Add Review
                    </button>
                  </div>
                ))}
              </div>

              <div className="flex justify-between p-4 mt-4">
                <button
                  onClick={() => setPageReadBooks((prev) => Math.max(1, prev - 1))}
                  disabled={pageReadBooks === 1}
                  className="bg-purple-500 text-white px-4 py-2 rounded-lg disabled:opacity-50"
                >
                  Prev
                </button>
                <button
                  onClick={() =>
                    setPageReadBooks((prev) =>
                      prev * itemsPerPageReadBooks < totalReadBooks ? prev + 1 : prev
                    )
                  }
                  disabled={pageReadBooks * itemsPerPageReadBooks >= totalReadBooks}
                  className="bg-purple-500 text-white px-4 py-2 rounded-lg disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        )
      case "changePassword":
        return (
            <div className="flex items-center justify-center h-full p-8">
            <div className="bg-white shadow-lg rounded-lg p-8 w-full max-w-md">
              <h1 className="text-2xl font-bold mb-4 text-center">Change Password</h1>
              <form onSubmit={handlePasswordChange} className="space-y-4">
                {error && <div className="text-red-500 text-center">{error}</div>}
                {successMessageChangePassword && (
                  <div className="text-green-500 text-center">{successMessageChangePassword}</div>
                )}
                <div>
                  <label className="block text-sm font-medium mb-1" htmlFor="currentPassword">
                    Current Password
                  </label>
                  <input
                    type="password"
                    id="currentPassword"
                    name="currentPassword"
                    value={formDataChangePassword.currentPassword}
                    onChange={handleInputChangePassword}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring focus:ring-blue-200"
                    required
                  />
                </div>
                <div>
                  <label
                    className="block text-sm font-medium mb-1"
                    htmlFor="newPassword"
                  >
                    New Password
                  </label>
                  <input
                    type="password"
                    id="newPassword"
                    name="newPassword"
                    value={formDataChangePassword.newPassword}
                    onChange={handleInputChangePassword}
                    className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring focus:ring-blue-200"
                    required
                  />
                </div>
                <button
                  type="submit"
                  className={`w-full px-4 py-2 rounded-lg text-white ${
                    loading
                      ? "bg-gray-400 cursor-not-allowed"
                      : "bg-blue-500 hover:bg-blue-600"
                  }`}
                  disabled={loading}
                >
                  {loading ? "Changing..." : "Change Password"}
                </button>
              </form>
            </div>
          </div>
        )
      case "deleteAccount":
        return (
            <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-b from-blue-100 to-indigo-100">
              <p className="text-lg font-semibold mb-4">Are you sure you want to delete your account?</p>
              <button
                onClick={deleteAccount}
                className="bg-red-500 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition"
              >
                Delete Account
              </button>
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
          <h2 className="text-2xl font-bold text-purple-700 text-center mb-6">User Panel</h2>
          <ul className="space-y-4">
            <li
              onClick={() => setActivePage("all_books")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "all_books" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
              }`}
            >
              All Books
            </li>
            <li
              onClick={() => setActivePage("to_read")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "to_read" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
              }`}
            >
              To Read
            </li>
            <li
              onClick={() => setActivePage("read")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "read" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
              }`}
            >
              Read
            </li>
            <li
              onClick={() => setActivePage("changePassword")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "changePassword" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
              }`}
            >
              Change Password
            </li>
            <li
              onClick={() => setActivePage("deleteAccount")}
              className={`px-4 py-2 rounded-lg cursor-pointer transition-colors ${
                activePage === "deleteAccount" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
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

export default UserDashboard;