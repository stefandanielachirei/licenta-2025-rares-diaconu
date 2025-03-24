"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import "./styles.css";

const UserDashboard = () => {
  const [activePage, setActivePage] = useState("all_books");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessageChangePassword, setSuccessChangePassword] = useState<string | null>(null);
  const [formDataChangePassword, setFormDataChangePassword] = useState({
    currentPassword: "",
    newPassword: "",
  });
  const router = useRouter();

  const handleInputChangePassword = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormDataChangePassword({ ...formDataChangePassword, [e.target.name]: e.target.value });
  };

  const changePage = (page: string) => {
    setActivePage(page);
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

      const userInfo = await validateResponse.json()

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
            return <h1 className="text-2xl font-bold">Books</h1>;         
      case "to_read":
        return <h1 className="text-2xl font-bold">To Read Books</h1>;
      case "read":
        return <h1 className="text-2xl font-bold">Read Books</h1>;
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
                activePage === "books" ? "bg-purple-200 text-purple-700" : "hover:bg-purple-200 hover:text-purple-700"
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