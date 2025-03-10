"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import './globals.css';

const LoginComponent = () => {
  const router = useRouter();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  useEffect(() => {
    document.title = "Moodle";
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      console.log("Starting fetch...");
      const response = await fetch("http://localhost:8080/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          username: email,
          password: password,
        }),
      });

      if (!response.ok) {
        if (response.status === 401 || response.status === 500) {
          throw new Error("Invalid email or password.");
        } else if (response.status === 403) {
          throw new Error("Access denied.");
        }
      }

      const data = await response.json();
      console.log("Login successful:", data);

      if (data.token) {
        window.localStorage.setItem("token", data.token);

        const validateResponse = await fetch("http://localhost:8080/validate", {
          method: "GET",
          headers: {
            Authorization: `Bearer ${data.token}`,
          },
        });

        if (!validateResponse.ok) {
          throw new Error("Token validation failed");
        }

        const userInfo = await validateResponse.json();
        const role = userInfo.role;
        const userEmail = userInfo.username;

        if(userEmail === 'admin@gmail.com'){
          router.push("/admin");
        }
        else{
          const idResponse = await fetch(`http://localhost:8000/api/academia?email=${userEmail}`, {
            method: 'GET',
            headers: {
              Authorization: `Bearer ${data.token}`,
            },
          });
          if(!idResponse.ok){
            throw new Error("Failed to retrieve user id.");
          }
  
          const { id } = await idResponse.json();
          console.log("User ID:", id);
  
          if (role === "teacher") {
            router.push(`/teacher/${id}`);
          } else if (role === "student") {
            router.push(`/student/${id}`);
          } else {
            alert("Invalid role");
            console.error("Invalid role");
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        if (err.name === "TypeError") {
          setError("Unable to connect to the server. Please try again later.");
        } else {
          setError(err.message || "An unexpected error occurred.");
        }
      } else {
        setError("An unknown error occurred.");
      }
    }
  };

  return (
    <div className="h-screen w-full flex items-center justify-center bg-gradient-to-b from-purple-700 to-blue-500">
      <div className="w-full max-w-sm bg-white bg-opacity-90 rounded-xl p-6 shadow-lg border-4 border-blue-400">
        <h1 className="text-3xl font-bold text-purple-700 text-center mb-6">Login</h1>
        {error && <p className="text-red-500 text-center mb-4">{error}</p>}
        <form onSubmit={handleLogin}>
          <div className="mb-4">
            <label htmlFor="email" className="block text-gray-700 font-medium mb-2">
              Email
            </label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-400 focus:outline-none transition"
              placeholder="Enter your email"
              required
            />
          </div>
          <div className="mb-6">
            <label htmlFor="password" className="block text-gray-700 font-medium mb-2">
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-400 focus:outline-none transition"
              placeholder="Enter your password"
              required
            />
          </div>
          <button
            type="submit"
            className="w-full py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold rounded-lg hover:from-purple-600 hover:to-blue-500 focus:outline-none transition-transform transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={loading}
          >
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginComponent;
