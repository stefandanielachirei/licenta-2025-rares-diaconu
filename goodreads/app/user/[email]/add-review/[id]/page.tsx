"use client";

import { useParams } from "next/navigation";

export default function AddReviewPage() {
  const params = useParams();
  const bookId = params?.id;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Add Review for Book ID: {bookId}</h1>
      {/* Form for review goes here */}
    </div>
  );
}
