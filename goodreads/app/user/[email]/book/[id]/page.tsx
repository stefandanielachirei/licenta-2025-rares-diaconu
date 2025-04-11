'use client';

import { useParams } from 'next/navigation';

export default function BookPage() {
  const params = useParams();
  const id = params?.id;

  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold">Book ID: {id}</h1>
    </div>
  );
}
