export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-white animate-pulse">
          🚀 Tailwind CSS 3.3.3 funcționează perfect!
        </h1>
        <p className="text-white opacity-90">
          Acum poți începe dezvoltarea proiectului tău!
        </p>
        <button className="bg-white text-purple-600 px-6 py-2 rounded-lg shadow-lg hover:scale-105 transition-transform">
          Start Coding
        </button>
      </div>
    </main>
  )
}