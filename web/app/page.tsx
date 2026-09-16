const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight">TakeoffLens</h1>
      <p className="mt-3 text-neutral-600">
        Upload an architectural floor plan to extract rooms, dimensions and areas.
      </p>
      <p className="mt-8 rounded-lg border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-500">
        Phase 0 scaffold. Upload and the plan viewer arrive in Phase 5.
        API: <code className="font-mono">{API_URL}</code>
      </p>
    </main>
  );
}
