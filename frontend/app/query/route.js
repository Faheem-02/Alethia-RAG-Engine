const API_BASE_URL = "http://127.0.0.1:8000";

export async function POST(request) {
  try {
    const body = await request.json();
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });

    const data = await response.json().catch(() => ({}));
    return Response.json(data, { status: response.status });
  } catch {
    return Response.json({ detail: "Failed to reach backend /query." }, { status: 502 });
  }
}
