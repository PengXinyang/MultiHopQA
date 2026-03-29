export async function fetchRuns() {
  const res = await fetch("/api/runs", { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch /runs");
  return await res.json();
}

export async function fetchEvents(runId, lastSeq) {
  const url = `/api/events?run_id=${encodeURIComponent(runId)}&last_seq=${encodeURIComponent(
    String(lastSeq || 0),
  )}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch /events");
  return await res.json();
}

