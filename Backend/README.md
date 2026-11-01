# Agentic RAG Jurisprudence API

## Run locally

From the `Backend` directory:

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --reload
```

The API documentation is available at `http://localhost:8000/docs`. By default, CORS allows
frontends on ports `3000` and `5173`. Set `FRONTEND_ORIGINS` to a comma-separated list to
configure other origins.

## Endpoints

- `GET /health` — health check.
- `POST /api/sessions` — create an in-memory session; returns `session_id`.
- `GET /api/sessions/{session_id}` — read the session's current status, selected cases, and
  latest answer and sources.
- `POST /api/sessions/{session_id}/messages` — send a message and receive progress as
  Server-Sent Events. Optional `selected_method` can be `TF-IDF`, `BM25`, `Semantic`, or
  `Hybrid`.
- `DELETE /api/sessions/{session_id}` — delete an idle session.

Example request body:

```json
{
  "message": "Je cherche des affaires concernant la résiliation d'un contrat de travail",
  "selected_method": "Hybrid"
}
```

Each message response is `text/event-stream`. Events are named `progress`, `completed`, or
`error`; their `data` is JSON. Progress events expose the current step, retrieval method,
fallback status, selected cases, conversational response, or QA citations. Greetings and other
messages that do not request a case search are handled as conversation and do not start retrieval.
Internal LLM reasoning is not sent to clients.

Example browser call (the POST stream is consumed with `fetch`, not native `EventSource`):

```javascript
const { session_id: sessionId } = await fetch("http://localhost:8000/api/sessions", {
  method: "POST",
}).then((response) => response.json());

const response = await fetch(
  `http://localhost:8000/api/sessions/${sessionId}/messages`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: "Je cherche des affaires concernant un contrat de travail",
      selected_method: "Hybrid",
    }),
  },
);

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const events = buffer.split("\n\n");
  buffer = events.pop() ?? "";
  for (const event of events) {
    const name = event.match(/^event: (.+)$/m)?.[1];
    const data = event.match(/^data: (.+)$/m)?.[1];
    if (name && data) {
      const payload = JSON.parse(data);
      console.log(name, payload);
    }
  }
}
```

Sessions are stored in process memory for this prototype. They disappear when the server
restarts and are not shared across multiple server workers. Only one message can be processed
at a time per session.