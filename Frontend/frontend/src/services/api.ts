import axios, { type AxiosProgressEvent } from "axios";
import {request} from "../axios";
import {
  AGENT_STATE_KEYS,
  type AgentState,
  type CaseItem,
  type MessageRequest,
  type SourceItem,
  type StreamEvent,
} from "../types/chat";

/* -------------------------------------------------------------------------- */
/* Errors                                                                     */
/* -------------------------------------------------------------------------- */

export type ApiErrorKind =
  | "validation"
  | "not_found"
  | "conflict"
  | "server"
  | "network"
  | "timeout"
  | "unknown";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status?: number;

  constructor(message: string, kind: ApiErrorKind, status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

type UnknownRecord = Record<string, unknown>;

const isRecord = (v: unknown): v is UnknownRecord =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const asString = (v: unknown): string | undefined => {
  if (typeof v === "string") return v.trim() || undefined;
  if (typeof v === "number") return String(v);
  return undefined;
};

const asNumber = (v: unknown): number | null => {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = parseFloat(v.replace(",", ".").replace("%", ""));
    return Number.isFinite(n) ? n : null;
  }
  return null;
};

const pick = (obj: UnknownRecord, keys: string[]): string | undefined => {
  for (const key of keys) {
    const value = asString(obj[key]);
    if (value) return value;
  }
  return undefined;
};

/** Extracts a readable message from a FastAPI error body ({ detail }), JSON or text. */
function readErrorDetail(data: unknown): string | undefined {
  let body = data;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      return undefined;
    }
  }
  if (isRecord(body) && "detail" in body) {
    const detail = body.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (isRecord(d) && "msg" in d ? String(d.msg) : String(d)))
        .join(" ; ");
    }
  }
  return undefined;
}

function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;

  if (axios.isCancel(err)) return new ApiError("Requête annulée.", "unknown");

  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    const detail = readErrorDetail(err.response?.data);

    if (status === 404) return new ApiError(detail ?? "Session introuvable.", "not_found", status);
    if (status === 409) {
      return new ApiError(detail ?? "Un message est déjà en cours pour cette session.", "conflict", status);
    }
    if (status === 422) {
      return new ApiError(detail ?? "Requête invalide : vérifiez les champs envoyés.", "validation", status);
    }
    if (status !== undefined && status >= 500) {
      return new ApiError(detail ?? "Une erreur est survenue côté serveur.", "server", status);
    }
    if (status !== undefined) {
      return new ApiError(detail ?? `Erreur inattendue (HTTP ${status}).`, "unknown", status);
    }
    if (err.code === "ECONNABORTED" || err.code === "ETIMEDOUT") {
      return new ApiError("Le serveur met trop de temps à répondre. Réessayez dans un instant.", "timeout");
    }
    return new ApiError(
      "Impossible de joindre le serveur. Vérifiez que l'API est démarrée et que CORS est autorisé.",
      "network",
    );
  }

  return new ApiError("Une erreur inattendue est survenue.", "unknown");
}

/* -------------------------------------------------------------------------- */
/* Sessions                                                                   */
/* -------------------------------------------------------------------------- */

export async function createSession(signal?: AbortSignal): Promise<string> {
  try {
    const { data } = await request.post<unknown>("sessions", undefined, { signal });
    const id = isRecord(data) ? pick(data, ["session_id", "id", "sessionId"]) : undefined;
    if (!id) {
      throw new ApiError("La réponse de création de session ne contient pas d'identifiant.", "unknown");
    }
    return id;
  } catch (err) {
    throw toApiError(err);
  }
}

export async function deleteSession(sessionId: string, signal?: AbortSignal): Promise<void> {
  try {
    await request.delete(`sessions/${encodeURIComponent(sessionId)}`, { signal });
  } catch (err) {
    throw toApiError(err);
  }
}

/* -------------------------------------------------------------------------- */
/* Stream protocol (SSE → StreamEvent)                                        */
/*                                                                            */
/* This is the only part tied to what `chat_controller.stream_message`        */
/* emits. It is deliberately tolerant; adjust `toStreamEvents` if needed.     */
/* -------------------------------------------------------------------------- */

const hasStateKeys = (o: UnknownRecord): boolean => AGENT_STATE_KEYS.some((k) => k in o);

/** Pulls an `AgentState` patch out of an event payload (handles LangGraph "updates" shape). */
function extractPatch(payload: unknown): { patch: AgentState; node?: string } | null {
  if (!isRecord(payload)) return null;

  let node = asString(payload.node) ?? asString(payload.langgraph_node);
  let source: UnknownRecord = payload;

  for (const key of ["state", "data", "update", "values", "output"]) {
    const candidate = payload[key];
    if (isRecord(candidate)) {
      source = candidate;
      break;
    }
  }

  if (!hasStateKeys(source)) {
    // LangGraph `updates` mode: { node_name: { ...state patch } }
    const entries = Object.entries(source).filter(
      (entry): entry is [string, UnknownRecord] => isRecord(entry[1]) && hasStateKeys(entry[1]),
    );
    if (!entries.length) return null;
    node = entries[entries.length - 1][0];
    source = Object.assign({}, ...entries.map(([, value]) => value)) as UnknownRecord;
  }

  const patch: UnknownRecord = {};
  for (const key of AGENT_STATE_KEYS) {
    if (source[key] !== undefined) patch[key] = source[key];
  }
  return { patch: patch as AgentState, node };
}

function extractMessage(payload: unknown): string {
  if (typeof payload === "string" && payload.trim()) return payload.trim();
  if (isRecord(payload)) {
    const found = pick(payload, ["message", "detail", "error", "text"]);
    if (found) return found;
  }
  return "Une erreur est survenue pendant le traitement.";
}

function toStreamEvents(eventName: string, rawData: string): StreamEvent[] {
  const trimmed = rawData.trim();
  if (trimmed === "[DONE]") return [{ type: "done" }];

  let payload: unknown = trimmed;
  let isJson = false;
  try {
    payload = JSON.parse(trimmed);
    isJson = true;
  } catch {
    /* plain-text payload */
  }

  // Event kind: SSE `event:` name, or a `type` / `event` field inside the JSON data.
  let kind = eventName.toLowerCase();
  if (kind === "message" && isRecord(payload)) {
    kind = (asString(payload.type) ?? asString(payload.event) ?? "message").toLowerCase();
  }

  switch (kind) {
    case "error":
    case "failure":
      return [{ type: "error", message: extractMessage(payload) }];

    case "done":
    case "end":
    case "complete":
    case "completed":
    case "final": {
      const found = extractPatch(payload);
      const events: StreamEvent[] = found ? [{ type: "state", ...found }] : [];
      events.push({ type: "done" });
      return events;
    }

    case "token":
    case "delta":
    case "chunk":
    case "answer_delta": {
      const text =
        typeof payload === "string"
          ? isJson
            ? payload
            : rawData // keep the whitespace of plain-text tokens
          : isRecord(payload)
            ? ((payload.text ?? payload.content ?? payload.delta ?? payload.token) as unknown)
            : undefined;
      return typeof text === "string" && text ? [{ type: "token", text }] : [];
    }

    case "status":
    case "progress": {
      const found = extractPatch(payload);
      if (found) return [{ type: "state", ...found }];
      const status = extractMessage(payload);
      return [{ type: "state", patch: { workflow_status: status } }];
    }

    default: {
      const found = extractPatch(payload);
      return found ? [{ type: "state", ...found }] : [];
    }
  }
}

/** Parses one SSE frame (the text between two blank lines). */
function parseFrame(frame: string): { event: string; data: string } | null {
  let event = "message";
  const data: string[] = [];

  for (const line of frame.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue; // blank line or heartbeat comment
    const idx = line.indexOf(":");
    const field = idx === -1 ? line : line.slice(0, idx);
    let value = idx === -1 ? "" : line.slice(idx + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }

  if (!data.length && event === "message") return null;
  return { event, data: data.join("\n") };
}

/* -------------------------------------------------------------------------- */
/* Message stream                                                             */
/* -------------------------------------------------------------------------- */

/** An agent run can last a while (retrieval + LLM calls). */
const STREAM_TIMEOUT_MS = 300_000;

/**
 * POSTs a message and reports the agent's progression while the response streams.
 *
 * axios (XHR adapter) exposes the partial body through `onDownloadProgress`:
 * `responseText` keeps growing, so we remember how much we already parsed and
 * only dispatch complete SSE frames. Resolves when the stream is closed.
 */
export async function streamMessage(
  sessionId: string,
  message: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const body: MessageRequest = { message: message.trim() };
  let offset = 0;

  const dispatch = (frame: string) => {
    const parsed = parseFrame(frame);
    if (!parsed) return;
    for (const event of toStreamEvents(parsed.event, parsed.data)) onEvent(event);
  };

  const consume = (fullText: string, final = false) => {
    const pending = fullText.slice(offset);
    const separator = /\r?\n\r?\n/g;
    let consumed = 0;
    let match: RegExpExecArray | null;

    while ((match = separator.exec(pending)) !== null) {
      dispatch(pending.slice(consumed, match.index));
      consumed = match.index + match[0].length;
    }
    offset += consumed;

    if (final && offset < fullText.length) {
      const rest = fullText.slice(offset);
      offset = fullText.length;
      if (rest.trim()) dispatch(rest);
    }
  };

  try {
    const response = await request.post<string>(
      `sessions/${encodeURIComponent(sessionId)}/messages`,
      body,
      {
        signal,
        timeout: STREAM_TIMEOUT_MS,
        responseType: "text",
        transformResponse: [(data: unknown) => data], // keep the raw text, no JSON.parse
        headers: { Accept: "text/event-stream" },
        onDownloadProgress: (progress: AxiosProgressEvent) => {
          const xhr = progress.event?.target as XMLHttpRequest | undefined;
          if (typeof xhr?.responseText === "string") consume(xhr.responseText);
        },
      },
    );
    // Flush whatever the last progress event did not cover.
    consume(typeof response.data === "string" ? response.data : "", true);
  } catch (err) {
    throw toApiError(err);
  }
}

/* -------------------------------------------------------------------------- */
/* Case / source normalisation                                                */
/* -------------------------------------------------------------------------- */

const excerpt = (text: string, max: number): string =>
  text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;

function metadataOf(raw: UnknownRecord): UnknownRecord | undefined {
  const meta = raw.metadatas ?? raw.metadata;
  return isRecord(meta) ? meta : undefined;
}

function toCaseItem(raw: unknown, index: number): CaseItem | null {
  if (!isRecord(raw)) return null;

  const metadata = metadataOf(raw);
  const content = pick(raw, ["documents", "document", "content", "text", "page_content"]);
  const title =
    pick(raw, ["title", "titre", "affaire", "name"]) ??
    (metadata ? pick(metadata, ["title", "titre", "affaire", "reference", "numero", "number"]) : undefined);
  const summary = pick(raw, ["summary", "resume", "résumé", "description", "nom de l'affaire"]);

  if (!title && !summary && !content) return null;

  return {
    id: pick(raw, ["chunk_id", "id", "case_id"]) ?? `case-${index}`,
    title: title ?? `Affaire ${index + 1}`,
    summary: summary ?? (content ? excerpt(content, 280) : ""),
    relevance_score: asNumber(raw.rerank_score ?? raw.relevance_score ?? raw.score),
    content,
    metadata,
  };
}

function toSourceItem(raw: unknown, index: number): SourceItem | null {
  if (!isRecord(raw)) return null;
  const text = pick(raw, ["documents", "document", "content", "text", "page_content", "excerpt"]);
  if (!text) return null;
  return {
    id: pick(raw, ["chunk_id", "id"]) ?? `source-${index}`,
    excerpt: text,
    metadata: metadataOf(raw),
  };
}

export function normalizeCases(list: unknown): CaseItem[] {
  if (!Array.isArray(list)) return [];
  return list.map(toCaseItem).filter((c): c is CaseItem => c !== null);
}

export function normalizeSources(list: unknown): SourceItem[] {
  if (!Array.isArray(list)) return [];
  return list.map(toSourceItem).filter((s): s is SourceItem => s !== null);
}