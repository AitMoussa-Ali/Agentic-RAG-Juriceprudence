/* -------------------------------------------------------------------------- */
/* Agent state (mirror of the LangGraph `ResearchState`)                      */
/* -------------------------------------------------------------------------- */

export type RetrievalMethod = "TF-IDF" | "BM25" | "Semantic" | "Hybrid";
export type AgentRoute = "research" | "qa";

/** Every key is optional: the state is filled progressively while the agent runs. */
export interface AgentState {
  user_message?: string;
  route?: AgentRoute;
  workflow_status?: string;
  query?: string;
  selected_method?: RetrievalMethod;
  n_results?: number;
  top_k?: number;
  retrieved_cases?: Record<string, unknown>[];
  research?: Record<string, unknown>;
  selected_cases?: Record<string, unknown>[];
  reranking?: Record<string, unknown>;
  user_question?: string;
  qa_answer?: string;
  qa_sources?: Record<string, unknown>[];
}

/** Runtime list of the state keys, used to recognise state patches in the stream. */
export const AGENT_STATE_KEYS: ReadonlyArray<keyof AgentState> = [
  "user_message",
  "route",
  "workflow_status",
  "query",
  "selected_method",
  "n_results",
  "top_k",
  "retrieved_cases",
  "research",
  "selected_cases",
  "reranking",
  "user_question",
  "qa_answer",
  "qa_sources",
];

/* -------------------------------------------------------------------------- */
/* API payloads                                                               */
/* -------------------------------------------------------------------------- */

/** Body of POST /api/sessions/{session_id}/messages. */
export interface MessageRequest {
  message: string;
}

/* -------------------------------------------------------------------------- */
/* Stream events (Server-Sent Events, normalised)                             */
/* -------------------------------------------------------------------------- */

export type StreamEvent =
  | { type: "state"; patch: AgentState; node?: string }
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

/* -------------------------------------------------------------------------- */
/* UI models                                                                  */
/* -------------------------------------------------------------------------- */

/** A legal case returned by the agent (normalised from a raw case dict). */
export interface CaseItem {
  id: string;
  title: string;
  summary: string;
  relevance_score: number | null;
  content?: string;
  metadata?: Record<string, unknown>;
}

/** A passage used by the agent to ground a Q&A answer. */
export interface SourceItem {
  id: string;
  excerpt: string;
  metadata?: Record<string, unknown>;
}

export type StepStatus = "pending" | "active" | "done" | "error";

/** One line of the agent's progression. */
export interface AgentStep {
  id: string;
  label: string;
  detail?: string;
  status: StepStatus;
}

/** Live data of the run currently streaming. */
export interface AgentRun {
  state: AgentState;
  streamedText: string;
}

export type MessageSender = "user" | "assistant";

export interface Message {
  id: string;
  sender: MessageSender;
  text: string;
  /** Cases found / retained by a research run. */
  cases?: CaseItem[];
  /** Passages backing a Q&A answer. */
  sources?: SourceItem[];
  /** Snapshot of the agent's steps for this answer (collapsed in the UI). */
  trace?: AgentStep[];
  isError?: boolean;
  /** ISO 8601 timestamp. */
  timestamp: string;
}