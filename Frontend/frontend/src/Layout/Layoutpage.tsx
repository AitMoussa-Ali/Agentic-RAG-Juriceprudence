import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import {
  ApiError,
  createSession,
  deleteSession,
  normalizeCases,
  normalizeSources,
  streamMessage,
} from "../services/api";
import type {
  AgentRun,
  AgentState,
  AgentStep,
  CaseItem,
  Message,
  StepStatus,
} from "../types/chat";

/* -------------------------------------------------------------------------- */
/* Helpers                                                                    */
/* -------------------------------------------------------------------------- */

const makeId = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

const formatTime = (iso: string): string =>
  new Date(iso).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });

const formatScore = (score: number | null): string | null =>
  score === null ? null : score.toLocaleString("fr-FR", { maximumFractionDigits: 2 });

const plural = (n: number, word: string): string => `${n} ${word}${n > 1 ? "s" : ""}`;

const truncate = (text: string, max: number): string =>
  text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;

/** First "summary-like" string of the agent's `research` object, if any. */
function researchNote(research: Record<string, unknown> | undefined, max: number): string | undefined {
  if (!research) return undefined;
  for (const key of ["summary", "synthesis", "analysis", "reasoning", "conclusion"]) {
    const value = research[key];
    if (typeof value === "string" && value.trim()) return truncate(value.trim(), max);
  }
  return undefined;
}

/** Up to three primitive metadata entries, formatted as "key: value". */
function metadataChips(metadata: Record<string, unknown> | undefined): string[] {
  if (!metadata) return [];
  return Object.entries(metadata)
    .filter(([, v]) => (typeof v === "string" && v.trim()) || typeof v === "number")
    .slice(0, 3)
    .map(([k, v]) => `${k.replace(/_/g, " ")} : ${truncate(String(v), 40)}`);
}

const WELCOME_MESSAGE: Message = {
  id: "welcome",
  sender: "assistant",
  text: "Décrivez le sujet qui vous intéresse : je recherche et sélectionne les affaires pertinentes. Ensuite, posez-moi vos questions sur les affaires retenues.",
  timestamp: new Date().toISOString(),
};

/* -------------------------------------------------------------------------- */
/* Agent progression                                                          */
/* -------------------------------------------------------------------------- */

type Phase = "running" | "done" | "error";

/** Derives the progression steps from the (partially filled) agent state. */
function buildSteps(state: AgentState, streamedText: string, phase: Phase): AgentStep[] {
  interface Def {
    id: string;
    label: string;
    done: boolean;
    detail?: string;
  }

  const defs: Def[] = [
    {
      id: "route",
      label: "Analyse de votre demande",
      done: !!state.route,
      detail:
        state.route === "qa"
          ? "Question sur les affaires retenues"
          : state.route === "research"
            ? "Recherche d'affaires"
            : state.route === "chat"
              ? "Conversation générale"
            : undefined,
    },
  ];

  if (state.route === "qa") {
    defs.push(
      {
        id: "sources",
        label: "Consultation des sources",
        done: state.qa_sources !== undefined,
        detail: state.qa_sources ? plural(state.qa_sources.length, "passage") : undefined,
      },
      {
        id: "answer",
        label: "Rédaction de la réponse",
        done: !!state.qa_answer,
        detail: !state.qa_answer && streamedText ? "Rédaction en cours…" : undefined,
      },
    );
  } else if (state.route === "chat") {
    defs.push({
      id: "chat",
      label: "Réponse",
      done: !!state.chat_response,
      detail: !state.chat_response && streamedText ? "Rédaction en cours…" : undefined,
    });
  } else if (state.route === "research") {
    const queryDetail = [
      state.query ? `« ${truncate(state.query, 80)} »` : undefined,
      state.selected_method,
      state.n_results ? `${state.n_results} résultats demandés` : undefined,
    ]
      .filter(Boolean)
      .join(" · ");
    const found = state.retrieved_cases?.length;
    const kept = state.selected_cases?.length;

    defs.push(
      {
        id: "query",
        label: "Formulation de la requête",
        done: !!state.query,
        detail: queryDetail || undefined,
      },
      {
        id: "retrieval",
        label: "Recherche dans la jurisprudence",
        done: state.retrieved_cases !== undefined,
        detail: found === undefined ? undefined : `${plural(found, "résultat")} trouvé${found > 1 ? "s" : ""}`,
      },
      {
        id: "research",
        label: "Analyse des résultats",
        done: state.research !== undefined,
        detail: researchNote(state.research, 160),
      },
      {
        id: "rerank",
        label: "Sélection et reclassement",
        done: state.reranking !== undefined || state.selected_cases !== undefined,
        detail: kept === undefined ? undefined : `${plural(kept, "affaire")} retenue${kept > 1 ? "s" : ""}`,
      },
    );
  }

  const firstPending = defs.findIndex((d) => !d.done);

  return defs.map((d, i): AgentStep => {
    let status: StepStatus;
    if (phase === "done" || d.done) status = "done";
    else if (i === firstPending) status = phase === "error" ? "error" : "active";
    else status = "pending";
    return { id: d.id, label: d.label, detail: d.detail, status };
  });
}

/** Turns the final agent state into the assistant message shown in the chat. */
function buildReply(state: AgentState, streamed: string): Message {
  const base = {
    id: makeId(),
    sender: "assistant" as const,
    timestamp: new Date().toISOString(),
    trace: buildSteps(state, streamed, "done"),
  };

  const answer = (state.qa_answer ?? streamed).trim();

  if (state.route === "chat") {
    return {
      ...base,
      text: (state.chat_response ?? "").trim() || "Bonjour ! Comment puis-je vous aider ?",
    };
  }

  // Question about the retained cases.
  if (state.route === "qa" || (answer && !state.retrieved_cases && !state.selected_cases)) {
    const sources = normalizeSources(state.qa_sources);
    return {
      ...base,
      text: answer || "Je n'ai pas trouvé d'élément de réponse dans les affaires retenues.",
      sources: sources.length ? sources : undefined,
    };
  }

  // Case research.
  if (state.route === "research" || state.selected_cases || state.retrieved_cases) {
    const picked = state.selected_cases?.length
      ? state.selected_cases
      : (state.retrieved_cases ?? []).slice(0, 8);
    const cases = normalizeCases(picked);
    const note = researchNote(state.research, 700);

    let text: string;
    if (!cases.length) {
      text = note ?? "Aucune affaire pertinente n'a été trouvée.";
    } else {
      text =
        note ??
        `J'ai retenu ${plural(cases.length, "affaire")}. Vous pouvez maintenant me poser des questions à leur sujet.`;
    }
    return { ...base, text, cases: cases.length ? cases : undefined };
  }

  return {
    ...base,
    text: answer || state.workflow_status || "Le traitement s'est terminé sans résultat.",
  };
}

/* -------------------------------------------------------------------------- */
/* Icons                                                                      */
/* -------------------------------------------------------------------------- */

const iconProps = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

const SendIcon = () => (
  <svg {...iconProps}>
    <path d="M12 19V5" />
    <path d="m5 12 7-7 7 7" />
  </svg>
);
const CloseIcon = () => (
  <svg {...iconProps} width={14} height={14}>
    <path d="M18 6 6 18" />
    <path d="m6 6 12 12" />
  </svg>
);
const ScaleIcon = () => (
  <svg {...iconProps} width={18} height={18}>
    <path d="M12 3v18" />
    <path d="M6 21h12" />
    <path d="M5 7h14" />
    <path d="m5 7-3 8a3.5 3.5 0 0 0 6 0Z" />
    <path d="m19 7-3 8a3.5 3.5 0 0 0 6 0Z" />
  </svg>
);
const AlertIcon = () => (
  <svg {...iconProps} width={18} height={18}>
    <circle cx="12" cy="12" r="10" />
    <path d="M12 8v4" />
    <path d="M12 16h.01" />
  </svg>
);
const PlusIcon = () => (
  <svg {...iconProps} width={14} height={14}>
    <path d="M12 5v14" />
    <path d="M5 12h14" />
  </svg>
);
const ChevronIcon = () => (
  <svg {...iconProps} width={14} height={14}>
    <path d="m6 9 6 6 6-6" />
  </svg>
);

/* -------------------------------------------------------------------------- */
/* Header                                                                     */
/* -------------------------------------------------------------------------- */

interface HeaderProps {
  retainedCases: CaseItem[];
  canReset: boolean;
  onNewConversation: () => void;
}

function Header({ retainedCases, canReset, onNewConversation }: HeaderProps) {
  const count = retainedCases.length;

  return (
    <header className="shrink-0 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="mx-auto flex w-full max-w-4xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-indigo-600 text-white">
            <ScaleIcon />
          </span>
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold leading-tight text-slate-900">
              Agent de recherche juridique
            </h1>
            <p className="truncate text-sm text-slate-500">
              Jurisprudence française, suivi du raisonnement en direct
            </p>
          </div>
        </div>

        <div className="flex max-w-full items-center gap-2">
          {count > 0 && (
            <div
              className="flex max-w-full items-center gap-1.5 rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 text-sm text-indigo-800"
              role="status"
              aria-live="polite"
              title={retainedCases.map((c) => c.title).join("\n")}
            >
              <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-indigo-500" />
              <span className="truncate font-medium">
                {plural(count, "affaire")} retenue{count > 1 ? "s" : ""}
              </span>
            </div>
          )}
          <button
            type="button"
            onClick={onNewConversation}
            disabled={!canReset}
            className="flex shrink-0 items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 text-sm text-slate-600 transition-colors hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <PlusIcon />
            <span className="hidden sm:inline">Nouvelle conversation</span>
            <span className="sm:hidden">Nouveau</span>
          </button>
        </div>
      </div>
    </header>
  );
}

/* -------------------------------------------------------------------------- */
/* Agent progress (live + recorded)                                           */
/* -------------------------------------------------------------------------- */

function StepIcon({ status }: { status: StepStatus }) {
  if (status === "done") {
    return (
      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-600 text-white">
        <svg {...iconProps} width={12} height={12} strokeWidth={3}>
          <path d="M20 6 9 17l-5-5" />
        </svg>
      </span>
    );
  }
  if (status === "active") {
    return (
      <span className="flex h-5 w-5 items-center justify-center rounded-full border-2 border-indigo-200">
        <span className="h-3 w-3 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent motion-reduce:animate-pulse" />
      </span>
    );
  }
  if (status === "error") {
    return (
      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-white">
        <CloseIcon />
      </span>
    );
  }
  return <span className="h-5 w-5 rounded-full border-2 border-slate-200 bg-white" />;
}

function StepList({ steps }: { steps: AgentStep[] }) {
  return (
    <ol className="space-y-3">
      {steps.map((step, i) => (
        <li key={step.id} className="relative flex gap-3">
          {i < steps.length - 1 && (
            <span
              aria-hidden
              className={`absolute left-[9px] top-6 h-[calc(100%-4px)] w-px ${
                step.status === "done" ? "bg-indigo-200" : "bg-slate-200"
              }`}
            />
          )}
          <span className="relative z-10 mt-0.5 shrink-0">
            <StepIcon status={step.status} />
          </span>
          <div className="min-w-0">
            <p
              className={`text-sm leading-snug ${
                step.status === "pending"
                  ? "text-slate-400"
                  : step.status === "active"
                    ? "font-medium text-slate-900"
                    : step.status === "error"
                      ? "font-medium text-red-700"
                      : "text-slate-700"
              }`}
            >
              {step.label}
            </p>
            {step.detail && (
              <p className="mt-0.5 break-words text-xs leading-relaxed text-slate-500">{step.detail}</p>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

/** Bubble shown while the agent works: steps, live status line and streamed text. */
function AgentProgress({ run }: { run: AgentRun }) {
  const steps = buildSteps(run.state, run.streamedText, "running");

  return (
    <div className="flex justify-start" role="status" aria-label="Le raisonnement de l'agent est en cours">
      <div className="w-full max-w-[90%] rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3.5 shadow-sm sm:max-w-md">
        <StepList steps={steps} />

        {run.state.workflow_status && (
          <p className="mt-3 flex items-start gap-2 border-t border-slate-100 pt-2.5 text-xs text-slate-500">
            <span className="mt-1 h-1.5 w-1.5 shrink-0 animate-pulse rounded-full bg-indigo-400" />
            <span className="min-w-0 break-words">{run.state.workflow_status}</span>
          </p>
        )}

        {run.streamedText && (
          <p className="mt-3 whitespace-pre-wrap break-words border-t border-slate-100 pt-2.5 text-[15px] leading-relaxed text-slate-800">
            {run.streamedText}
            <span className="ml-0.5 inline-block h-4 w-px animate-pulse bg-slate-400 align-middle" />
          </p>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* CaseSelector                                                               */
/* -------------------------------------------------------------------------- */

function CaseSelector({ cases }: { cases: CaseItem[] }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <ul className="mt-3 grid gap-3 sm:grid-cols-2" aria-label="Affaires retenues">
      {cases.map((item, index) => {
        const isOpen = item.id === expandedId;
        const score = formatScore(item.relevance_score);
        const chips = metadataChips(item.metadata);

        return (
          <li key={item.id} className="flex">
            <div
              className={[
                "flex w-full flex-col rounded-xl border bg-white transition duration-150",
                isOpen
                  ? "border-indigo-500 shadow-md ring-1 ring-indigo-500"
                  : "border-slate-200 hover:-translate-y-0.5 hover:border-indigo-300 hover:shadow-md motion-reduce:hover:translate-y-0",
              ].join(" ")}
            >
              <button
                type="button"
                onClick={() => setExpandedId(isOpen ? null : item.id)}
                aria-expanded={isOpen}
                className="flex w-full flex-col rounded-xl p-4 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
              >
                <span className="flex items-start gap-2">
                  <span className="mt-0.5 flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md bg-indigo-50 px-1 text-xs font-semibold text-indigo-700">
                    {index + 1}
                  </span>
                  <span className="min-w-0 flex-1 text-sm font-semibold leading-snug text-slate-900">
                    {item.title}
                  </span>
                  <span
                    className={`mt-0.5 shrink-0 text-slate-400 transition-transform ${isOpen ? "rotate-180" : ""}`}
                  >
                    <ChevronIcon />
                  </span>
                </span>

                {item.summary && (
                  <span
                    className={`mt-1.5 text-sm leading-relaxed text-slate-600 ${isOpen ? "" : "line-clamp-3"}`}
                  >
                    {item.summary}
                  </span>
                )}

                {(chips.length > 0 || score) && (
                  <span className="mt-3 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-slate-100 pt-2 text-xs text-slate-500">
                    {chips.map((chip) => (
                      <span key={chip} className="max-w-full truncate">
                        {chip}
                      </span>
                    ))}
                    {score && <span className="ml-auto font-medium text-slate-700">Score {score}</span>}
                  </span>
                )}
              </button>

              {isOpen && item.content && (
                <div className="max-h-72 overflow-y-auto whitespace-pre-wrap break-words border-t border-slate-100 px-4 py-3 text-sm leading-relaxed text-slate-600">
                  {item.content}
                </div>
              )}
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/* -------------------------------------------------------------------------- */
/* MessageList                                                                */
/* -------------------------------------------------------------------------- */

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.sender === "user";
  const hasCases = !!message.cases?.length;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={["min-w-0", hasCases ? "w-full" : "max-w-[85%] sm:max-w-xl"].join(" ")}>
        {!!message.trace?.length && (
          <details className="mb-2 w-fit max-w-full rounded-lg border border-slate-200 bg-white text-sm text-slate-600">
            <summary className="cursor-pointer select-none rounded-lg px-3 py-1.5 font-medium text-slate-700 transition-colors hover:text-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
              Étapes de l'agent ({message.trace.length})
            </summary>
            <div className="border-t border-slate-100 p-3">
              <StepList steps={message.trace} />
            </div>
          </details>
        )}

        <div
          className={[
            "w-fit max-w-full whitespace-pre-wrap break-words px-4 py-2.5 text-[15px] leading-relaxed",
            isUser
              ? "ml-auto rounded-2xl rounded-tr-sm bg-indigo-600 text-white"
              : message.isError
                ? "rounded-2xl rounded-tl-sm border border-red-200 bg-red-50 text-red-800"
                : "rounded-2xl rounded-tl-sm border border-slate-200 bg-white text-slate-800 shadow-sm",
          ].join(" ")}
        >
          {message.text}
        </div>

        {!!message.sources?.length && (
          <details className="mt-2 w-fit max-w-full rounded-lg border border-slate-200 bg-white text-sm text-slate-600">
            <summary className="cursor-pointer select-none rounded-lg px-3 py-1.5 font-medium text-slate-700 transition-colors hover:text-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500">
              Sources ({message.sources.length})
            </summary>
            <ul className="max-h-64 space-y-3 overflow-y-auto border-t border-slate-100 p-3">
              {message.sources.map((source) => (
                <li key={source.id}>
                  <p className="break-all text-xs text-slate-400">{source.id}</p>
                  <p className="mt-0.5 line-clamp-5 whitespace-pre-wrap break-words leading-relaxed">
                    {source.excerpt}
                  </p>
                </li>
              ))}
            </ul>
          </details>
        )}

        {hasCases && <CaseSelector cases={message.cases!} />}

        <time
          dateTime={message.timestamp}
          className={`mt-1 block px-1 text-xs text-slate-400 ${isUser ? "text-right" : ""}`}
        >
          {formatTime(message.timestamp)}
        </time>
      </div>
    </div>
  );
}

interface MessageListProps {
  messages: Message[];
  run: AgentRun | null;
}

function MessageList({ messages, run }: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null);

  // Smooth-scroll to the newest content whenever messages or the live run change.
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, run]);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto" role="log" aria-live="polite">
      <div className="mx-auto flex w-full max-w-4xl flex-col gap-5 px-4 py-6 sm:px-6">
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        {run && <AgentProgress run={run} />}
        <div ref={endRef} />
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* ErrorBanner                                                                */
/* -------------------------------------------------------------------------- */

interface ErrorBannerProps {
  message: string;
  onDismiss: () => void;
  onRetry?: () => void;
}

function ErrorBanner({ message, onDismiss, onRetry }: ErrorBannerProps) {
  return (
    <div className="shrink-0 border-t border-red-200 bg-red-50" role="alert">
      <div className="mx-auto flex w-full max-w-4xl items-start gap-3 px-4 py-2.5 text-sm text-red-800 sm:px-6">
        <span className="mt-0.5 shrink-0 text-red-500">
          <AlertIcon />
        </span>
        <p className="min-w-0 flex-1 break-words">{message}</p>
        <div className="flex shrink-0 items-center gap-1">
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="rounded-md px-2 py-1 font-medium text-red-700 transition-colors hover:bg-red-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
            >
              Réessayer
            </button>
          )}
          <button
            type="button"
            onClick={onDismiss}
            aria-label="Fermer le message d'erreur"
            className="flex h-7 w-7 items-center justify-center rounded-md text-red-600 transition-colors hover:bg-red-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
          >
            <CloseIcon />
          </button>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* ChatInput                                                                  */
/* -------------------------------------------------------------------------- */

interface ChatInputProps {
  placeholder: string;
  hint: string;
  onSubmit: (text: string) => void;
  disabled: boolean;
}

function ChatInput({ placeholder, hint, onSubmit, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSubmit = value.trim().length > 0 && !disabled;

  // Auto-grow the textarea up to a maximum height.
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  const submit = useCallback(() => {
    if (!canSubmit) return;
    onSubmit(value.trim());
    setValue("");
  }, [canSubmit, onSubmit, value]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="shrink-0 border-t border-slate-200 bg-white pb-[env(safe-area-inset-bottom)]">
      <div className="mx-auto w-full max-w-4xl px-4 py-3 sm:px-6">
        <div className="flex items-end gap-2 rounded-2xl border border-slate-300 bg-white p-2 shadow-sm transition focus-within:border-indigo-500 focus-within:ring-2 focus-within:ring-indigo-100">
          <label htmlFor="chat-input" className="sr-only">
            Votre message
          </label>
          <textarea
            id="chat-input"
            ref={textareaRef}
            rows={1}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className="max-h-40 min-h-[40px] w-full flex-1 resize-none bg-transparent px-2 py-2 text-[15px] text-slate-900 placeholder:text-slate-400 focus:outline-none"
          />
          <button
            type="button"
            onClick={submit}
            disabled={!canSubmit}
            aria-label="Envoyer"
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition-colors hover:bg-indigo-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
          >
            {disabled ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-400/40 border-t-slate-500 motion-reduce:animate-pulse" />
            ) : (
              <SendIcon />
            )}
          </button>
        </div>
        <p className="mt-1.5 px-2 text-xs text-slate-400">{hint}</p>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* LayoutPage (parent)                                                        */
/* -------------------------------------------------------------------------- */

export default function LayoutPage() {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [loading, setLoading] = useState(false);
  /** Live data of the run being streamed (null when idle). */
  const [run, setRun] = useState<AgentRun | null>(null);
  /** Cases the agent kept during the last research; the session remembers them server-side. */
  const [retainedCases, setRetainedCases] = useState<CaseItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  const sessionIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const lastMessageRef = useRef<string | null>(null);

  // Cancel any in-flight request when the page unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  const runMessage = useCallback(async (text: string) => {
    lastMessageRef.current = text;
    setError(null);
    setLoading(true);
    setRun({ state: {}, streamedText: "" });

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    /** One full attempt: make sure a session exists, stream, then publish the reply. */
    const attempt = async () => {
      const sessionId =
        sessionIdRef.current ?? (sessionIdRef.current = await createSession(controller.signal));

      let state: AgentState = {};
      let streamed = "";
      let streamError = null as string | null;
      setRun({ state, streamedText: streamed });

      await streamMessage(
        sessionId,
        text,
        (event) => {
          if (event.type === "state") state = { ...state, ...event.patch };
          else if (event.type === "token") streamed += event.text;
          else if (event.type === "error") streamError = event.message;
          else return;
          setRun({ state, streamedText: streamed });
        },
        controller.signal,
      );

      if (streamError) throw new ApiError(streamError, "server");

      setMessages((prev) => [...prev, buildReply(state, streamed)]);
      if (state.selected_cases?.length) setRetainedCases(normalizeCases(state.selected_cases));
    };

    try {
      try {
        await attempt();
      } catch (err) {
        // The backend forgot the session (restart, cleanup): open a new one and retry once.
        if (err instanceof ApiError && err.kind === "not_found" && !controller.signal.aborted) {
          sessionIdRef.current = null;
          setRetainedCases([]);
          setMessages((prev) => [
            ...prev,
            {
              id: makeId(),
              sender: "assistant",
              text: "La session avait expiré : une nouvelle conversation a été ouverte. Les affaires retenues précédemment ont été oubliées.",
              timestamp: new Date().toISOString(),
            },
          ]);
          await attempt();
        } else {
          throw err;
        }
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(
        err instanceof ApiError ? err.message : "Une erreur inattendue est survenue. Veuillez réessayer.",
      );
    } finally {
      if (abortRef.current === controller) {
        setLoading(false);
        setRun(null);
        abortRef.current = null;
      }
    }
  }, []);

  const handleSend = useCallback(
    (text: string) => {
      setMessages((prev) => [
        ...prev,
        { id: makeId(), sender: "user", text, timestamp: new Date().toISOString() },
      ]);
      void runMessage(text);
    },
    [runMessage],
  );

  const handleRetry = useCallback(() => {
    const last = lastMessageRef.current;
    if (last && !loading) void runMessage(last);
  }, [loading, runMessage]);

  const handleNewConversation = useCallback(() => {
    const id = sessionIdRef.current;
    sessionIdRef.current = null;
    lastMessageRef.current = null;
    setMessages([{ ...WELCOME_MESSAGE, timestamp: new Date().toISOString() }]);
    setRetainedCases([]);
    setError(null);
    if (id) void deleteSession(id).catch(() => undefined); // best effort
  }, []);

  const hasRetained = retainedCases.length > 0;

  return (
    <div className="flex h-dvh flex-col bg-slate-50 text-slate-900 antialiased">
      <Header
        retainedCases={retainedCases}
        canReset={!loading}
        onNewConversation={handleNewConversation}
      />

      <MessageList messages={messages} run={run} />

      {error && (
        <ErrorBanner
          message={error}
          onDismiss={() => setError(null)}
          onRetry={lastMessageRef.current ? handleRetry : undefined}
        />
      )}

      <ChatInput
        placeholder={
          hasRetained
            ? "Posez une question sur les affaires retenues, ou lancez une nouvelle recherche…"
            : "Ex. licenciement abusif, clause de non-concurrence…"
        }
        hint="Entrée pour envoyer, Maj + Entrée pour un retour à la ligne."
        onSubmit={handleSend}
        disabled={loading}
      />
    </div>
  );
}

// Named exports for the sub-components, in case you want to split them into files later.
export { Header, MessageList, CaseSelector, ChatInput, AgentProgress };