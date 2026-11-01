import asyncio
import json
import logging
import threading
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from Agents.State import ResearchState, RetrievalMethod

logger = logging.getLogger(__name__)

ProgressStage = Literal[
    "accepted",
    "monitoring",
    "research",
    "retrieval",
    "qa",
    "completed",
    "error",
]


class CreateSessionResponse(BaseModel):
    session_id: str
    status: str


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    selected_method: RetrievalMethod | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("Le message ne peut pas être vide.")
        return message


class SessionResponse(BaseModel):
    session_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    selected_cases: list[dict[str, Any]] = Field(default_factory=list)
    research: dict[str, Any] = Field(default_factory=dict)
    qa_answer: str = ""
    qa_sources: list[dict[str, Any]] = Field(default_factory=list)


class SessionNotFoundError(Exception):
    pass


class SessionBusyError(Exception):
    pass


class ChatController:
    """Own in-memory chat state and stream LangGraph node updates."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._busy_sessions: set[str] = set()
        self._lock = threading.RLock()

    def create_session(self) -> CreateSessionResponse:
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        with self._lock:
            self._sessions[session_id] = {
                "state": {},
                "created_at": now,
                "updated_at": now,
            }
        return CreateSessionResponse(session_id=session_id, status="created")

    def get_session(self, session_id: str) -> SessionResponse:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)

            state = session["state"]
            status = "processing" if session_id in self._busy_sessions else "idle"
            return SessionResponse(
                session_id=session_id,
                status=status,
                created_at=session["created_at"],
                updated_at=session["updated_at"],
                selected_cases=state.get("selected_cases", []),
                research=state.get("research", {}),
                qa_answer=state.get("qa_answer", ""),
                qa_sources=state.get("qa_sources", []),
            )

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._sessions:
                raise SessionNotFoundError(session_id)
            if session_id in self._busy_sessions:
                raise SessionBusyError(session_id)
            del self._sessions[session_id]

    def _begin_turn(
        self,
        session_id: str,
        request: MessageRequest,
    ) -> ResearchState:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(session_id)
            if session_id in self._busy_sessions:
                raise SessionBusyError(session_id)

            self._busy_sessions.add(session_id)
            state = dict(session["state"])
            state["user_message"] = request.message
            state["workflow_status"] = "accepted"
            if request.selected_method is not None:
                state["selected_method"] = request.selected_method
            return state

    def begin_message(
        self,
        session_id: str,
        request: MessageRequest,
    ) -> ResearchState:
        return self._begin_turn(session_id, request)

    async def stream_message(
        self,
        session_id: str,
        state: ResearchState,
    ) -> AsyncIterator[str]:
        completed = False

        try:
            yield self._sse(
                "progress",
                {
                    "stage": "accepted",
                    "message": "Votre message a été reçu.",
                },
            )

            from Agents.Monitoring_agent import get_monitoring_graph

            graph = get_monitoring_graph()
            async for update in graph.astream(state, stream_mode="updates"):
                for node_name, node_update in update.items():
                    state.update(node_update)
                    event_name, payload = self._progress_for_node(
                        node_name,
                        node_update,
                    )
                    yield self._sse(event_name, payload)

            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    raise SessionNotFoundError(session_id)
                session["state"] = state
                session["updated_at"] = datetime.now(timezone.utc)
            completed = True
            yield self._sse(
                "completed",
                {
                    "stage": "completed",
                    "message": "Le traitement est terminé.",
                    "result": self._public_result(state),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to process chat message for session %s", session_id)
            yield self._sse(
                "error",
                {
                    "stage": "error",
                    "message": "Le traitement a échoué. Réessayez dans un instant.",
                },
            )
        finally:
            with self._lock:
                self._busy_sessions.discard(session_id)
                session = self._sessions.get(session_id)
                if session is not None and not completed:
                    session["updated_at"] = datetime.now(timezone.utc)

    @staticmethod
    def _progress_for_node(
        node_name: str,
        node_update: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        if node_name == "monitoring":
            route = node_update.get("route")
            return "progress", {
                "stage": "monitoring",
                "route": route,
                "message": (
                    "Une recherche d'affaires va démarrer."
                    if route == "research"
                    else "Une réponse sur les affaires sélectionnées va être préparée."
                ),
            }

        if node_name == "research":
            research = node_update.get("research", {})
            cases = node_update.get("retrieved_cases", [])
            return "progress", {
                "stage": "research",
                "status": node_update.get("workflow_status"),
                "cases_found": len(cases),
                "method_used": research.get("method_used"),
                "fallback_used": research.get("fallback_used", False),
                "attempted_methods": research.get("attempted_methods", []),
                "message": (
                    f"{len(cases)} affaire(s) trouvée(s)."
                    if cases
                    else "Aucune affaire n'a été trouvée."
                ),
            }

        if node_name == "retrieval":
            selected_cases = node_update.get("selected_cases", [])
            return "progress", {
                "stage": "retrieval",
                "status": node_update.get("workflow_status"),
                "selected_count": len(selected_cases),
                "cases": selected_cases,
                "message": f"{len(selected_cases)} affaire(s) retenue(s) après classement.",
            }

        if node_name == "qa":
            return "progress", {
                "stage": "qa",
                "status": node_update.get("workflow_status"),
                "message": "La réponse est prête.",
                "answer": node_update.get("qa_answer", ""),
                "sources": node_update.get("qa_sources", []),
            }

        logger.warning("Received progress update from unknown graph node: %s", node_name)
        return "progress", {
            "stage": "monitoring",
            "message": "Une étape du traitement est terminée.",
        }

    @staticmethod
    def _public_result(state: ResearchState) -> dict[str, Any]:
        return {
            "route": state.get("route"),
            "workflow_status": state.get("workflow_status"),
            "research": state.get("research", {}),
            "selected_cases": state.get("selected_cases", []),
            "qa_answer": state.get("qa_answer", ""),
            "qa_sources": state.get("qa_sources", []),
        }

    @staticmethod
    def _sse(event: str, data: dict[str, Any]) -> str:
        payload = json.dumps(data, ensure_ascii=False, default=str)
        return f"event: {event}\ndata: {payload}\n\n"


chat_controller = ChatController()
