from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from Controllers.Chat_controller import (
    CreateSessionResponse,
    MessageRequest,
    SessionBusyError,
    SessionNotFoundError,
    SessionResponse,
    chat_controller,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _session_or_http_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Session introuvable.",
    )


@router.post(
    "",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_session() -> CreateSessionResponse:
    return chat_controller.create_session()


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    try:
        return chat_controller.get_session(session_id)
    except SessionNotFoundError as exc:
        raise _session_or_http_error() from exc


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str) -> None:
    try:
        chat_controller.delete_session(session_id)
    except SessionNotFoundError as exc:
        raise _session_or_http_error() from exc
    except SessionBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Impossible de supprimer une session pendant le traitement d'un message.",
        ) from exc


@router.post("/{session_id}/messages")
async def send_message(
    session_id: str,
    request: MessageRequest,
) -> StreamingResponse:
    try:
        state = chat_controller.begin_message(session_id, request)
    except SessionNotFoundError as exc:
        raise _session_or_http_error() from exc
    except SessionBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un message est déjà en cours pour cette session.",
        ) from exc

    stream: AsyncIterator[str] = chat_controller.stream_message(session_id, state)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
