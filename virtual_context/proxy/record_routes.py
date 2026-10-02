"""Dashboard routes for reviewing and editing a conversation's record.

Admins list facts beside their source turns, reject or restore a fact, edit or
remove a turn, forget a topic, and read the audit trail. Changes to the record
commit before the response; the rebuild of what derives from them runs in the
conversation's background compaction pool under its compaction lease.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .state import ProxyState

BASE = "/dashboard/conversations/{conversation_id}/record"


def _not_found() -> JSONResponse:
    return JSONResponse(
        {"error": "not_found", "message": "Conversation not loaded."}, status_code=404,
    )


async def _body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def register_record_routes(
    app: "FastAPI", resolve_state: Callable[[str], "ProxyState | None"],
) -> None:
    """Register the record routes; *resolve_state* finds a conversation's state."""

    def _who(body: dict) -> tuple[str, str]:
        return (str(body.get("actor") or "").strip() or "dashboard", str(body.get("reason") or ""))

    def _mutate(conversation_id, run):
        state = resolve_state(conversation_id)
        if state is None:
            return _not_found()
        editor = state.engine.record_editor
        try:
            report = run(editor)
        except KeyError as exc:
            return JSONResponse({"error": "not_found", "message": str(exc)}, status_code=404)
        except ValueError as exc:
            return JSONResponse({"error": "invalid_edit", "message": str(exc)}, status_code=400)
        except RuntimeError as exc:
            return JSONResponse({"error": "edit_in_progress", "message": str(exc)}, status_code=409)
        if report.get("found") is False:
            return JSONResponse({"error": "topic_not_found", **report}, status_code=404)
        operation_id = report.get("operation_id")
        if operation_id:
            op = editor.status(operation_id)
            if op and op[0]["status"] == "pending":
                state.submit_record_edit(operation_id)
        return JSONResponse(report, status_code=202 if operation_id else 200)

    def _read(conversation_id, read):
        state = resolve_state(conversation_id)
        if state is None:
            return _not_found()
        return JSONResponse({"conversation_id": conversation_id, **read(state.engine)})

    @app.get(BASE + "/facts")
    async def record_facts(conversation_id: str, request: Request):
        topic = request.query_params.get("topic") or None
        return await asyncio.to_thread(
            _read, conversation_id, lambda engine: {"facts": engine.record_editor.facts(topic)},
        )

    @app.post(BASE + "/verify")
    async def record_verify(conversation_id: str):
        return await asyncio.to_thread(
            _read, conversation_id,
            lambda engine: {"trust_states": engine.record_editor.verify_facts()},
        )

    @app.post(BASE + "/facts/{fact_id}/{verdict}")
    async def record_fact_verdict(conversation_id: str, fact_id: str, verdict: str, request: Request):
        if verdict not in ("reject", "restore"):
            return JSONResponse({"error": "invalid_verdict", "message": "use reject or restore"}, status_code=400)
        actor, reason = _who(await _body(request))
        return await asyncio.to_thread(
            _mutate, conversation_id,
            lambda editor: (editor.reject_fact if verdict == "reject" else editor.restore_fact)(
                fact_id, actor=actor, reason=reason,
            ),
        )

    @app.post(BASE + "/forget")
    async def record_forget(conversation_id: str, request: Request):
        body = await _body(request)
        topic = str(body.get("topic") or "").strip()
        if not topic:
            return JSONResponse({"error": "invalid_edit", "message": "topic is required"}, status_code=400)
        actor, reason = _who(body)
        return await asyncio.to_thread(
            _mutate, conversation_id,
            lambda editor: editor.forget_topic(topic, actor=actor, reason=reason),
        )

    @app.post(BASE + "/turns/{turn_id}/edit")
    async def record_edit_turn(conversation_id: str, turn_id: str, request: Request):
        body = await _body(request)
        user, assistant = body.get("user_content"), body.get("assistant_content")
        if not isinstance(user, (str, type(None))) or not isinstance(assistant, (str, type(None))):
            return JSONResponse({"error": "invalid_edit", "message": "content must be text"}, status_code=400)
        actor, reason = _who(body)
        return await asyncio.to_thread(
            _mutate, conversation_id,
            lambda editor: editor.edit_turn(
                turn_id, user_content=user, assistant_content=assistant, actor=actor, reason=reason,
            ),
        )

    @app.post(BASE + "/turns/{turn_id}/remove")
    async def record_remove_turn(conversation_id: str, turn_id: str, request: Request):
        actor, reason = _who(await _body(request))
        return await asyncio.to_thread(
            _mutate, conversation_id,
            lambda editor: editor.remove_turn(turn_id, actor=actor, reason=reason),
        )

    @app.get(BASE + "/edits")
    async def record_edits(conversation_id: str, request: Request):
        turn_id = request.query_params.get("turn_id") or None
        return await asyncio.to_thread(
            _read, conversation_id,
            lambda engine: {"edits": engine._store.get_turn_edits(
                engine.config.conversation_id, canonical_turn_id=turn_id,
            )},
        )

    @app.get(BASE + "/verdicts")
    async def record_verdicts(conversation_id: str):
        return await asyncio.to_thread(
            _read, conversation_id,
            lambda engine: {"verdicts": engine._store.get_fact_verdicts(engine.config.conversation_id)},
        )

    @app.get(BASE + "/operations")
    async def record_operations(conversation_id: str, request: Request):
        operation_id = request.query_params.get("operation_id") or None
        return await asyncio.to_thread(
            _read, conversation_id,
            lambda engine: {"operations": engine.record_editor.status(operation_id)},
        )
