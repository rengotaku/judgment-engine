"""FastAPI アプリ。エンドポイントは設計案 v1 のとおり。

- `POST /v1/decisions`（Jev 互換。`POST /v1/systemone` も同じ処理を受ける）
- `POST /v1/feedback`
- `GET /v1/models`, `GET /health`, `GET /metrics`

`/health` と `/metrics` は認証を要求しない（中身は設定と件数だけで、入力は含まない）。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

from . import __version__
from .auth import BearerAuth, RateLimiter
from .config import Settings, read_api_key
from .inference.base import Scorer
from .inference.engine import JudgmentEngine
from .inference.factory import build_scorer
from .masking import mask_text
from .schemas import DecisionRequest, DecisionResponse, FeedbackRequest, parse_questions

logger = logging.getLogger("judgment_engine")

MAX_STORED_NOTE = 200


class BodyTooLarge(Exception):
    pass


class BodyLimitMiddleware:
    """本文の上限を ASGI 層で数える（Content-Length を信じない）。"""

    def __init__(self, app: Callable[..., Any], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[[], Any], send: Callable[[], Any]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    await _too_large(send)
                    return
            except ValueError:
                await _too_large(send)
                return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except BodyTooLarge:
            await _too_large(send)


async def _too_large(send: Callable[[], Any]) -> None:
    body = json.dumps({"error": {"type": "payload_too_large", "message": "request body is too large"}}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_app(settings: Settings | None = None, scorer: Scorer | None = None, api_key: str | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    if api_key is None:
        api_key = read_api_key()
    if api_key is None and not settings.allow_insecure:
        raise RuntimeError(
            "JE_API_KEY（または JE_API_KEY_FILE）がありません。"
            "認証なしで起動するには JE_ALLOW_INSECURE=1 を明示してください"
        )

    auth = BearerAuth(api_key, settings.allow_insecure)
    limiter = RateLimiter(settings.rate_limit_per_min)
    registry = CollectorRegistry()
    decisions_counter = Counter("judgment_decisions_total", "判断リクエスト数", ["backend"], registry=registry)
    answers_total = Counter("judgment_answers_total", "返した回答数", ["type", "route"], registry=registry)
    errors_total = Counter("judgment_errors_total", "エラー数", ["code"], registry=registry)
    latency = Histogram(
        "judgment_request_seconds",
        "1 リクエストの処理時間",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=registry,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.scorer = scorer or build_scorer(settings)
        app.state.engine = JudgmentEngine(app.state.scorer, settings)
        try:
            yield
        finally:
            app.state.scorer.close()

    app = FastAPI(title="judgment-engine", version=__version__, lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.limits.max_body_bytes)
    app.state.auth = auth
    app.state.limiter = limiter
    app.state.metrics = {
        "decisions": decisions_counter,
        "answers": answers_total,
        "errors": errors_total,
        "latency": latency,
        "registry": registry,
    }
    app.state.feedback_path = Path(settings.data_dir) / "feedback.jsonl"

    def check_auth(request: Request, authorization: str | None) -> None:
        app.state.auth.verify(authorization)
        allowed, _remaining = app.state.limiter.check(RateLimiter.identity(request, authorization))
        if not allowed:
            raise _http_error(429, "rate_limited", "リクエストが多すぎます")

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors_total.labels(code="422").inc()
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "validation_error",
                    "message": "リクエストの形が不正です",
                    "details": [_short_detail(item) for item in exc.errors()],
                }
            },
        )

    @app.exception_handler(_ValidationFailure)
    async def _semantic_handler(request: Request, exc: _ValidationFailure) -> JSONResponse:
        errors_total.labels(code="422").inc()
        return JSONResponse(
            status_code=422,
            content={"error": {"type": "validation_error", "message": str(exc)}},
        )

    @app.post("/v1/decisions", response_model=DecisionResponse)
    @app.post("/v1/systemone", response_model=DecisionResponse, include_in_schema=False)
    async def decisions(
        payload: DecisionRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> DecisionResponse:
        check_auth(request, authorization)
        try:
            questions = parse_questions(payload.questions, settings.limits)
        except ValueError as exc:
            raise _ValidationFailure(str(exc)) from exc

        request_id = payload.request_id or uuid.uuid4().hex
        started = time.perf_counter()
        try:
            answers, routing, pairs, tokens = app.state.engine.decide(payload.state, questions, payload.policy)
        except ValueError as exc:
            errors_total.labels(code="422").inc()
            return JSONResponse(
                status_code=422,
                content={"error": {"type": "validation_error", "message": _safe_message(str(exc))}},
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        decisions_counter.labels(backend=settings.backend).inc()
        latency.observe(elapsed_ms / 1000.0)
        for qid, answer in answers.items():
            answers_total.labels(type=answer.type, route=routing[qid].route).inc()

        if settings.log_prompts:
            logger.info(
                "decision request_id=%s questions=%d pairs=%d latency_ms=%.1f state=%s",
                request_id,
                len(questions),
                pairs,
                elapsed_ms,
                mask_text(
                    payload.state if isinstance(payload.state, str) else json.dumps(payload.state, ensure_ascii=False),
                    200,
                ),
            )
        else:
            logger.info(
                "decision request_id=%s questions=%d pairs=%d latency_ms=%.1f",
                request_id,
                len(questions),
                pairs,
                elapsed_ms,
            )

        return DecisionResponse(
            model=payload.model or f"judgment-engine-0:{settings.backend}",
            answers=answers,
            routing=routing,
            usage={
                "input_tokens": tokens,
                "output_tokens": len(answers),
                "estimated": not app.state.scorer.exact_token_counts,
                "latency_ms": round(elapsed_ms, 3),
                "pairs": pairs,
            },
            request_id=request_id,
            backend=settings.backend,
        )

    @app.post("/v1/feedback", status_code=202)
    async def feedback(
        payload: FeedbackRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        check_auth(request, authorization)
        path: Path = app.state.feedback_path
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(UTC).isoformat(),
            "request_id": payload.request_id,
            "question_id": payload.question_id,
            "outcome": payload.outcome,
            # 入力そのものは残さない（伏字のみ）
            "label": payload.label,
            "note": mask_text(payload.note or "", limit=MAX_STORED_NOTE),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {"accepted": True}

    @app.get("/v1/models")
    async def models(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check_auth(request, authorization)
        return {
            "models": [
                {
                    "id": f"judgment-engine-0:{settings.backend}",
                    "backend": settings.backend,
                    "types": ["choice", "score", "noul"],
                    "limits": {
                        "max_body_bytes": settings.limits.max_body_bytes,
                        "max_state_chars": settings.limits.max_state_chars,
                        "max_questions": settings.limits.max_questions,
                        "max_options": settings.limits.max_options,
                        "max_score_levels": settings.limits.max_score_levels,
                    },
                    "temperature": settings.temperature,
                }
            ]
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "backend": settings.backend}

    @app.get("/metrics")
    async def metrics() -> Response:
        return PlainTextResponse(
            generate_latest(registry).decode("utf-8"), media_type="text/plain; version=0.0.4"
        )

    return app


class _ValidationFailure(Exception):
    pass


def _http_error(code: int, kind: str, message: str) -> Exception:
    from fastapi import HTTPException

    return HTTPException(status_code=code, detail={"type": kind, "message": message})


def _short_detail(item: dict[str, Any]) -> dict[str, Any]:
    """バリデーションの詳細から入力値そのものを落とす（値が秘密でも返さない）。"""
    location = ".".join(str(part) for part in item.get("loc", ()))
    return {"loc": location, "msg": item.get("msg", ""), "type": item.get("type", "")}


def _safe_message(message: str) -> str:
    return mask_text(message, limit=300)
