"""HTTP-обвязка для удалённого режима (PLAN §7.3).

Аутентификация сделана здесь, а не в Caddy, сознательно. Caddy на сервере один
и обслуживает все проекты сразу; чтобы он проверял токен, ему пришлось бы
передать переменную окружения, то есть отредактировать compose чужого проекта.
Меньше риска — держать проверку внутри своего контейнера.

Сервер публичен и ключей не хранит, поэтому его компрометация не стоит ничего,
кроме IP. Но без токена это бесплатный прокси к Binance от вашего адреса, и
чужой трафик способен упереть вас в лимит или получить 418 — бан IP до трёх
суток, во время которого сервис не работает.
"""

from __future__ import annotations

import hmac
import logging
import os

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

logger = logging.getLogger(__name__)

#: Пути, доступные без токена. Health нужен деплою и мониторингу.
PUBLIC_PATHS = frozenset({"/health"})


async def health(_: Request) -> Response:
    return JSONResponse({"status": "ok", "service": "cryptomcp"})


def _unauthorized(detail: str) -> Response:
    return JSONResponse(
        {"error": {"kind": "unauthorized", "message": detail}},
        status_code=401,
        headers={"WWW-Authenticate": 'Bearer realm="cryptomcp"'},
    )


def bearer_auth_middleware(token: str):
    """Проверка статического Bearer-токена."""

    async def middleware(request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, provided = header.partition(" ")
        if scheme.lower() != "bearer" or not provided:
            return _unauthorized("Требуется заголовок Authorization: Bearer <токен>.")
        # Сравнение с постоянным временем: обычное == позволяет подобрать
        # токен по времени ответа.
        if not hmac.compare_digest(provided, token):
            logger.warning("отклонён запрос с неверным токеном на %s", request.url.path)
            return _unauthorized("Неверный токен.")
        return await call_next(request)

    return middleware


def build_app(server, *, token: str | None = None) -> Starlette:
    """Starlette-приложение MCP с health-эндпоинтом и проверкой токена."""
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        # Хост нужен транспорту для проверки заголовка Host: за прокси он
        # приходит доменным именем, а не адресом контейнера.
        host=os.environ.get("CRYPTOMCP_PUBLIC_HOST", "0.0.0.0"),
    )
    app.router.routes.append(Route("/health", health, methods=["GET"]))

    if token:
        # add_middleware, а не декоратор .middleware(): на уже собранном
        # Starlette-приложении декоратора нет.
        app.add_middleware(BaseHTTPMiddleware, dispatch=bearer_auth_middleware(token))
    else:
        logger.warning(
            "MCP_AUTH_TOKEN не задан — сервер отвечает без аутентификации. "
            "Допустимо только для локального запуска."
        )
    return app
