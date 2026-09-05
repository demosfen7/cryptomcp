"""HTTP-обвязка для удалённого режима (PLAN §7.3).

OAuth и совместимость со старым статическим ключом сделаны в приложении, а не
в Caddy, сознательно. Caddy на сервере один и обслуживает все проекты сразу;
чтобы он проверял секрет, пришлось бы передать ему переменную окружения, то
есть отредактировать compose чужого проекта. Меньше риска — держать проверку
внутри своего контейнера.

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
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

#: Пути, доступные без токена. Health нужен деплою и мониторингу.
PUBLIC_PATHS = frozenset({"/health"})

#: Запасной заголовок для токена.
#:
#: Старые custom connector на claude.ai могли передать статический токен только
#: этим заголовком. OAuth теперь основной путь, но заголовок оставлен на время
#: миграции — без префикса `Bearer`, одним значением.
API_KEY_HEADER = "x-api-key"


class APIKeyCompatibilityMiddleware:
    """Преобразовать верный старый x-api-key в Bearer до OAuth middleware.

    Это оставляет уже настроенный коннектор рабочим, пока владелец переводит
    его на OAuth. Неверный ключ не получает особой ветки и заканчивается
    стандартным OAuth 401 с discovery-ссылкой.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") == "/mcp":
            headers = list(scope.get("headers", []))
            has_authorization = any(name.lower() == b"authorization" for name, _ in headers)
            api_key = next(
                (value for name, value in headers if name.lower() == API_KEY_HEADER.encode()),
                None,
            )
            if (
                not has_authorization
                and api_key is not None
                and hmac.compare_digest(api_key, self.token.encode())
            ):
                scope = dict(scope)
                scope["headers"] = [*headers, (b"authorization", b"Bearer " + api_key)]
        await self.app(scope, receive, send)


def extract_token(request: Request) -> str | None:
    """Токен из Authorization: Bearer или из запасного заголовка."""
    authorization = request.headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value:
        return value
    return request.headers.get(API_KEY_HEADER) or None


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

        provided = extract_token(request)
        if not provided:
            return _unauthorized(
                "Требуется токен: заголовок Authorization: Bearer <токен> "
                f"либо {API_KEY_HEADER}: <токен>."
            )
        # Сравнение с постоянным временем: обычное == позволяет подобрать
        # токен по времени ответа.
        if not hmac.compare_digest(provided, token):
            logger.warning("отклонён запрос с неверным токеном на %s", request.url.path)
            return _unauthorized("Неверный токен.")
        return await call_next(request)

    return middleware


def build_app(server, *, token: str | None = None) -> Starlette:
    """Starlette-приложение MCP с health и OAuth/совместимой проверкой токена."""
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        # Хост нужен транспорту для проверки заголовка Host: за прокси он
        # приходит доменным именем, а не адресом контейнера.
        host=os.environ.get("CRYPTOMCP_PUBLIC_HOST", "0.0.0.0"),
    )
    app.router.routes.append(Route("/health", health, methods=["GET"]))

    if token and server.settings.auth is not None:
        app.add_middleware(APIKeyCompatibilityMiddleware, token=token)
    elif token:
        # add_middleware, а не декоратор .middleware(): на уже собранном
        # Starlette-приложении декоратора нет.
        app.add_middleware(BaseHTTPMiddleware, dispatch=bearer_auth_middleware(token))
    else:
        logger.warning(
            "MCP_AUTH_TOKEN не задан — сервер отвечает без аутентификации. "
            "Допустимо только для локального запуска."
        )
    return app
