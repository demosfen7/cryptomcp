"""Тесты HTTP-обвязки: извлечение токена и аутентификация (PLAN §7.3)."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from cryptomcp.app import (
    API_KEY_HEADER,
    bearer_auth_middleware,
    extract_token,
    health,
)

TOKEN = "s3cr3t-token-value"


def make_request(headers: dict[str, str]) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/mcp", "headers": raw})


class TestExtractToken:
    def test_bearer(self):
        assert extract_token(make_request({"Authorization": f"Bearer {TOKEN}"})) == TOKEN

    def test_bearer_scheme_is_case_insensitive(self):
        assert extract_token(make_request({"Authorization": f"bearer {TOKEN}"})) == TOKEN

    def test_api_key_header(self):
        """Запасной путь для custom connector на claude.ai.

        В форме коннектора имя Authorization зарезервировано под OAuth и вручную
        не задаётся, поэтому токен принимается и через x-api-key — без префикса.
        """
        assert extract_token(make_request({API_KEY_HEADER: TOKEN})) == TOKEN

    def test_api_key_header_name(self):
        assert API_KEY_HEADER == "x-api-key"

    def test_authorization_wins_over_api_key(self):
        request = make_request({
            "Authorization": f"Bearer {TOKEN}",
            API_KEY_HEADER: "другой",
        })
        assert extract_token(request) == TOKEN

    def test_api_key_used_when_authorization_is_not_bearer(self):
        request = make_request({"Authorization": "Basic xxx", API_KEY_HEADER: TOKEN})
        assert extract_token(request) == TOKEN

    def test_no_headers(self):
        assert extract_token(make_request({})) is None

    def test_empty_values(self):
        assert extract_token(make_request({"Authorization": "Bearer"})) is None
        assert extract_token(make_request({API_KEY_HEADER: ""})) is None


def build_test_app(token: str | None) -> Starlette:
    async def protected(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    app = Starlette(routes=[
        Route("/mcp", protected, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
    ])
    if token:
        app.add_middleware(BaseHTTPMiddleware, dispatch=bearer_auth_middleware(token))
    return app


class TestAuthMiddleware:
    def setup_method(self):
        self.client = TestClient(build_test_app(TOKEN))

    def test_health_is_public(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_rejects_without_token(self):
        response = self.client.get("/mcp")
        assert response.status_code == 401
        assert response.json()["error"]["kind"] == "unauthorized"
        # Сообщение должно называть оба способа, иначе непонятно, что делать.
        assert API_KEY_HEADER in response.json()["error"]["message"]

    def test_rejects_wrong_token(self):
        response = self.client.get("/mcp", headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401

    def test_accepts_bearer(self):
        response = self.client.get("/mcp", headers={"Authorization": f"Bearer {TOKEN}"})
        assert response.status_code == 200

    def test_accepts_api_key(self):
        response = self.client.get("/mcp", headers={API_KEY_HEADER: TOKEN})
        assert response.status_code == 200

    def test_rejects_wrong_api_key(self):
        response = self.client.get("/mcp", headers={API_KEY_HEADER: "wrong"})
        assert response.status_code == 401

    def test_challenge_header_present(self):
        response = self.client.get("/mcp")
        assert "WWW-Authenticate" in response.headers

    def test_open_when_no_token_configured(self):
        """Без MCP_AUTH_TOKEN сервер отвечает всем — режим локального запуска."""
        client = TestClient(build_test_app(None))
        assert client.get("/mcp").status_code == 200
