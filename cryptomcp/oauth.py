"""Однопользовательский OAuth 2.1 provider для удалённого MCP.

MCP SDK берёт на себя протокольные эндпоинты (metadata, DCR, authorize,
token и revoke), а этот модуль отвечает за вход владельца и долговечное
хранилище клиентов, кодов и токенов. Состояние хранится отдельно от рыночной
базы: MCP-сервер остаётся только читателем market.sqlite.
"""

from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import secrets
import sqlite3
import threading
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import Any

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

OAUTH_SCOPE = "mcp"
AUTH_REQUEST_TTL_S = 10 * 60
AUTH_CODE_TTL_S = 5 * 60
ACCESS_TOKEN_TTL_S = 60 * 60
REFRESH_TOKEN_TTL_S = 30 * 24 * 60 * 60
MAX_CLIENTS = 100
MAX_LOGIN_ATTEMPTS = 5


def _secret_key(value: str) -> str:
    """Не класть одноразовые коды и bearer-токены в SQLite открытым текстом."""
    return hashlib.sha256(value.encode()).hexdigest()


class SQLiteOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    """Authorization server для одного владельца с постоянным SQLite store."""

    def __init__(
        self,
        *,
        db_path: str,
        issuer_url: str,
        resource_url: str,
        login_secret: str,
        static_token: str | None = None,
    ) -> None:
        if not login_secret:
            raise ValueError("OAuth login secret must not be empty")
        self.db_path = Path(db_path)
        self.issuer_url = issuer_url.rstrip("/")
        self.resource_url = resource_url.rstrip("/")
        self.login_secret = login_secret
        self.static_token = static_token or None
        self._lock = threading.RLock()
        self._prepare_database()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout = 10000")
        return con

    def _prepare_database(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA journal_mode = WAL")
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS oauth_records (
                    kind TEXT NOT NULL,
                    record_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    expires_at REAL,
                    family_id TEXT,
                    PRIMARY KEY (kind, record_key)
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS oauth_expiry "
                "ON oauth_records (kind, expires_at)"
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS oauth_family "
                "ON oauth_records (family_id)"
            )
            # Смена ключа владельца означает отзыв всех выданных полномочий.
            # Регистрации клиентов можно оставить: они не дают доступа сами по
            # себе и смогут пройти новый вход без повторного DCR.
            fingerprint = _secret_key(f"cryptomcp-owner:{self.login_secret}")
            previous = con.execute(
                """
                SELECT payload FROM oauth_records
                WHERE kind = 'meta' AND record_key = 'login_secret_fingerprint'
                """
            ).fetchone()
            if previous is not None and previous["payload"] != fingerprint:
                con.execute(
                    """
                    DELETE FROM oauth_records
                    WHERE kind IN ('request', 'code', 'access', 'refresh')
                    """
                )
            con.execute(
                """
                INSERT OR REPLACE INTO oauth_records
                    (kind, record_key, payload, expires_at, family_id)
                VALUES ('meta', 'login_secret_fingerprint', ?, NULL, NULL)
                """,
                (fingerprint,),
            )
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:  # pragma: no cover - Windows не поддерживает POSIX mode
            pass

    @staticmethod
    def _dump(value: Any, *, exclude: set[str] | None = None) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", exclude=exclude or set())
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    def _cleanup(self, con: sqlite3.Connection) -> None:
        con.execute(
            "DELETE FROM oauth_records WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),),
        )

    def _put(
        self,
        kind: str,
        record_key: str,
        payload: str,
        *,
        expires_at: float | None = None,
        family_id: str | None = None,
        con: sqlite3.Connection | None = None,
    ) -> None:
        def write(connection: sqlite3.Connection) -> None:
            connection.execute(
                """
                INSERT OR REPLACE INTO oauth_records
                    (kind, record_key, payload, expires_at, family_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (kind, record_key, payload, expires_at, family_id),
            )

        if con is not None:
            write(con)
            return
        with self._lock, self._connect() as connection:
            self._cleanup(connection)
            write(connection)

    def _get(self, kind: str, record_key: str) -> sqlite3.Row | None:
        with self._lock, self._connect() as con:
            self._cleanup(con)
            return con.execute(
                """
                SELECT payload, expires_at, family_id
                FROM oauth_records WHERE kind = ? AND record_key = ?
                """,
                (kind, record_key),
            ).fetchone()

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        row = self._get("client", client_id)
        if row is None:
            return None
        return OAuthClientInformationFull.model_validate_json(row["payload"])

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        with self._lock, self._connect() as con:
            self._cleanup(con)
            count = con.execute(
                "SELECT COUNT(*) FROM oauth_records WHERE kind = 'client'"
            ).fetchone()[0]
            if count >= MAX_CLIENTS:
                raise RegistrationError(
                    error="invalid_client_metadata",
                    error_description="Превышен лимит зарегистрированных клиентов.",
                )
            self._put(
                "client",
                client_info.client_id,
                self._dump(client_info),
                con=con,
            )

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        if params.resource is not None and params.resource.rstrip("/") != self.resource_url:
            raise AuthorizeError(
                error="invalid_target",
                error_description="Запрошен неизвестный MCP resource.",
            )

        request_id = secrets.token_urlsafe(32)
        expires_at = time.time() + AUTH_REQUEST_TTL_S
        payload = {
            "client_id": client.client_id,
            "client_name": client.client_name or "MCP-клиент",
            "redirect_uri": str(params.redirect_uri),
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "code_challenge": params.code_challenge,
            "scopes": params.scopes or [OAUTH_SCOPE],
            "state": params.state,
            "resource": self.resource_url,
            "attempts": 0,
        }
        self._put(
            "request",
            _secret_key(request_id),
            self._dump(payload),
            expires_at=expires_at,
        )
        return f"{self.issuer_url}/oauth/authorize?request={request_id}"

    def _load_request(self, request_id: str) -> tuple[dict[str, Any], float] | None:
        row = self._get("request", _secret_key(request_id))
        if row is None:
            return None
        return json.loads(row["payload"]), float(row["expires_at"])

    def _authorization_code_value(self, request_id: str) -> str:
        """Стабильный код для безопасной повторной отправки одной формы.

        Некоторые встроенные браузеры повторяют POST после callback-навигации.
        Код выводится из случайного request id и секрета сервера, поэтому его
        можно вернуть ещё раз, не сохраняя bearer-секрет открытым текстом.
        """
        digest = hmac.new(
            self.login_secret.encode(),
            f"cryptomcp-code:{request_id}".encode(),
            hashlib.sha256,
        ).digest()
        return urlsafe_b64encode(digest).decode().rstrip("=")

    def _authorization_redirect(self, request_id: str, data: dict[str, Any]) -> Response:
        target = construct_redirect_uri(
            data["redirect_uri"],
            code=self._authorization_code_value(request_id),
            state=data.get("state"),
        )
        return RedirectResponse(target, status_code=302, headers={"Cache-Control": "no-store"})

    def _render_login(
        self,
        request_id: str,
        data: dict[str, Any],
        *,
        error: str | None = None,
        status_code: int = 200,
    ) -> HTMLResponse:
        client_name = html.escape(str(data.get("client_name") or "MCP-клиент"))
        error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
        content = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Подключение cryptomcp</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
            background: #101419; color: #eef2f5; }}
    main {{ width: min(420px, calc(100% - 40px)); padding: 32px;
            border: 1px solid #303943; border-radius: 16px; background: #171d24; }}
    h1 {{ margin: 0 0 12px; font-size: 1.45rem; }}
    p {{ color: #b9c2ca; line-height: 1.5; }}
    label {{ display: block; margin: 24px 0 8px; font-weight: 650; }}
    input {{ box-sizing: border-box; width: 100%; padding: 12px; border-radius: 9px;
             border: 1px solid #46515c; background: #0f1419; color: inherit; }}
    button {{ width: 100%; margin-top: 16px; padding: 12px; border: 0;
              border-radius: 9px; background: #2f78ff; color: white;
              font-weight: 700; cursor: pointer; }}
    .error {{ color: #ff9b9b; }}
    small {{ display: block; margin-top: 18px; color: #89949e; line-height: 1.4; }}
  </style>
</head>
<body>
  <main>
    <h1>Разрешить доступ к cryptomcp</h1>
    <p><strong>{client_name}</strong> запрашивает доступ к рыночному контексту.</p>
    {error_html}
    <form method="post" action="/oauth/authorize">
      <input type="hidden" name="request" value="{html.escape(request_id, quote=True)}">
      <label for="access_key">Ключ доступа</label>
      <input id="access_key" name="access_key" type="password" autocomplete="current-password"
             required autofocus>
      <button type="submit">Подключить</button>
    </form>
    <small>Ключ проверяется только этим сервером и не передаётся MCP-клиенту.</small>
  </main>
</body>
</html>"""
        return HTMLResponse(
            content,
            status_code=status_code,
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'unsafe-inline'; "
                    "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
                ),
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
            },
        )

    async def authorization_page(self, request: Request) -> Response:
        """GET показывает форму, POST проверяет ключ и выдаёт authorization code."""
        if request.method == "GET":
            request_id = request.query_params.get("request", "")
            loaded = self._load_request(request_id) if request_id else None
            if loaded is None:
                return HTMLResponse("Запрос авторизации истёк или недействителен.", 400)
            data, _ = loaded
            if data.get("approved"):
                return self._authorization_redirect(request_id, data)
            return self._render_login(request_id, data)

        form = await request.form()
        request_id = form.get("request")
        access_key = form.get("access_key")
        if not isinstance(request_id, str) or not isinstance(access_key, str):
            return HTMLResponse("Неверные параметры запроса.", 400)

        loaded = self._load_request(request_id)
        if loaded is None:
            return HTMLResponse("Запрос авторизации истёк или недействителен.", 400)
        data, expires_at = loaded

        if not hmac.compare_digest(access_key.encode(), self.login_secret.encode()):
            attempts = int(data.get("attempts", 0)) + 1
            if attempts >= MAX_LOGIN_ATTEMPTS:
                with self._lock, self._connect() as con:
                    con.execute(
                        "DELETE FROM oauth_records WHERE kind = 'request' AND record_key = ?",
                        (_secret_key(request_id),),
                    )
                return self._render_login(
                    request_id,
                    data,
                    error="Слишком много попыток. Начните подключение заново.",
                    status_code=429,
                )
            data["attempts"] = attempts
            self._put(
                "request",
                _secret_key(request_id),
                self._dump(data),
                expires_at=expires_at,
            )
            return self._render_login(
                request_id, data, error="Неверный ключ доступа.", status_code=401
            )

        # Повторный POST той же формы должен повторить redirect, а не ломать
        # уже одобренный flow. Сам authorization code всё равно одноразовый на
        # /token и удаляется SDK/provider при обмене.
        if data.get("approved"):
            return self._authorization_redirect(request_id, data)

        code_value = self._authorization_code_value(request_id)
        code_expires_at = time.time() + AUTH_CODE_TTL_S
        code = AuthorizationCode(
            code=code_value,
            client_id=data["client_id"],
            redirect_uri=data["redirect_uri"],
            redirect_uri_provided_explicitly=data["redirect_uri_provided_explicitly"],
            expires_at=code_expires_at,
            scopes=data["scopes"],
            code_challenge=data["code_challenge"],
            resource=data["resource"],
            subject="owner",
        )

        with self._lock, self._connect() as con:
            self._cleanup(con)
            request_key = _secret_key(request_id)
            current = con.execute(
                """
                SELECT payload FROM oauth_records
                WHERE kind = 'request' AND record_key = ?
                """,
                (request_key,),
            ).fetchone()
            if current is None:
                return HTMLResponse("Запрос авторизации уже использован.", 400)
            data["approved"] = True
            self._put(
                "request",
                request_key,
                self._dump(data),
                expires_at=code_expires_at,
                con=con,
            )
            self._put(
                "code",
                _secret_key(code_value),
                self._dump(code, exclude={"code"}),
                expires_at=code.expires_at,
                family_id=request_key,
                con=con,
            )

        return self._authorization_redirect(request_id, data)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        row = self._get("code", _secret_key(authorization_code))
        if row is None:
            return None
        code = AuthorizationCode(
            code=authorization_code,
            **json.loads(row["payload"]),
        )
        return code if code.client_id == client.client_id else None

    def _mint_tokens(
        self,
        *,
        client_id: str,
        scopes: list[str],
        resource: str,
        subject: str | None,
        con: sqlite3.Connection,
    ) -> OAuthToken:
        now = int(time.time())
        family_id = secrets.token_urlsafe(24)
        access_value = secrets.token_urlsafe(32)
        refresh_value = secrets.token_urlsafe(40)
        access = AccessToken(
            token=access_value,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + ACCESS_TOKEN_TTL_S,
            resource=resource,
            subject=subject,
            claims={"iss": self.issuer_url},
        )
        refresh = RefreshToken(
            token=refresh_value,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + REFRESH_TOKEN_TTL_S,
            subject=subject,
        )
        self._put(
            "access",
            _secret_key(access_value),
            self._dump(access, exclude={"token"}),
            expires_at=access.expires_at,
            family_id=family_id,
            con=con,
        )
        self._put(
            "refresh",
            _secret_key(refresh_value),
            self._dump(refresh, exclude={"token"}),
            expires_at=refresh.expires_at,
            family_id=family_id,
            con=con,
        )
        return OAuthToken(
            access_token=access_value,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL_S,
            scope=" ".join(scopes),
            refresh_token=refresh_value,
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        with self._lock, self._connect() as con:
            self._cleanup(con)
            code_key = _secret_key(authorization_code.code)
            row = con.execute(
                """
                SELECT family_id FROM oauth_records
                WHERE kind = 'code' AND record_key = ?
                """,
                (code_key,),
            ).fetchone()
            if row is None or authorization_code.client_id != client.client_id:
                raise TokenError(error="invalid_grant", error_description="Код уже использован.")
            con.execute(
                "DELETE FROM oauth_records WHERE kind = 'code' AND record_key = ?",
                (code_key,),
            )
            if row["family_id"]:
                con.execute(
                    "DELETE FROM oauth_records WHERE kind = 'request' AND record_key = ?",
                    (row["family_id"],),
                )
            return self._mint_tokens(
                client_id=client.client_id,
                scopes=authorization_code.scopes,
                resource=authorization_code.resource or self.resource_url,
                subject=authorization_code.subject,
                con=con,
            )

    async def load_access_token(self, token: str) -> AccessToken | None:
        # Совместимость с существующими Bearer-подключениями после выката OAuth.
        if self.static_token and hmac.compare_digest(token.encode(), self.static_token.encode()):
            return AccessToken(
                token=token,
                client_id="static-token",
                scopes=[OAUTH_SCOPE],
                resource=self.resource_url,
                subject="owner",
                claims={"iss": self.issuer_url},
            )

        row = self._get("access", _secret_key(token))
        if row is None:
            return None
        access = AccessToken(token=token, **json.loads(row["payload"]))
        if access.resource and access.resource.rstrip("/") != self.resource_url:
            return None
        return access

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        row = self._get("refresh", _secret_key(refresh_token))
        if row is None:
            return None
        token = RefreshToken(token=refresh_token, **json.loads(row["payload"]))
        return token if token.client_id == client.client_id else None

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        key = _secret_key(refresh_token.token)
        with self._lock, self._connect() as con:
            self._cleanup(con)
            row = con.execute(
                """
                SELECT family_id FROM oauth_records
                WHERE kind = 'refresh' AND record_key = ?
                """,
                (key,),
            ).fetchone()
            if row is None or refresh_token.client_id != client.client_id:
                raise TokenError(
                    error="invalid_grant", error_description="Refresh token уже использован."
                )
            con.execute("DELETE FROM oauth_records WHERE family_id = ?", (row["family_id"],))
            return self._mint_tokens(
                client_id=client.client_id,
                scopes=scopes,
                resource=self.resource_url,
                subject=refresh_token.subject,
                con=con,
            )

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        kind = "access" if isinstance(token, AccessToken) else "refresh"
        with self._lock, self._connect() as con:
            row = con.execute(
                """
                SELECT family_id FROM oauth_records
                WHERE kind = ? AND record_key = ?
                """,
                (kind, _secret_key(token.token)),
            ).fetchone()
            if row is not None:
                con.execute("DELETE FROM oauth_records WHERE family_id = ?", (row["family_id"],))
