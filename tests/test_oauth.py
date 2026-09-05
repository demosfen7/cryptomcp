"""Полный OAuth authorization-code flow для удалённого MCP."""

from __future__ import annotations

import asyncio
import base64
import hashlib
from urllib.parse import parse_qs, urlsplit

from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.mcpserver import MCPServer
from starlette.testclient import TestClient

from cryptomcp.app import API_KEY_HEADER, build_app
from cryptomcp.oauth import OAUTH_SCOPE, SQLiteOAuthProvider

BASE_URL = "http://localhost:8000"
RESOURCE_URL = f"{BASE_URL}/mcp"
LOGIN_SECRET = "one-owner-secret"
REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"


def make_provider(tmp_path) -> SQLiteOAuthProvider:
    return SQLiteOAuthProvider(
        db_path=str(tmp_path / "oauth.sqlite"),
        issuer_url=BASE_URL,
        resource_url=RESOURCE_URL,
        login_secret=LOGIN_SECRET,
        static_token=LOGIN_SECRET,
    )


def make_app(tmp_path):
    provider = make_provider(tmp_path)
    auth = AuthSettings(
        issuer_url=BASE_URL,
        resource_server_url=RESOURCE_URL,
        required_scopes=[OAUTH_SCOPE],
        client_registration_options=ClientRegistrationOptions(
            enabled=True,
            valid_scopes=[OAUTH_SCOPE],
            default_scopes=[OAUTH_SCOPE],
        ),
        revocation_options=RevocationOptions(enabled=True),
    )
    server = MCPServer("oauth-test", auth=auth, auth_server_provider=provider)
    server.custom_route("/oauth/authorize", methods=["GET", "POST"])(
        provider.authorization_page
    )
    return build_app(server, token=LOGIN_SECRET), provider


def register(client: TestClient) -> dict:
    response = client.post(
        "/register",
        json={
            "client_name": "Claude",
            "redirect_uris": [REDIRECT_URI],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "scope": OAUTH_SCOPE,
        },
    )
    assert response.status_code == 201
    return response.json()


def authorize(client: TestClient, client_id: str) -> tuple[str, str]:
    verifier = "v" * 64
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
    challenge = challenge.decode().rstrip("=")
    response = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "state": "client-state",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "scope": OAUTH_SCOPE,
            "resource": RESOURCE_URL,
        },
        follow_redirects=False,
    )
    assert response.status_code == 302

    login_url = urlsplit(response.headers["location"])
    request_id = parse_qs(login_url.query)["request"][0]
    page = client.get(f"{login_url.path}?{login_url.query}")
    assert page.status_code == 200
    assert "Claude" in page.text

    wrong = client.post(
        "/oauth/authorize",
        data={"request": request_id, "access_key": "wrong"},
    )
    assert wrong.status_code == 401

    approved = client.post(
        "/oauth/authorize",
        data={"request": request_id, "access_key": LOGIN_SECRET},
        follow_redirects=False,
    )
    assert approved.status_code == 302
    callback = urlsplit(approved.headers["location"])
    params = parse_qs(callback.query)
    assert params["state"] == ["client-state"]
    assert params["iss"] == [BASE_URL]
    return params["code"][0], verifier


def test_oauth_discovery_and_full_flow(tmp_path):
    app, provider = make_app(tmp_path)
    with TestClient(app) as client:
        unauthenticated = client.get("/mcp")
        assert unauthenticated.status_code == 401
        assert "resource_metadata=" in unauthenticated.headers["www-authenticate"]

        resource_metadata = client.get("/.well-known/oauth-protected-resource/mcp")
        assert resource_metadata.status_code == 200
        assert resource_metadata.json()["resource"] == RESOURCE_URL
        assert resource_metadata.json()["authorization_servers"] == [BASE_URL]

        server_metadata = client.get("/.well-known/oauth-authorization-server")
        assert server_metadata.status_code == 200
        assert server_metadata.json()["registration_endpoint"] == f"{BASE_URL}/register"
        assert server_metadata.json()["code_challenge_methods_supported"] == ["S256"]

        registration = register(client)
        code, verifier = authorize(client, registration["client_id"])
        token_response = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": registration["client_id"],
                "code_verifier": verifier,
                "resource": RESOURCE_URL,
            },
        )
        assert token_response.status_code == 200
        tokens = token_response.json()
        assert tokens["token_type"] == "Bearer"
        assert tokens["scope"] == OAUTH_SCOPE
        assert tokens["refresh_token"]

        access = client.get(
            "/mcp", headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        assert access.status_code != 401

        refreshed = client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": registration["client_id"],
                "resource": RESOURCE_URL,
            },
        )
        assert refreshed.status_code == 200
        new_tokens = refreshed.json()
        assert new_tokens["access_token"] != tokens["access_token"]
        assert new_tokens["refresh_token"] != tokens["refresh_token"]

        revoked = client.post(
            "/revoke",
            data={
                "token": new_tokens["access_token"],
                "token_type_hint": "access_token",
                "client_id": registration["client_id"],
                "client_secret": "",
            },
        )
        assert revoked.status_code == 200

    assert asyncio.run(provider.load_access_token(new_tokens["access_token"])) is None
    stored_client = asyncio.run(provider.get_client(registration["client_id"]))
    assert stored_client is not None
    assert (
        asyncio.run(
            provider.load_refresh_token(
                stored_client,
                new_tokens["refresh_token"],
            )
        )
        is None
    )


def test_tokens_and_clients_survive_provider_restart(tmp_path):
    app, provider = make_app(tmp_path)
    with TestClient(app) as client:
        registration = register(client)
        code, verifier = authorize(client, registration["client_id"])
        response = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": registration["client_id"],
                "code_verifier": verifier,
                "resource": RESOURCE_URL,
            },
        )
        tokens = response.json()

    restarted = make_provider(tmp_path)
    assert restarted.db_path == provider.db_path
    assert provider is not restarted

    assert asyncio.run(restarted.get_client(registration["client_id"])) is not None
    assert asyncio.run(restarted.load_access_token(tokens["access_token"])) is not None

    rotated = SQLiteOAuthProvider(
        db_path=str(tmp_path / "oauth.sqlite"),
        issuer_url=BASE_URL,
        resource_url=RESOURCE_URL,
        login_secret="rotated-owner-secret",
        static_token="rotated-owner-secret",
    )
    assert asyncio.run(rotated.get_client(registration["client_id"])) is not None
    assert asyncio.run(rotated.load_access_token(tokens["access_token"])) is None


def test_static_bearer_and_api_key_remain_compatible(tmp_path):
    app, _ = make_app(tmp_path)
    with TestClient(app) as client:
        bearer = client.get(
            "/mcp", headers={"Authorization": f"Bearer {LOGIN_SECRET}"}
        )
        api_key = client.get("/mcp", headers={API_KEY_HEADER: LOGIN_SECRET})
        assert bearer.status_code != 401
        assert api_key.status_code != 401
