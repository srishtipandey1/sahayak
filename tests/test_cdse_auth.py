"""Tests for Copernicus Data Space token caching and secret handling."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest
import requests

from sahayak.config import Settings
from sahayak.ingest.cdse_auth import (
    HTTP_USER_AGENT,
    TOKEN_URL,
    AuthenticationError,
    CdseTokenClient,
    RequestFailure,
    post_with_retries,
)


class FakeResponse:
    """Small HTTP response fake for token and API requests."""

    def __init__(
        self,
        status_code: int,
        *,
        content: bytes = b"",
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store the response values consumed by the clients."""
        self.status_code = status_code
        self.content = content
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        """Return the configured JSON object."""
        return self._payload


class FakeSession:
    """Record POST arguments and return a known sequence of responses."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        """Initialize a response queue."""
        self.responses = deque(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        """Return the next response while recording request arguments."""
        self.calls.append((url, kwargs))
        return self.responses.popleft()


def test_token_is_cached_until_refresh_margin() -> None:
    session = FakeSession(
        [
            FakeResponse(200, payload={"access_token": "first-secret-token", "expires_in": 120}),
            FakeResponse(200, payload={"access_token": "second-secret-token", "expires_in": 600}),
        ]
    )
    clock_value = [10.0]
    client = CdseTokenClient(
        Settings(
            cdse_client_id="private-client-id",
            cdse_client_secret="private-client-secret",
            _env_file=None,
        ),
        session=session,
        clock=lambda: clock_value[0],
    )

    assert client.get_token() == "first-secret-token"
    assert client.get_token() == "first-secret-token"
    assert len(session.calls) == 1
    clock_value[0] = 71.0
    assert client.get_token() == "second-secret-token"
    assert len(session.calls) == 2
    assert session.calls[0][0] == TOKEN_URL
    assert session.calls[0][1]["data"]["grant_type"] == "client_credentials"


@pytest.mark.parametrize(
    ("client_id", "client_secret", "missing"),
    [
        (None, "secret", "CDSE_CLIENT_ID"),
        ("client", None, "CDSE_CLIENT_SECRET"),
    ],
)
def test_missing_token_settings_are_named(
    client_id: str | None,
    client_secret: str | None,
    missing: str,
) -> None:
    client = CdseTokenClient(
        Settings(
            cdse_client_id=client_id,
            cdse_client_secret=client_secret,
            _env_file=None,
        ),
        session=FakeSession([]),
    )

    with pytest.raises(AuthenticationError, match=missing):
        client.get_token()


def test_token_client_repr_and_failure_do_not_expose_secrets() -> None:
    secret = "extremely-private-client-secret"
    client = CdseTokenClient(
        Settings(
            cdse_client_id="private-client-id",
            cdse_client_secret=secret,
            _env_file=None,
        ),
        session=FakeSession([FakeResponse(401)]),
    )

    with pytest.raises(AuthenticationError) as error:
        client.get_token()

    assert secret not in repr(client)
    assert secret not in str(error.value)


def test_http_authentication_error_exposes_only_status_and_error_code() -> None:
    client = CdseTokenClient(
        Settings(
            cdse_client_id="private-client-id",
            cdse_client_secret="private-client-secret",
            _env_file=None,
        ),
        session=FakeSession(
            [
                FakeResponse(
                    401,
                    payload={
                        "error": "invalid_client",
                        "error_description": "private response detail",
                    },
                )
            ]
        ),
    )

    with pytest.raises(AuthenticationError) as error:
        client.get_token()

    assert error.value.status_code == 401
    assert error.value.error_code == "invalid_client"
    assert "private response detail" not in str(error.value)


def test_post_retries_connection_timeout_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakySession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            del url, kwargs
            self.calls += 1
            if self.calls == 1:
                raise requests.ConnectTimeout("temporary timeout")
            return FakeResponse(200)

    sleeps: list[float] = []
    monkeypatch.setattr("sahayak.ingest.cdse_auth.time.sleep", sleeps.append)
    session = FlakySession()

    response, attempt = post_with_retries(
        session,
        "https://example.invalid/search",
        service_name="Catalog test",
        timeout=(15, 120),
    )

    assert response.status_code == 200
    assert attempt == 2
    assert session.calls == 2
    assert sleeps == [1.0]


def test_post_does_not_retry_http_400() -> None:
    session = FakeSession([FakeResponse(400)])

    response, attempt = post_with_retries(
        session,
        "https://example.invalid/search",
        service_name="Catalog test",
    )

    assert response.status_code == 400
    assert attempt == 1
    assert len(session.calls) == 1


def test_post_retries_transient_status_and_sets_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InspectingSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.responses = deque(
                [FakeResponse(503, headers={"Retry-After": "0"}), FakeResponse(200)]
            )

        def post(self, url: str, **kwargs: Any) -> FakeResponse:
            del url
            self.calls.append(kwargs)
            return self.responses.popleft()

    sleeps: list[float] = []
    monkeypatch.setattr("sahayak.ingest.cdse_auth.time.sleep", sleeps.append)
    session = InspectingSession()

    response, attempt = post_with_retries(
        session,
        "https://example.invalid/search",
        service_name="Catalog test",
    )

    assert response.status_code == 200
    assert attempt == 2
    assert sleeps == [1.0]
    assert all(call["headers"]["User-Agent"] == HTTP_USER_AGENT for call in session.calls)


def test_transport_failure_reports_cause_and_scrubs_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_token = "test-private-access-token"

    class FailingSession:
        def post(self, url: str, **kwargs: Any) -> None:
            del url, kwargs
            raise requests.ConnectTimeout(f"Bearer {access_token} timed out")

    monkeypatch.setattr("sahayak.ingest.cdse_auth.time.sleep", lambda _: None)

    with pytest.raises(RequestFailure) as error:
        post_with_retries(
            FailingSession(),
            "https://example.invalid/search",
            service_name="Catalog test",
            secrets=(access_token,),
        )

    message = str(error.value)
    assert "ConnectTimeout" in message
    assert "attempt 3/3" in message
    assert access_token not in message


def test_unauthorized_api_request_refreshes_token_once(tmp_path: Any) -> None:
    settings = Settings(
        cdse_client_id="private-client-id",
        cdse_client_secret="private-client-secret",
        data_dir=tmp_path / "data",
        _env_file=None,
    )
    from sahayak.ingest.cache import RequestCache
    from sahayak.ingest.s1 import SentinelHubClient

    session = FakeSession(
        [
            FakeResponse(200, payload={"access_token": "token-one", "expires_in": 600}),
            FakeResponse(401),
            FakeResponse(200, payload={"access_token": "token-two", "expires_in": 600}),
            FakeResponse(200, content=b'{"features": [], "context": {}}'),
        ]
    )
    client = SentinelHubClient(settings, RequestCache(settings), session=session)

    result = client.catalog_search({"collections": ["sentinel-1-grd"]})

    assert result["features"] == []
    assert len(session.calls) == 4
    assert session.calls[1][1]["headers"]["Authorization"] == "Bearer token-one"
    assert session.calls[3][1]["headers"]["Authorization"] == "Bearer token-two"
    assert "private-client-secret" not in repr(client.token_client)
