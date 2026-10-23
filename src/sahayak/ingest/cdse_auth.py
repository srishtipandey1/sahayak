"""OAuth client-credentials authentication for Copernicus Data Space."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import requests
from pydantic import SecretStr

from sahayak.config import Settings

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
TOKEN_REFRESH_MARGIN_SECONDS = 60
HTTP_MAX_ATTEMPTS = 3
HTTP_RETRY_STATUSES = {429, 502, 503, 504}
HTTP_CONNECT_TIMEOUT_SECONDS = 15
HTTP_READ_TIMEOUT_SECONDS = 120
HTTP_USER_AGENT = "Sahayak/0.1.0 (Sentinel-1 ingestion; educational prototype)"
HTTP_TIMEOUT = (HTTP_CONNECT_TIMEOUT_SECONDS, HTTP_READ_TIMEOUT_SECONDS)
HTTP_BACKOFF_BASE_SECONDS = 1.0


class AuthenticationError(RuntimeError):
    """Raised when Copernicus Data Space authentication cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        """Keep safe diagnostic metadata without retaining a response body."""
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class RequestFailure(RuntimeError):
    """A retried HTTP request failed before receiving a response."""


def scrub_sensitive_text(message: str, secrets: Iterable[str] = ()) -> str:
    """Remove supplied credentials and bearer tokens from an exception message."""
    scrubbed = re.sub(
        r"(?i)\bBearer\s+\S+",
        "Bearer [REDACTED]",
        message,
    )
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        scrubbed = scrubbed.replace(secret, "[REDACTED]")
    return scrubbed


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After as seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def post_with_retries(
    session: Any,
    url: str,
    *,
    service_name: str,
    secrets: Iterable[str] = (),
    **request_kwargs: Any,
) -> tuple[Any, int]:
    """POST with bounded retries for transient transport failures and HTTP statuses."""
    headers = dict(request_kwargs.pop("headers", {}))
    headers.setdefault("User-Agent", HTTP_USER_AGENT)
    request_kwargs["headers"] = headers

    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            response = session.post(url, **request_kwargs)
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as error:
            if attempt == HTTP_MAX_ATTEMPTS:
                details = scrub_sensitive_text(str(error), secrets)
                message = (
                    f"{service_name} request failed before a response on attempt "
                    f"{attempt}/{HTTP_MAX_ATTEMPTS}: {type(error).__name__}: {details}"
                )
                raise RequestFailure(message) from error
            time.sleep(HTTP_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
            continue

        if response.status_code not in HTTP_RETRY_STATUSES:
            return response, attempt
        if attempt == HTTP_MAX_ATTEMPTS:
            return response, attempt
        retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
        backoff = HTTP_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
        time.sleep(max(backoff, retry_after or 0.0))

    raise AssertionError("HTTP retry loop terminated unexpectedly.")


class CdseTokenClient:
    """Acquire and reuse a short-lived OAuth access token without exposing it."""

    def __init__(
        self,
        settings: Settings,
        *,
        session: requests.Session | Any | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        """Initialize the token client with settings and an optional mockable session."""
        self._settings = settings
        self._session = session or requests.Session()
        self._clock = clock
        self._token: str | None = None
        self._expires_at = 0.0

    def __repr__(self) -> str:
        """Return a representation that never contains credential material."""
        return "CdseTokenClient(<credentials redacted>)"

    def get_token(self, *, force_refresh: bool = False) -> str:
        """Return a cached token or request a new token with client credentials."""
        if not force_refresh and self._token and self._clock() < self._expires_at:
            return self._token

        client_id = self._secret_value(self._settings.cdse_client_id, "CDSE_CLIENT_ID")
        client_secret = self._secret_value(
            self._settings.cdse_client_secret, "CDSE_CLIENT_SECRET"
        )
        try:
            response, attempt = post_with_retries(
                self._session,
                TOKEN_URL,
                service_name="Copernicus Data Space token request",
                secrets=(client_id, client_secret, self._token or ""),
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=HTTP_TIMEOUT,
            )
        except RequestFailure as error:
            raise AuthenticationError(str(error)) from error

        if response.status_code != 200:
            error_code = None
            try:
                payload = response.json()
            except (ValueError, TypeError):
                payload = None
            if isinstance(payload, dict):
                candidate = payload.get("error_code", payload.get("error"))
                if isinstance(candidate, str) and re.fullmatch(
                    r"[A-Za-z0-9_.-]{1,80}", candidate
                ):
                    error_code = candidate
            raise AuthenticationError(
                "Copernicus Data Space token request failed "
                f"(HTTP {response.status_code}) on attempt {attempt}/{HTTP_MAX_ATTEMPTS}.",
                status_code=response.status_code,
                error_code=error_code,
            )
        try:
            payload = response.json()
            token = payload["access_token"]
            expires_in = float(payload["expires_in"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuthenticationError(
                "Copernicus Data Space token response was missing a valid access token or expiry."
            ) from error
        if (
            not isinstance(token, str)
            or not token
            or expires_in <= 0
            or not math.isfinite(expires_in)
        ):
            raise AuthenticationError(
                "Copernicus Data Space token response was missing a valid access token or expiry."
            )

        self._token = token
        self._expires_at = self._clock() + max(
            0.0, expires_in - TOKEN_REFRESH_MARGIN_SECONDS
        )
        return token

    def invalidate(self) -> None:
        """Discard the current token after a service rejects it."""
        self._token = None
        self._expires_at = 0.0

    @staticmethod
    def _secret_value(secret: SecretStr | None, setting_name: str) -> str:
        """Return a configured secret or raise an error that names the setting."""
        if secret is None or not secret.get_secret_value():
            raise AuthenticationError(
                f"Missing required setting {setting_name} for Copernicus Data Space OAuth."
            )
        return secret.get_secret_value()
