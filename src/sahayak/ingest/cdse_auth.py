"""OAuth client-credentials authentication for Copernicus Data Space."""

from __future__ import annotations

import math
import re
import time
from typing import Any

import requests
from pydantic import SecretStr

from sahayak.config import Settings

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
TOKEN_TIMEOUT_SECONDS = 30
TOKEN_REFRESH_MARGIN_SECONDS = 60


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
            response = self._session.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=TOKEN_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            message = "Could not reach the Copernicus Data Space token service."
            raise AuthenticationError(message) from error

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
                f"Copernicus Data Space token request failed (HTTP {response.status_code}).",
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
