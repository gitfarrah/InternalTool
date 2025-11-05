from __future__ import annotations

import os
import requests
import streamlit as st
from slack_sdk.web import WebClient


def _get_secret(key: str, default: str = "") -> str:
    """Fetch config from Streamlit secrets or environment with fallback."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)


def _require_oauth_config() -> dict:
    client_id = _get_secret("SLACK_CLIENT_ID")
    client_secret = _get_secret("SLACK_CLIENT_SECRET")
    redirect_uri = _get_secret("SLACK_REDIRECT_URI")

    missing = []
    if not client_id:
        missing.append("SLACK_CLIENT_ID")
    if not client_secret:
        missing.append("SLACK_CLIENT_SECRET")
    if not redirect_uri:
        missing.append("SLACK_REDIRECT_URI")

    if missing:
        raise RuntimeError(
            "Missing Slack OAuth config: "
            + ", ".join(missing)
            + ". Set them in st.secrets or environment variables."
        )
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
    }


def get_oauth_url() -> str:
    """Generate Slack OAuth authorization URL."""
    cfg = _require_oauth_config()

    # Request USER token scopes (use user_scope param)
    user_scopes = [
        "channels:history",
        "channels:read",
        "groups:history",
        "groups:read",
        "search:read",
        "users:read",
    ]

    return (
        "https://slack.com/oauth/v2/authorize?"
        f"client_id={cfg['client_id']}&"
        f"user_scope={','.join(user_scopes)}&"
        f"redirect_uri={cfg['redirect_uri']}"
    )


def exchange_code_for_token(code: str) -> str:
    """Exchange OAuth code for user access token."""
    cfg = _require_oauth_config()
    response = requests.post(
        "https://slack.com/api/oauth.v2.access",
        data={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "code": code,
            "redirect_uri": cfg["redirect_uri"],
        },
        timeout=15,
    )
    data = response.json()
    if not data.get("ok"):
        raise Exception(f"OAuth failed: {data.get('error')}")
    return data["authed_user"]["access_token"]


def get_user_info(token: str) -> dict:
    """Get authenticated user information."""
    client = WebClient(token=token)
    # users_identity requires appropriate scopes; fallback to auth_test + users_info if needed
    try:
        identity = client.users_identity()
        return {
            "id": identity["user"]["id"],
            "name": identity["user"].get("name") or identity["user"].get("username", "Unknown"),
            "email": identity["user"].get("email", ""),
        }
    except Exception:
        auth = client.auth_test()
        user_id = auth.get("user_id", "")
        name = auth.get("user", "")
        email = ""
        if user_id:
            try:
                ui = client.users_info(user=user_id)
                profile = (ui.get("user") or {}).get("profile", {})
                email = profile.get("email", "")
                name = profile.get("real_name") or profile.get("display_name") or name
            except Exception:
                pass
        return {"id": user_id, "name": name or "Unknown", "email": email}


def is_token_valid(token: str) -> bool:
    """Check if token is still valid."""
    try:
        client = WebClient(token=token)
        client.auth_test()
        return True
    except Exception:
        return False


__all__ = [
    "get_oauth_url",
    "exchange_code_for_token",
    "get_user_info",
    "is_token_valid",
]

