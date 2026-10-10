"""
Authentication middleware for TensorForge 2.0 API.
Implements:
  - Constant-time comparison using hmac.compare_digest
  - X-API-Key and Authorization: Bearer header extraction
  - Rejection if API_KEY env var is missing or empty
  - Never logs the secret key
"""

import os
import hmac

def get_configured_api_key() -> str:
    """Reads expected API key from environment variable API_KEY."""
    key = os.environ.get("API_KEY", "").strip()
    return key

def extract_api_key_from_headers(headers: dict) -> str:
    """
    Extracts key from 'X-API-Key' or 'Authorization: Bearer <key>'.
    Case-insensitive header lookup.
    """
    # 1. Check X-API-Key
    for h, v in headers.items():
        if h.lower() == "x-api-key":
            return v.strip()
    
    # 2. Check Authorization: Bearer <key>
    for h, v in headers.items():
        if h.lower() == "authorization":
            parts = v.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
    
    return ""

def verify_request_auth(headers: dict) -> tuple[bool, str]:
    """
    Validates authentication.
    Returns (is_valid, error_message).
    """
    configured_key = get_configured_api_key()
    if not configured_key:
        # If API_KEY is not configured, refuse to serve prediction endpoints
        return False, "Service authentication is unconfigured (API_KEY missing). Requests refused."

    provided_key = extract_api_key_from_headers(headers)
    if not provided_key:
        return False, "Missing API key. Send X-API-Key or Authorization Bearer."

    # Constant-time comparison
    if hmac.compare_digest(provided_key.encode("utf-8"), configured_key.encode("utf-8")):
        return True, ""
    
    return False, "Invalid API key."

