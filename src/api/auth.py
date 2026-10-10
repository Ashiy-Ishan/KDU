import os
import hmac

def get_configured_api_key() -> str:
    return os.environ.get("API_KEY", "").strip()

def extract_api_key_from_headers(headers: dict) -> str:
    for h, v in headers.items():
        if h.lower() == "x-api-key":
            return v.strip()
    for h, v in headers.items():
        if h.lower() == "authorization":
            parts = v.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
    return ""

def verify_request_auth(headers: dict) -> tuple[bool, str]:
    configured_key = get_configured_api_key()
    if not configured_key:
        return False, "Service authentication is unconfigured (API_KEY missing). Requests refused."

    provided_key = extract_api_key_from_headers(headers)
    if not provided_key:
        return False, "Missing API key. Send X-API-Key or Authorization Bearer."

    if hmac.compare_digest(provided_key.encode("utf-8"), configured_key.encode("utf-8")):
        return True, ""
    
    return False, "Invalid API key."
