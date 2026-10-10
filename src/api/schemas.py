"""
Schema validation helpers matching TensorForge 2.0 OpenAPI v2.1.0 specifications.
Performs strict validation and generates structured error details.
"""

import re

VALID_CHANNELS = {"email", "chat", "call_transcript"}
NON_WHITESPACE_REGEX = re.compile(r"\S")

def validate_single_ticket(data: dict, require_ticket_id: bool = False, index: int = None) -> list[dict]:
    """
    Validates a single ticket object.
    Returns a list of issue dicts: [{"field": ..., "issue": ... (, "index": ...)}]
    """
    issues = []

    def add_issue(field: str, issue: str):
        item = {"field": field, "issue": issue}
        if index is not None:
            item["index"] = index
        issues.append(item)

    if not isinstance(data, dict):
        add_issue("ticket", "must be an object")
        return issues

    # 1. ticket_id
    ticket_id = data.get("ticket_id")
    if require_ticket_id:
        if ticket_id is None or ticket_id == "":
            add_issue("ticket_id", "is required")
        elif not isinstance(ticket_id, str):
            add_issue("ticket_id", "must be a string")
        elif len(ticket_id) > 64:
            add_issue("ticket_id", "must not exceed 64 characters")
    else:
        if ticket_id is not None:
            if not isinstance(ticket_id, str):
                add_issue("ticket_id", "must be a string")
            elif len(ticket_id) > 64:
                add_issue("ticket_id", "must not exceed 64 characters")

    # 2. channel
    if "channel" not in data or data["channel"] is None:
        add_issue("channel", "is required")
    elif not isinstance(data["channel"], str):
        add_issue("channel", "must be a string")
    elif data["channel"] not in VALID_CHANNELS:
        add_issue("channel", f"must be one of {', '.join(sorted(VALID_CHANNELS))}")

    # 3. text
    if "text" not in data or data["text"] is None:
        add_issue("text", "is required")
    elif not isinstance(data["text"], str):
        add_issue("text", "must be a string")
    else:
        text = data["text"]
        if len(text) == 0 or not NON_WHITESPACE_REGEX.search(text):
            add_issue("text", "must contain at least one non-whitespace character")
        elif len(text) > 10000:
            add_issue("text", "must not exceed 10000 characters")

    # 4. subject
    if "subject" in data and data["subject"] is not None:
        if not isinstance(data["subject"], str):
            add_issue("subject", "must be a string")
        elif len(data["subject"]) > 500:
            add_issue("subject", "must not exceed 500 characters")

    return issues

def validate_batch_tickets(tickets: list, max_items: int = 100) -> tuple[bool, list[dict]]:
    """
    Validates a list of tickets for batch requests.
    Enforces count bounds, per-item schemas, and ticket_id uniqueness.
    """
    details = []

    if not isinstance(tickets, list):
        details.append({"field": "tickets", "issue": "must be an array"})
        return False, details

    if len(tickets) < 1 or len(tickets) > max_items:
        details.append({
            "field": "tickets",
            "issue": f"must contain between 1 and {max_items} items"
        })
        return False, details

    seen_ids = set()
    for idx, item in enumerate(tickets):
        item_issues = validate_single_ticket(item, require_ticket_id=True, index=idx)
        details.extend(item_issues)

        # Check duplicate ticket_id
        if isinstance(item, dict):
            t_id = item.get("ticket_id")
            if t_id and isinstance(t_id, str):
                if t_id in seen_ids:
                    details.append({
                        "index": idx,
                        "field": "ticket_id",
                        "issue": "duplicate of an earlier item"
                    })
                else:
                    seen_ids.add(t_id)

    if details:
        return False, details
    return True, []

