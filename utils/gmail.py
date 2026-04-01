"""Gmail compose URL generator — builds a mailto-style link that opens
Gmail with the entire email pre-filled (to, subject, body).

The URL format is:
  https://mail.google.com/mail/?view=cm&fs=1&to=...&su=...&body=...
"""
from __future__ import annotations
import urllib.parse


def gmail_compose_link(
    subject: str,
    body: str,
    to: str = "",
    cc: str = "",
) -> str:
    """
    Return a Gmail compose URL with the draft pre-filled.

    Args:
        subject: Email subject line.
        body:    Email body (plain text, newlines are preserved).
        to:      Recipient email address (optional — can fill in Gmail).
        cc:      CC address (optional).
    """
    params: dict[str, str] = {
        "view": "cm",
        "fs": "1",
    }
    if to:
        params["to"] = to
    if cc:
        params["cc"] = cc
    if subject:
        params["su"] = subject
    if body:
        params["body"] = body

    return "https://mail.google.com/mail/?" + urllib.parse.urlencode(params)


def parse_email_draft(draft: str) -> tuple[str, str]:
    """
    Split a generated email draft into (subject, body).
    Expects the draft to start with a line like "Subject: ..."
    """
    lines = draft.strip().splitlines()
    subject = ""
    body_lines: list[str] = []
    body_started = False

    for line in lines:
        if not body_started and line.lower().startswith("subject:"):
            subject = line.split(":", 1)[1].strip()
            body_started = True
        elif body_started:
            body_lines.append(line)

    # If no Subject: line found, treat entire draft as body
    if not subject:
        return "", draft.strip()

    # Skip leading blank line after subject
    body = "\n".join(body_lines).strip()
    return subject, body
