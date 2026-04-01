"""
Tracker — thin JSON-based persistence layer for outreach records.

Schema per record:
{
    "id": "uuid",
    "name": "Jordan Smith",
    "role": "Founder",
    "company": "Vapi AI",
    "linkedin": "https://linkedin.com/in/...",
    "email": "",
    "channel": "linkedin",
    "message": "...",
    "sent_at": "2026-04-01T12:00:00",
    "replied": false,
    "reply_at": null,
    "follow_up_sent": false,
    "follow_up_at": null,
    "notes": ""
}
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.config import TRACKER_FILE


def _load() -> list[dict[str, Any]]:
    if TRACKER_FILE.exists():
        return json.loads(TRACKER_FILE.read_text())
    return []


def _save(records: list[dict[str, Any]]) -> None:
    TRACKER_FILE.write_text(json.dumps(records, indent=2))


def add_record(
    name: str,
    role: str,
    company: str,
    channel: str,
    message: str,
    linkedin: str = "",
    email: str = "",
    notes: str = "",
) -> dict[str, Any]:
    records = _load()
    rec: dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "name": name,
        "role": role,
        "company": company,
        "linkedin": linkedin,
        "email": email,
        "channel": channel,
        "message": message,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "replied": False,
        "reply_at": None,
        "follow_up_sent": False,
        "follow_up_at": None,
        "notes": notes,
    }
    records.append(rec)
    _save(records)
    return rec


def mark_replied(record_id: str, notes: str = "") -> bool:
    records = _load()
    for rec in records:
        if rec["id"] == record_id:
            rec["replied"] = True
            rec["reply_at"] = datetime.now(timezone.utc).isoformat()
            if notes:
                rec["notes"] = notes
            _save(records)
            return True
    return False


def mark_followup_sent(record_id: str, followup_message: str = "") -> bool:
    records = _load()
    for rec in records:
        if rec["id"] == record_id:
            rec["follow_up_sent"] = True
            rec["follow_up_at"] = datetime.now(timezone.utc).isoformat()
            if followup_message:
                rec["notes"] = (rec.get("notes") or "") + f"\n[FOLLOWUP] {followup_message}"
            _save(records)
            return True
    return False


def get_pending_followups(days: int = 4) -> list[dict[str, Any]]:
    """Return records sent ≥ `days` ago with no reply and no follow-up yet."""
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    result = []
    for rec in _load():
        if rec["replied"] or rec["follow_up_sent"]:
            continue
        sent = datetime.fromisoformat(rec["sent_at"])
        if sent <= cutoff:
            result.append(rec)
    return result


def get_all() -> list[dict[str, Any]]:
    return _load()


def stats() -> dict[str, int]:
    records = _load()
    return {
        "total": len(records),
        "replied": sum(1 for r in records if r["replied"]),
        "pending_followup": len(get_pending_followups()),
        "follow_up_sent": sum(1 for r in records if r["follow_up_sent"]),
        "no_reply_no_followup": sum(
            1 for r in records if not r["replied"] and not r["follow_up_sent"]
        ),
    }
