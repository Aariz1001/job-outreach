"""
Follow-up Agent — finds records due for a follow-up and generates messages.

Run with:
    python -m agents.followup_agent          # list pending, print messages
    python -m agents.followup_agent --mark   # also mark them as follow-up sent
"""
from __future__ import annotations

import sys
from rich.console import Console
from rich.panel import Panel

from agents.tracker import get_pending_followups, mark_followup_sent
from agents.message_agent import generate_followup

console = Console()


def run(mark: bool = False, days: int = 4) -> None:
    pending = get_pending_followups(days=days)

    if not pending:
        console.print(f"[green]No follow-ups due (threshold: {days} days).[/green]")
        return

    console.print(f"[yellow]{len(pending)} follow-up(s) due:[/yellow]\n")

    for rec in pending:
        follow_msg = generate_followup(
            person_name=rec["name"],
            company=rec["company"],
            original_message=rec["message"],
            days_since=days,
        )
        console.print(
            Panel(
                follow_msg,
                title=f"[bold]{rec['name']} @ {rec['company']}[/bold]  "
                      f"[dim](ID: {rec['id'][:8]})[/dim]",
                subtitle=f"Channel: {rec['channel']}",
            )
        )

        if mark:
            mark_followup_sent(rec["id"], follow_msg)
            console.print(f"  [green]✓ Marked follow-up sent for {rec['id'][:8]}[/green]\n")
        else:
            console.print(
                f"  [dim]Run with --mark to record as sent (ID: {rec['id'][:8]})[/dim]\n"
            )


if __name__ == "__main__":
    mark_flag = "--mark" in sys.argv
    run(mark=mark_flag)
