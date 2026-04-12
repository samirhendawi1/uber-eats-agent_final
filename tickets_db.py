"""Support ticket database backed by SQLite. Tickets persist across sessions."""

from __future__ import annotations
import json, random, sqlite3, string, time
from config import TICKETS_DB


def _connect():
    conn = sqlite3.connect(TICKETS_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS tickets (
        ticket_id TEXT PRIMARY KEY,
        customer_id TEXT,
        order_id TEXT,
        ticket_type TEXT,
        status TEXT,
        details TEXT,
        resolution TEXT,
        created_at REAL,
        updated_at REAL
    )""")
    conn.commit()
    return conn


def generate_ticket_id() -> str:
    return "UE-" + "".join(random.choices(string.digits, k=6))


def create_ticket(
    customer_id: str,
    order_id: str,
    ticket_type: str,
    details: dict,
    status: str = "Open",
    resolution: str | None = None,
) -> dict:
    """Create a new ticket and return it."""
    tid = generate_ticket_id()
    now = time.time()
    conn = _connect()
    conn.execute(
        "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tid, customer_id, order_id, ticket_type, status, json.dumps(details), resolution, now, now),
    )
    conn.commit()
    conn.close()
    return get_ticket(tid)


def get_ticket(ticket_id: str) -> dict | None:
    """Look up a ticket by ID."""
    conn = _connect()
    row = conn.execute(
        "SELECT ticket_id, customer_id, order_id, ticket_type, status, details, resolution, created_at, updated_at FROM tickets WHERE ticket_id = ?",
        (ticket_id.strip().upper(),),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "ticket_id": row[0],
        "customer_id": row[1],
        "order_id": row[2],
        "ticket_type": row[3],
        "status": row[4],
        "details": json.loads(row[5]),
        "resolution": row[6],
        "created_at": row[7],
        "updated_at": row[8],
    }


def get_tickets_by_customer(customer_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT ticket_id, customer_id, order_id, ticket_type, status, details, resolution, created_at, updated_at FROM tickets WHERE customer_id = ? ORDER BY created_at DESC",
        (customer_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "ticket_id": r[0], "customer_id": r[1], "order_id": r[2],
            "ticket_type": r[3], "status": r[4], "details": json.loads(r[5]),
            "resolution": r[6], "created_at": r[7], "updated_at": r[8],
        }
        for r in rows
    ]


def get_tickets_by_order(order_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT ticket_id, customer_id, order_id, ticket_type, status, details, resolution, created_at, updated_at FROM tickets WHERE order_id = ? ORDER BY created_at DESC",
        (order_id,),
    ).fetchall()
    conn.close()
    return [
        {
            "ticket_id": r[0], "customer_id": r[1], "order_id": r[2],
            "ticket_type": r[3], "status": r[4], "details": json.loads(r[5]),
            "resolution": r[6], "created_at": r[7], "updated_at": r[8],
        }
        for r in rows
    ]


def update_ticket(ticket_id: str, status: str | None = None, resolution: str | None = None):
    conn = _connect()
    if status:
        conn.execute("UPDATE tickets SET status = ?, updated_at = ? WHERE ticket_id = ?", (status, time.time(), ticket_id))
    if resolution:
        conn.execute("UPDATE tickets SET resolution = ?, updated_at = ? WHERE ticket_id = ?", (resolution, time.time(), ticket_id))
    conn.commit()
    conn.close()
