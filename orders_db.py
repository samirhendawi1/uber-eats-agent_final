"""
Order database backed by SQLite. Seeds from CSV on first run,
then supports real-time updates when tools execute.

Updated fields on actions:
  - status (e.g., "In Progress" -> "Cancelled")
  - refund_status (e.g., None -> "Refund issued - $24.87")
  - rating
"""

from __future__ import annotations
import csv, os, sqlite3
from config import DATA_DIR

ORDERS_DB = os.path.join(DATA_DIR, "orders.db")
ORDERS_CSV = os.path.join(DATA_DIR, "orders.csv")


def _connect():
    conn = sqlite3.connect(ORDERS_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT,
        customer_name TEXT,
        customer_email TEXT,
        restaurant TEXT,
        items TEXT,
        total TEXT,
        status TEXT,
        order_time TEXT,
        delivery_time TEXT,
        address TEXT,
        payment_method TEXT,
        driver TEXT,
        refund_status TEXT,
        rating TEXT,
        special_instructions TEXT
    )""")
    conn.commit()
    return conn


def _seed_if_empty():
    """Seed SQLite from CSV on first run."""
    conn = _connect()
    count = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    if count > 0:
        conn.close()
        return
    with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conn.execute(
                """INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["order_id"], row["customer_id"], row["customer_name"],
                    row["customer_email"], row["restaurant"], row["items"],
                    row["total"], row["status"], row["order_time"],
                    row["delivery_time"], row["address"], row["payment_method"],
                    row["driver"], row["refund_status"], row["rating"],
                    row["special_instructions"],
                ),
            )
    conn.commit()
    conn.close()


def _row_to_dict(row) -> dict:
    """Convert a sqlite3.Row to a dict with items as a list."""
    d = dict(row)
    d["items"] = [i.strip() for i in (d["items"] or "").split(",")]
    d["refund_status"] = d["refund_status"] or None
    d["rating"] = d["rating"] or None
    d["delivery_time"] = d["delivery_time"] or None
    d["driver"] = d["driver"] or "N/A"
    d["special_instructions"] = d["special_instructions"] or None
    return d


def get_order(order_id: str) -> dict | None:
    _seed_if_empty()
    conn = _connect()
    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id.strip(),)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_orders_by_customer(customer_id: str) -> list[dict]:
    _seed_if_empty()
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM orders WHERE customer_id = ? ORDER BY order_time DESC",
        (customer_id,),
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_all_orders() -> list[dict]:
    _seed_if_empty()
    conn = _connect()
    rows = conn.execute("SELECT * FROM orders ORDER BY order_time DESC").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


# ── Update functions (called by actions.py) ──────────────────────────

def update_order_status(order_id: str, new_status: str):
    """Update order status (e.g., 'In Progress' -> 'Cancelled')."""
    _seed_if_empty()
    conn = _connect()
    conn.execute("UPDATE orders SET status = ? WHERE order_id = ?", (new_status, order_id.strip()))
    conn.commit()
    conn.close()


def update_order_refund(order_id: str, refund_status: str):
    """Update refund status (e.g., 'Approved - $8.99 refund')."""
    _seed_if_empty()
    conn = _connect()
    conn.execute("UPDATE orders SET refund_status = ? WHERE order_id = ?", (refund_status, order_id.strip()))
    conn.commit()
    conn.close()


def update_order(order_id: str, **fields):
    """Update arbitrary fields on an order."""
    _seed_if_empty()
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [order_id.strip()]
    conn = _connect()
    conn.execute(f"UPDATE orders SET {set_clause} WHERE order_id = ?", values)
    conn.commit()
    conn.close()
