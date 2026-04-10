"""Order database backed by CSV."""

from __future__ import annotations
import csv, os
from config import DATA_DIR

ORDERS_CSV = os.path.join(DATA_DIR, "orders.csv")

_cache: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _cache
    if _cache is not None:
        return _cache
    _cache = {}
    with open(ORDERS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["items"] = [i.strip() for i in row["items"].split(",")]
            row["refund_status"] = row["refund_status"] or None
            row["rating"] = row["rating"] or None
            row["delivery_time"] = row["delivery_time"] or None
            row["driver"] = row["driver"] or "N/A"
            row["special_instructions"] = row["special_instructions"] or None
            _cache[row["order_id"]] = row
    return _cache


def get_order(order_id: str) -> dict | None:
    return _load().get(order_id.strip())


def get_orders_by_customer(customer_id: str) -> list[dict]:
    return sorted(
        [o for o in _load().values() if o["customer_id"] == customer_id],
        key=lambda o: o["order_time"],
        reverse=True,
    )


def get_all_orders() -> list[dict]:
    return list(_load().values())
