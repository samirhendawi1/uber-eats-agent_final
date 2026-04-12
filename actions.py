"""
Tool execution engine with RAG-informed resolution.
All RAG doc access uses getattr() for safety across environments.
"""

from __future__ import annotations
import random
from memory import get_state, set_state
from orders_db import get_order, update_order_status, update_order_refund, update_order
from tickets_db import create_ticket, get_ticket, get_tickets_by_order, get_tickets_by_customer
from tools import TOOLS
from vectorstore import retrieve as rag_retrieve
from llm_client import chat_completion


# ── Helpers for safe doc access ──────────────────────────────────────

def _doc_text(d) -> str:
    return getattr(d, "page_content", str(d))

def _doc_meta(d) -> dict:
    return getattr(d, "metadata", {})

def _doc_title(d) -> str:
    return _doc_meta(d).get("title", "Unknown")


# ── RAG-informed resolution engine ───────────────────────────────────

RESOLUTION_PROMPT = """You are an Uber Eats support resolution engine. Based on the policy documents
retrieved from our knowledge base AND the specific order details, determine the correct resolution.

## Retrieved Policy Documents
{rag_context}

## Order Details
- Order ID: {order_id}
- Restaurant: {restaurant}
- Items: {items}
- Total: {total}
- Status: {status}
- Order Time: {order_time}
- Delivery Time: {delivery_time}
- Driver: {driver}

## Issue Reported
- Type: {issue_type}
- Details: {issue_details}

## Instructions
Based on the policy documents above, determine:
1. What resolution is the customer entitled to? (full refund, partial refund, credit, investigation needed)
2. What is the refund amount if applicable?
3. Should this be auto-resolved or does it need investigation?
4. What timeline should the customer expect?

Respond naturally as a support agent delivering the resolution to the customer.
Include the specific policy reasoning (e.g., "Per our missing items policy...").
Be empathetic, specific about amounts/timelines, and reference the policy.
Do NOT use JSON. Respond in natural customer-facing language with markdown formatting."""


def _rag_resolve(order: dict, issue_type: str, issue_details: str) -> tuple[str, str, str]:
    """Use RAG to retrieve policies, then LLM to determine resolution."""
    search_queries = [issue_type, f"{issue_type} refund policy", f"uber eats {issue_details}"]

    all_docs = []
    seen = set()
    for q in search_queries:
        docs = rag_retrieve(q, k=3)
        for d in docs:
            doc_id = _doc_meta(d).get("id", _doc_text(d)[:50])
            if doc_id not in seen:
                seen.add(doc_id)
                all_docs.append(d)

    rag_context = "\n\n".join(
        f"[Policy: {_doc_title(d)} | Category: {_doc_meta(d).get('category', '')}]\n{_doc_text(d)}"
        for d in all_docs[:6]
    )

    prompt = RESOLUTION_PROMPT.format(
        rag_context=rag_context,
        order_id=order["order_id"],
        restaurant=order["restaurant"],
        items=", ".join(order["items"]),
        total=order["total"],
        status=order["status"],
        order_time=order["order_time"],
        delivery_time=order["delivery_time"] or "N/A",
        driver=order["driver"],
        issue_type=issue_type,
        issue_details=issue_details,
    )

    resolution_text = chat_completion([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=512)

    # Determine status based on issue
    issue_lower = issue_details.lower() + " " + issue_type.lower()
    if any(kw in issue_lower for kw in ["not delivered", "never arrived"]):
        status = "Investigating"
        summary = f"Investigation initiated for order {order['order_id']}"
    elif any(kw in issue_lower for kw in ["damaged", "spilled"]):
        status = "Auto-Resolved"
        summary = f"Full refund of {order['total']} issued for damaged order"
    elif "late" in issue_lower:
        status = "Auto-Resolved"
        summary = "Uber Eats credit issued for late delivery"
    elif any(kw in issue_lower for kw in ["cold", "quality", "stale"]):
        status = "Auto-Resolved"
        summary = f"Partial refund issued for food quality issue"
    elif any(kw in issue_lower for kw in ["missing", "not included"]):
        status = "Auto-Resolved"
        summary = "Refund initiated for missing items"
    elif any(kw in issue_lower for kw in ["wrong", "incorrect"]):
        status = "Under Review"
        summary = "Wrong items report under review"
    else:
        status = "Under Review"
        summary = "Issue reported and under review"

    sources = [_doc_title(d) for d in all_docs[:3]]
    source_text = "\n\n**Policies referenced:** " + ", ".join(f"_{s}_" for s in sources if s and s != "Unknown")

    return resolution_text + source_text, status, summary


# ── Parameter collection ─────────────────────────────────────────────

def get_next_missing_param(session_id: str, tool_name: str) -> str | None:
    collected = get_state(session_id, f"tool_{tool_name}_params", {})
    for p in TOOLS[tool_name]["required_params"]:
        if p not in collected:
            return p
    return None


def get_param_prompt(session_id: str, tool_name: str) -> str:
    param = get_next_missing_param(session_id, tool_name)
    if not param:
        return ""
    prompt = TOOLS[tool_name]["param_prompts"][param]
    if param == "order_id":
        prompt += "\n\n*You can find your order ID in the Orders tab in the sidebar.*"
    return prompt


def store_param(session_id: str, tool_name: str, param: str, value: str):
    collected = get_state(session_id, f"tool_{tool_name}_params", {})
    collected[param] = value
    set_state(session_id, f"tool_{tool_name}_params", collected)


def clear_tool(session_id: str, tool_name: str):
    set_state(session_id, f"tool_{tool_name}_params", {})
    set_state(session_id, "current_tool", None)


# ── Tool Executors ───────────────────────────────────────────────────

def execute_tool(session_id: str, tool_name: str, customer_id: str = "") -> str:
    params = get_state(session_id, f"tool_{tool_name}_params", {})
    executors = {
        "track_order": _exec_track_order,
        "check_refund_status": _exec_check_refund,
        "view_order_details": _exec_view_order,
        "contact_driver": _exec_contact_driver,
        "apply_promo_code": _exec_apply_promo,
        "cancel_order": _exec_cancel_order,
        "report_missing_items": _exec_report_missing,
        "report_wrong_items": _exec_report_wrong,
        "report_delivery_problem": _exec_report_delivery,
        "contact_support": _exec_contact_support,
        "lookup_ticket": _exec_lookup_ticket,
    }
    executor = executors.get(tool_name)
    if not executor:
        clear_tool(session_id, tool_name)
        return "Sorry, that action isn't available right now."
    result = executor(params, customer_id)
    clear_tool(session_id, tool_name)
    return result


# ── INSTANT executors ────────────────────────────────────────────────

def _exec_track_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found. Please check the order ID."
    status = order["status"]
    if "In Progress" in status:
        stages = ["Order confirmed", "Restaurant preparing", "**Driver picking up**", "On the way", "Arriving"]
        eta = random.randint(10, 30)
        return (
            f"**Live Tracking - Order #{order['order_id']}**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Items:** {', '.join(order['items'])}\n\n"
            f"**Progress:**\n" + "\n".join(f"  {s}" for s in stages) + f"\n\n"
            f"**Driver:** {order['driver']}\n**ETA:** ~{eta} minutes"
        )
    elif "Delivered" in status:
        return (
            f"**Order #{order['order_id']} was delivered**\n\n"
            f"**Restaurant:** {order['restaurant']}\n**Delivered at:** {order['delivery_time']}\n"
            f"**Driver:** {order['driver']}\n\nIf you had an issue, I can help you report it."
        )
    elif "Cancelled" in status:
        return (
            f"**Order #{order['order_id']} was cancelled**\n\n"
            f"**Restaurant:** {order['restaurant']}\n**Refund:** {order['refund_status'] or 'Processing'}"
        )
    return f"**Order #{order['order_id']}** - Status: {status}"


def _exec_check_refund(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    tickets = get_tickets_by_order(params["order_id"])

    # RAG for refund policy
    docs = rag_retrieve("refund status processing time", k=3)
    policy_context = " ".join(_doc_text(d) for d in docs)[:500]

    refund = order["refund_status"]
    if refund:
        result = (
            f"**Refund Status - Order #{order['order_id']}**\n\n"
            f"**Restaurant:** {order['restaurant']}\n**Order total:** {order['total']}\n"
            f"**Refund:** {refund}\n\n"
        )
        try:
            explanation = chat_completion(
                [{"role": "user", "content": f"Based on this Uber Eats refund policy: {policy_context}\n\nThe customer's refund status is: {refund}\nGive a brief 1-2 sentence update about what they should expect. Reference the policy."}],
                temperature=0.3, max_tokens=150,
            )
            result += explanation
        except Exception:
            result += "Your refund is being processed. Please allow 3-5 business days."
    else:
        result = f"**Order #{order['order_id']}** ({order['restaurant']})\n\nNo refund requested. If something was wrong, I can help you report it."

    if tickets:
        result += "\n\n**Related tickets:** " + ", ".join(f"`{t['ticket_id']}`" for t in tickets)
    return result


def _exec_view_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    items_str = "\n".join(f"  - {item}" for item in order["items"])
    result = (
        f"**Order #{order['order_id']} - Full Details**\n\n"
        f"**Restaurant:** {order['restaurant']}\n**Status:** {order['status']}\n"
        f"**Items:**\n{items_str}\n\n"
        f"**Total:** {order['total']}\n**Ordered:** {order['order_time']}\n"
        f"**Delivered:** {order['delivery_time'] or 'Pending'}\n"
        f"**Driver:** {order['driver']}\n**Payment:** {order['payment_method']}\n"
        f"**Address:** {order['address']}"
    )
    if order["special_instructions"]:
        result += f"\n**Notes:** {order['special_instructions']}"
    if order["refund_status"]:
        result += f"\n\n**Refund:** {order['refund_status']}"
    tickets = get_tickets_by_order(params["order_id"])
    if tickets:
        result += "\n**Tickets:** " + ", ".join(f"`{t['ticket_id']}` ({t['status']})" for t in tickets)
    return result


def _exec_contact_driver(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    if "In Progress" not in order["status"]:
        return f"Order #{order['order_id']} is **{order['status']}**, so the driver is no longer active."
    return (
        f"**Message sent to {order['driver']}!**\n\n"
        f"**Your message:** \"{params['message']}\"\n\n"
        f"{order['driver']} has been notified and typically responds within a few minutes."
    )


def _exec_apply_promo(params: dict, customer_id: str) -> str:
    code = params["promo_code"].strip().upper()
    valid = {"WELCOME10": "10% off your next order (up to $5)", "FREEDELIVERY": "Free delivery on next 3 orders", "SAVE5": "$5 off orders over $25"}
    if code in valid:
        return f"**Promo code applied!**\n\n**Code:** `{code}`\n**Benefit:** {valid[code]}"
    return f"Promo code `{code}` is invalid or expired."


def _exec_cancel_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    confirm = params.get("confirm", "").lower().strip()
    if confirm not in ("yes", "y", "confirm", "sure", "ok"):
        return "No problem, your order has **not** been cancelled."

    # RAG for cancellation policy
    docs = rag_retrieve("cancel order cancellation fee policy", k=3)
    policy_context = "\n".join(f"[{_doc_title(d)}]: {_doc_text(d)}" for d in docs)

    update_order_status(params["order_id"], "Cancelled")
    update_order_refund(params["order_id"], f"Refund pending - cancellation of {order['total']}")

    try:
        result = chat_completion(
            [{"role": "user", "content": f"Based on these Uber Eats cancellation policies:\n{policy_context}\n\nOrder: #{order['order_id']} from {order['restaurant']}, total {order['total']}, status: {order['status']}\nCustomer confirmed cancellation.\n\nWrite a brief response confirming the cancellation and explaining any fees/refund based on status and policy."}],
            temperature=0.3, max_tokens=256,
        )
    except Exception:
        result = f"**Order #{order['order_id']} has been cancelled.** A refund of {order['total']} is being processed."
    return result


# ── RAG-INFORMED TICKET executors ────────────────────────────────────

def _exec_report_missing(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    resolution_text, status, summary = _rag_resolve(order, "Missing Items", f"Customer reports missing items: {params['missing_items']}")
    ticket = create_ticket(customer_id=customer_id, order_id=params["order_id"], ticket_type="Missing Items",
        details={"missing_items": params["missing_items"], "order_items": ", ".join(order["items"])}, status=status, resolution=summary)
    update_order_status(params["order_id"], "Delivered - Reported Issue")
    update_order_refund(params["order_id"], summary)
    return f"**Ticket `{ticket['ticket_id']}` created for Order #{order['order_id']}**\n\n{resolution_text}\n\n---\n**Ticket reference:** `{ticket['ticket_id']}`"


def _exec_report_wrong(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    resolution_text, status, summary = _rag_resolve(order, "Wrong Items Received", f"Received: {params['wrong_items']}. Expected: {params['expected_items']}")
    ticket = create_ticket(customer_id=customer_id, order_id=params["order_id"], ticket_type="Wrong Items",
        details={"wrong_items": params["wrong_items"], "expected_items": params["expected_items"]}, status=status, resolution=summary)
    update_order_status(params["order_id"], "Delivered - Reported Issue")
    update_order_refund(params["order_id"], summary)
    return f"**Ticket `{ticket['ticket_id']}` created for Order #{order['order_id']}**\n\n{resolution_text}\n\n---\n**Ticket reference:** `{ticket['ticket_id']}`"


def _exec_report_delivery(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    resolution_text, status, summary = _rag_resolve(order, "Delivery Problem", f"Problem reported: {params['problem_type']}")
    ticket = create_ticket(customer_id=customer_id, order_id=params["order_id"], ticket_type=f"Delivery: {params['problem_type'].title()}",
        details={"problem": params["problem_type"], "driver": order["driver"]}, status=status, resolution=summary)
    update_order_status(params["order_id"], "Delivered - Reported Issue")
    update_order_refund(params["order_id"], summary)
    return f"**Ticket `{ticket['ticket_id']}` created for Order #{order['order_id']}**\n\n{resolution_text}\n\n---\n**Ticket reference:** `{ticket['ticket_id']}`"


# ── ESCALATION ───────────────────────────────────────────────────────

def _exec_contact_support(params: dict, customer_id: str) -> str:
    queue_time = random.randint(2, 15)
    ticket = create_ticket(customer_id=customer_id, order_id="N/A", ticket_type="Support Escalation",
        details={"issue": params["issue_summary"], "urgency": params["urgency"]}, status="Escalated to Agent")
    return (
        f"**Connected to support!**\n\n**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Issue:** {params['issue_summary']}\n**Priority:** {params['urgency'].capitalize()}\n"
        f"**Estimated wait:** {queue_time} minutes\n\n**Reference:** `{ticket['ticket_id']}`"
    )


def _exec_lookup_ticket(params: dict, customer_id: str) -> str:
    ticket = get_ticket(params["ticket_id"])
    if not ticket:
        return f"Ticket `{params['ticket_id']}` not found. Format: UE-XXXXXX."
    from datetime import datetime
    created = datetime.fromtimestamp(ticket["created_at"]).strftime("%Y-%m-%d %H:%M")
    result = (
        f"**Ticket {ticket['ticket_id']}**\n\n**Type:** {ticket['ticket_type']}\n"
        f"**Status:** {ticket['status']}\n**Order:** #{ticket['order_id']}\n**Created:** {created}\n"
    )
    if ticket["details"]:
        for k, v in ticket["details"].items():
            result += f"**{k.replace('_', ' ').title()}:** {v}\n"
    if ticket["resolution"]:
        result += f"\n**Resolution:** {ticket['resolution']}"
    return result
