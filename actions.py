"""
Tool execution engine — real Uber Eats support behavior.

Each tool does what Uber Eats would actually do:
  - Instant tools: look up data, return it
  - Action tools: create tickets, update orders, explain policy, direct to forms
  - Escalation: connect to human support

RAG provides policy context for customer-facing messages.
Code executes the actual action (ticket, refund, status update).
"""

from __future__ import annotations
import random
from memory import get_state, set_state
from orders_db import get_order, update_order_status, update_order_refund
from tickets_db import create_ticket, get_ticket, get_tickets_by_order
from tools import TOOLS
from vectorstore import retrieve as rag_retrieve
from llm_client import chat_completion

HELP_URL = "https://help.uber.com/en/ubereats"


# ── Helpers ──────────────────────────────────────────────────────────

def _doc_text(d) -> str:
    return getattr(d, "page_content", str(d))

def _doc_title(d) -> str:
    return getattr(d, "metadata", {}).get("title", "Unknown")

def _get_policy(query: str, k: int = 3) -> str:
    docs = rag_retrieve(query, k=k)
    if not docs:
        return ""
    return "\n".join(f"[{_doc_title(d)}]: {_doc_text(d)}" for d in docs)

def _policy_message(action_done: str, policy: str, order: dict, issue: str) -> str:
    """LLM writes customer message referencing policy + confirming action taken."""
    if not policy:
        return action_done
    try:
        return chat_completion([{"role": "user", "content": (
            f"You are an Uber Eats support agent. Write a brief, empathetic message to the customer.\n\n"
            f"ACTION ALREADY TAKEN: {action_done}\n"
            f"RELEVANT POLICY: {policy[:800]}\n"
            f"ORDER: #{order['order_id']} from {order['restaurant']}, total {order['total']}\n"
            f"ISSUE: {issue}\n\n"
            f"Confirm what was done, reference the policy naturally, mention timelines. "
            f"Do NOT ask the customer to do anything else — the action is complete. "
            f"Keep it to 2-3 sentences. No JSON."
        )}], temperature=0.3, max_tokens=200)
    except Exception:
        return action_done


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


# ── Main executor ────────────────────────────────────────────────────

def execute_tool(session_id: str, tool_name: str, customer_id: str = "") -> str:
    params = get_state(session_id, f"tool_{tool_name}_params", {})
    executors = {
        "track_order": _exec_track_order,
        "check_refund_status": _exec_check_refund,
        "view_order_details": _exec_view_order,
        "contact_driver": _exec_contact_driver,
        "lookup_ticket": _exec_lookup_ticket,
        "cancel_order": _exec_cancel_order,
        "report_missing_items": _exec_report_missing,
        "report_wrong_items": _exec_report_wrong,
        "report_delivery_issue": _exec_report_delivery,
        "contact_support": _exec_contact_support,
    }
    executor = executors.get(tool_name)
    if not executor:
        clear_tool(session_id, tool_name)
        return "Sorry, that action isn't available right now."
    result = executor(params, customer_id)
    clear_tool(session_id, tool_name)
    return result


# ══════════════════════════════════════════════════════════════════════
# INSTANT TOOLS — data lookup, no ticket
# ══════════════════════════════════════════════════════════════════════

def _exec_track_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found. Please check the order ID and try again."
    status = order["status"]
    if "In Progress" in status:
        eta = random.randint(10, 30)
        return (
            f"**Order #{order['order_id']} — Live Tracking**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Items:** {', '.join(order['items'])}\n\n"
            f"**Status:** Your order is being prepared and a driver is on the way.\n"
            f"**Driver:** {order['driver']}\n"
            f"**ETA:** ~{eta} minutes\n\n"
            f"You can message your driver by saying *\"send a message to my driver\"*."
        )
    elif "Delivered" in status:
        return (
            f"**Order #{order['order_id']} — Delivered**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Delivered at:** {order['delivery_time']}\n"
            f"**Driver:** {order['driver']}\n\n"
            f"If something was wrong with this order, I can help you report it."
        )
    elif "Cancelled" in status:
        return (
            f"**Order #{order['order_id']} — Cancelled**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Refund:** {order['refund_status'] or 'Processing'}"
        )
    return f"**Order #{order['order_id']}** — Status: {status}"


def _exec_check_refund(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    tickets = get_tickets_by_order(params["order_id"])
    refund = order["refund_status"]

    if refund:
        policy = _get_policy("refund processing time how long")
        msg = _policy_message(f"Refund status: {refund}", policy, order, "Checking refund status")
        result = (
            f"**Refund Status — Order #{order['order_id']}**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Order total:** {order['total']}\n"
            f"**Refund:** {refund}\n\n"
            f"{msg}"
        )
    else:
        result = (
            f"**Order #{order['order_id']}** ({order['restaurant']})\n\n"
            f"No refund has been requested for this order. "
            f"If something was wrong, I can help you report it and start the refund process."
        )
    if tickets:
        result += "\n\n**Related tickets:** " + ", ".join(f"`{t['ticket_id']}`" for t in tickets)
    return result


def _exec_view_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    items_str = "\n".join(f"  - {item}" for item in order["items"])
    result = (
        f"**Order #{order['order_id']}**\n\n"
        f"**Restaurant:** {order['restaurant']}\n**Status:** {order['status']}\n"
        f"**Items:**\n{items_str}\n\n"
        f"**Total:** {order['total']}\n**Ordered:** {order['order_time']}\n"
        f"**Delivered:** {order['delivery_time'] or 'Pending'}\n"
        f"**Driver:** {order['driver']}\n**Payment:** {order['payment_method']}"
    )
    if order["special_instructions"]:
        result += f"\n**Notes:** {order['special_instructions']}"
    if order["refund_status"]:
        result += f"\n\n**Refund:** {order['refund_status']}"
    return result


def _exec_contact_driver(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    if "In Progress" not in order["status"]:
        return (
            f"Order #{order['order_id']} is **{order['status']}** — the driver is no longer on this delivery.\n\n"
            f"If you had a problem with the delivery, I can help you report it."
        )
    return (
        f"**Message sent to {order['driver']}!**\n\n"
        f"**Your message:** \"{params['message']}\"\n\n"
        f"{order['driver']} has been notified and typically responds within a few minutes. "
        f"Keep your phone nearby for updates."
    )


def _exec_lookup_ticket(params: dict, customer_id: str) -> str:
    ticket = get_ticket(params["ticket_id"])
    if not ticket:
        return f"Ticket `{params['ticket_id']}` not found. Please check the ID (format: UE-XXXXXX)."
    from datetime import datetime
    created = datetime.fromtimestamp(ticket["created_at"]).strftime("%Y-%m-%d %H:%M")
    result = (
        f"**Ticket {ticket['ticket_id']}**\n\n"
        f"**Type:** {ticket['ticket_type']}\n**Status:** {ticket['status']}\n"
        f"**Order:** #{ticket['order_id']}\n**Created:** {created}\n"
    )
    if ticket["details"]:
        for k, v in ticket["details"].items():
            result += f"**{k.replace('_', ' ').title()}:** {v}\n"
    if ticket["resolution"]:
        result += f"\n**Resolution:** {ticket['resolution']}"
    else:
        status_msgs = {
            "Investigating": "This ticket is being investigated. You'll receive an update within 2 hours.",
            "Under Review": "This ticket is under review. You'll receive an update within 24 hours.",
            "Escalated to Agent": "A support specialist is handling this. They'll reach out shortly.",
            "Auto-Resolved": "This issue has been resolved automatically.",
        }
        result += f"\n{status_msgs.get(ticket['status'], '')}"
    return result


# ══════════════════════════════════════════════════════════════════════
# ACTION TOOLS — create ticket, update order, follow real Uber policy
# ══════════════════════════════════════════════════════════════════════

def _exec_cancel_order(params: dict, customer_id: str) -> str:
    """
    Real Uber Eats cancellation behavior:
    - Free if restaurant hasn't accepted yet
    - Partial refund (total minus delivery fee) if accepted but not picked up
    - Full charge if driver has picked up
    """
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    confirm = params.get("confirm", "").lower().strip()
    if confirm not in ("yes", "y", "confirm", "sure", "ok"):
        return "No problem, your order has **not** been cancelled."

    # Determine cancellation outcome based on order status (simulates real Uber logic)
    status = order["status"]
    if "Delivered" in status:
        return f"Order #{order['order_id']} has already been delivered and cannot be cancelled. If there was an issue, I can help you report it."
    elif "Cancelled" in status:
        return f"Order #{order['order_id']} is already cancelled."

    # Execute cancellation
    update_order_status(params["order_id"], "Cancelled")

    if "In Progress" in status:
        # Restaurant has accepted — partial refund per Uber policy
        update_order_refund(params["order_id"], f"Partial refund — {order['total']} minus delivery fee (restaurant already preparing)")
        resolution = f"Cancelled after restaurant accepted. Partial refund of {order['total']} minus delivery fee."
    else:
        update_order_refund(params["order_id"], f"Full refund — {order['total']}")
        resolution = f"Cancelled before restaurant accepted. Full refund of {order['total']}."

    ticket = create_ticket(
        customer_id=customer_id, order_id=params["order_id"],
        ticket_type="Order Cancellation", details={"previous_status": status},
        status="Auto-Resolved", resolution=resolution,
    )

    policy = _get_policy("cancellation policy fee refund")
    msg = _policy_message(resolution, policy, order, "Customer requested cancellation")

    return (
        f"**Order #{order['order_id']} — Cancelled**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n\n"
        f"{msg}\n\n"
        f"Refunds typically appear on your statement within 3-5 business days."
    )


def _exec_report_missing(params: dict, customer_id: str) -> str:
    """
    Real Uber Eats missing items behavior:
    - Must report within 48 hours
    - Refund for missing items (sales price + tax)
    - Creates ticket for tracking
    - Directs to help form for additional details/photos
    """
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."

    missing = params["missing_items"]

    # Create ticket + initiate refund
    ticket = create_ticket(
        customer_id=customer_id, order_id=params["order_id"],
        ticket_type="Missing Items",
        details={"missing_items": missing, "order_items": ", ".join(order["items"])},
        status="Auto-Resolved",
        resolution=f"Refund initiated for missing items: {missing}",
    )
    update_order_status(params["order_id"], "Delivered — Reported Issue")
    update_order_refund(params["order_id"], f"Refund initiated for missing: {missing}")

    policy = _get_policy("missing items refund policy 48 hours")
    msg = _policy_message(
        f"Refund initiated for missing items: {missing}",
        policy, order, f"Missing items: {missing}"
    )

    return (
        f"**Missing Items Report — Order #{order['order_id']}**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Restaurant:** {order['restaurant']}\n"
        f"**Missing:** {missing}\n\n"
        f"{msg}\n\n"
        f"Per Uber Eats policy, you're eligible for a refund of the missing item(s) sales price including tax. "
        f"Refunds typically appear within 3-5 business days.\n\n"
        f"If you need to provide additional details, you can also report this through the Uber Eats app: "
        f"**Orders → Select this order → Help → Missing items**, or visit [{HELP_URL}]({HELP_URL}).\n\n"
        f"**Ticket reference:** `{ticket['ticket_id']}`"
    )


def _exec_report_wrong(params: dict, customer_id: str) -> str:
    """
    Real Uber Eats wrong items behavior:
    - Must report within 48 hours
    - Photos requested as evidence
    - Under review (not auto-resolved — needs verification)
    - Full refund or redelivery possible
    """
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."

    wrong = params["wrong_items"]
    expected = params["expected_items"]

    ticket = create_ticket(
        customer_id=customer_id, order_id=params["order_id"],
        ticket_type="Wrong Items",
        details={"wrong_items": wrong, "expected_items": expected},
        status="Under Review",
        resolution="Pending review — photo evidence requested",
    )
    update_order_status(params["order_id"], "Delivered — Reported Issue")
    update_order_refund(params["order_id"], "Under review — wrong items reported")

    policy = _get_policy("wrong order incorrect items refund")
    msg = _policy_message(
        f"Wrong items report submitted for review.",
        policy, order, f"Received: {wrong}. Expected: {expected}"
    )

    return (
        f"**Wrong Items Report — Order #{order['order_id']}**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Restaurant:** {order['restaurant']}\n"
        f"**Received:** {wrong}\n"
        f"**Expected:** {expected}\n\n"
        f"{msg}\n\n"
        f"**To help us resolve this faster**, please submit photos of the incorrect items through the Uber Eats app: "
        f"**Orders → Select this order → Help → Wrong items**, or visit [{HELP_URL}]({HELP_URL}).\n\n"
        f"Photos help us verify the issue and process your refund faster. "
        f"Please report within 48 hours for eligibility.\n\n"
        f"**Ticket reference:** `{ticket['ticket_id']}`"
    )


def _exec_report_delivery(params: dict, customer_id: str) -> str:
    """
    Real Uber Eats delivery issue behavior — varies by problem type:
    - Not delivered: GPS investigation, potential full refund
    - Damaged/spilled: auto-refund (full)
    - Late (20+ min past ETA): credit or 20% Uber Cash
    - Cold food: partial refund (50%)
    - Driver issue: escalation to safety team
    """
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."

    problem = params["problem_type"].lower()

    # Determine resolution based on problem type (real Uber behavior)
    if "not delivered" in problem or "never arrived" in problem:
        status = "Investigating"
        resolution = f"Investigation started — checking GPS and delivery confirmation for order {order['order_id']}"
        ticket_type = "Order Not Delivered"
        policy_query = "order never arrived not delivered refund investigation"
        follow_up = (
            f"We're checking GPS data and the driver's delivery confirmation. "
            f"If we confirm the order wasn't delivered, a **full refund of {order['total']}** will be issued.\n\n"
            f"You'll hear back within **2 hours**. If your order shows as delivered but you didn't receive it, "
            f"please also check your door if you selected 'Leave at door'."
        )

    elif "damaged" in problem or "spilled" in problem:
        status = "Auto-Resolved"
        resolution = f"Full refund of {order['total']} issued for damaged/spilled order"
        ticket_type = "Damaged Order"
        policy_query = "damaged spilled order refund"
        follow_up = (
            f"A **full refund of {order['total']}** has been issued to your original payment method. "
            f"Refunds typically appear within 3-5 business days.\n\n"
            f"We apologize for the inconvenience. If you'd like to submit photos for our records, "
            f"go to **Orders → Help → Order arrived damaged** in the app."
        )

    elif "late" in problem:
        status = "Auto-Resolved"
        resolution = "Uber Eats credit (20% of order) issued for late delivery"
        ticket_type = "Late Delivery"
        policy_query = "late delivery compensation refund credit"
        order_val = float(order['total'].replace('$', '').replace(',', ''))
        credit = f"${order_val * 0.20:.2f}"
        follow_up = (
            f"We've issued **{credit} in Uber Cash** (20% of your order) for the late delivery. "
            f"This credit is automatically applied to your account for future orders.\n\n"
            f"Per our policy, orders arriving 20+ minutes past the estimated delivery window are eligible for "
            f"compensation. You can also cancel late orders for a full refund through the app."
        )

    elif "cold" in problem:
        status = "Auto-Resolved"
        order_val = float(order['total'].replace('$', '').replace(',', ''))
        refund_amt = f"${order_val * 0.50:.2f}"
        resolution = f"50% refund ({refund_amt}) issued for food quality issue"
        ticket_type = "Food Quality"
        policy_query = "food quality cold order refund"
        follow_up = (
            f"A **50% refund ({refund_amt})** has been issued for the food quality issue. "
            f"Refunds appear within 3-5 business days.\n\n"
            f"If you'd like to request a full refund, you can submit additional details and photos through "
            f"the app: **Orders → Help → Food quality issue**, or visit [{HELP_URL}]({HELP_URL})."
        )

    elif "driver" in problem:
        status = "Escalated to Agent"
        resolution = "Escalated to safety team for driver behavior review"
        ticket_type = "Driver Issue"
        policy_query = "driver issue safety report"
        follow_up = (
            f"This has been **escalated to our safety team** for review. "
            f"A specialist will reach out within 24 hours.\n\n"
            f"If you feel unsafe, please contact local authorities. "
            f"You can also report safety concerns directly: **Account → Help → Safety issue**."
        )

    else:
        status = "Under Review"
        resolution = f"Delivery issue reported — under review: {params['problem_type']}"
        ticket_type = "Delivery Problem"
        policy_query = "delivery problem report help"
        follow_up = (
            f"We've logged your report and our team will review it within 24 hours.\n\n"
            f"For faster resolution, you can also report through the app: "
            f"**Orders → Help**, or visit [{HELP_URL}]({HELP_URL})."
        )

    # Create ticket + update order
    ticket = create_ticket(
        customer_id=customer_id, order_id=params["order_id"],
        ticket_type=ticket_type,
        details={"problem": params["problem_type"], "driver": order["driver"]},
        status=status, resolution=resolution,
    )
    update_order_status(params["order_id"], "Delivered — Reported Issue")
    update_order_refund(params["order_id"], resolution)

    # RAG policy message
    policy = _get_policy(policy_query)
    msg = _policy_message(resolution, policy, order, f"Delivery issue: {params['problem_type']}")

    return (
        f"**Delivery Issue Report — Order #{order['order_id']}**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Restaurant:** {order['restaurant']}\n"
        f"**Problem:** {params['problem_type'].title()}\n"
        f"**Status:** {status}\n\n"
        f"{msg}\n\n"
        f"{follow_up}\n\n"
        f"**Ticket reference:** `{ticket['ticket_id']}`"
    )


# ══════════════════════════════════════════════════════════════════════
# ESCALATION — connect to human support
# ══════════════════════════════════════════════════════════════════════

def _exec_contact_support(params: dict, customer_id: str) -> str:
    queue_time = random.randint(2, 15)
    ticket = create_ticket(
        customer_id=customer_id, order_id="N/A",
        ticket_type="Support Escalation",
        details={"issue": params["issue_summary"], "urgency": params["urgency"]},
        status="Escalated to Agent",
    )
    return (
        f"**Connected to Support**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Issue:** {params['issue_summary']}\n"
        f"**Priority:** {params['urgency'].capitalize()}\n"
        f"**Estimated wait:** {queue_time} minutes\n\n"
        f"A support specialist will reach out via in-app chat. "
        f"You can also reach Uber Eats support by phone at **1-800-253-6882** "
        f"or through the help center at [{HELP_URL}]({HELP_URL}).\n\n"
        f"**Ticket reference:** `{ticket['ticket_id']}`"
    )
