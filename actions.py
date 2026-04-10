"""Tool execution engine. Handles instant resolutions, ticket creation, and escalations."""

from __future__ import annotations
import random, time
from memory import get_state, set_state
from orders_db import get_order
from tickets_db import create_ticket, get_ticket, get_tickets_by_order, get_tickets_by_customer
from tools import TOOLS


# ── Parameter collection helpers ─────────────────────────────────────

def get_next_missing_param(session_id: str, tool_name: str) -> str | None:
    collected = get_state(session_id, f"tool_{tool_name}_params", {})
    for p in TOOLS[tool_name]["required_params"]:
        if p not in collected:
            return p
    return None


def get_param_prompt(session_id: str, tool_name: str) -> str:
    """Get the prompt for the next missing parameter, with order hints."""
    param = get_next_missing_param(session_id, tool_name)
    if not param:
        return ""
    prompt = TOOLS[tool_name]["param_prompts"][param]
    # Add order ID hints for order-related params
    if param == "order_id":
        prompt += "\n\n💡 *You can find your order ID in the **📦 Orders** tab in the sidebar, or just tell me about your issue and I'll look it up.*"
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
    """Execute a tool and return the response text."""
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


# ── INSTANT RESOLUTION executors ─────────────────────────────────────

def _exec_track_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"⚠️ Order `{params['order_id']}` not found. Please check the order ID and try again."
    
    status = order["status"]
    
    if "In Progress" in status:
        stages = ["🔵 Order confirmed", "🔵 Restaurant preparing", "🟢 **Driver picking up**", "⚪ On the way", "⚪ Arriving"]
        eta = random.randint(10, 30)
        driver = order["driver"]
        return (
            f"📍 **Live Tracking — Order #{order['order_id']}**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Items:** {', '.join(order['items'])}\n\n"
            f"**Delivery Progress:**\n" + "\n".join(stages) + f"\n\n"
            f"**Driver:** {driver}\n"
            f"**Estimated arrival:** ~{eta} minutes\n\n"
            f"💡 You can contact your driver by saying *\"send a message to my driver\"*."
        )
    elif "Delivered" in status:
        return (
            f"✅ **Order #{order['order_id']} was delivered**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Delivered at:** {order['delivery_time']}\n"
            f"**Driver:** {order['driver']}\n\n"
            f"If you had an issue with this order, I can help you report it."
        )
    elif "Cancelled" in status:
        return (
            f"❌ **Order #{order['order_id']} was cancelled**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Refund:** {order['refund_status'] or 'Processing'}\n\n"
            f"If you need help with the refund, just let me know."
        )
    else:
        return f"📋 **Order #{order['order_id']}** — Status: {status}"


def _exec_check_refund(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"⚠️ Order `{params['order_id']}` not found."
    
    # Also check for tickets on this order
    tickets = get_tickets_by_order(params["order_id"])
    
    refund = order["refund_status"]
    if refund:
        result = (
            f"💰 **Refund Status — Order #{order['order_id']}**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Order total:** {order['total']}\n"
            f"**Refund:** {refund}\n\n"
        )
        if "Approved" in refund or "issued" in refund.lower():
            result += "✅ Your refund has been processed. It should appear on your statement within 3-5 business days."
        elif "Pending" in refund or "review" in refund.lower():
            result += "⏳ Your refund is being reviewed. You'll receive an update within 24-48 hours."
        else:
            result += "Your refund is being processed."
    else:
        result = (
            f"📋 **Order #{order['order_id']}** ({order['restaurant']})\n\n"
            f"No refund has been requested for this order.\n"
        )
        if "Delivered" in order["status"]:
            result += "If something was wrong with your order, I can help you report it and start a refund."
    
    if tickets:
        result += f"\n\n📎 **Related tickets:** " + ", ".join(f"`{t['ticket_id']}`" for t in tickets)
    
    return result


def _exec_view_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"⚠️ Order `{params['order_id']}` not found."
    
    items_str = "\n".join(f"  • {item}" for item in order["items"])
    result = (
        f"📋 **Order #{order['order_id']} — Full Details**\n\n"
        f"**Restaurant:** {order['restaurant']}\n"
        f"**Status:** {order['status']}\n"
        f"**Items:**\n{items_str}\n\n"
        f"**Total:** {order['total']}\n"
        f"**Ordered:** {order['order_time']}\n"
        f"**Delivered:** {order['delivery_time'] or 'Pending'}\n"
        f"**Driver:** {order['driver']}\n"
        f"**Payment:** {order['payment_method']}\n"
        f"**Address:** {order['address']}\n"
    )
    if order["special_instructions"]:
        result += f"**Special instructions:** {order['special_instructions']}\n"
    if order["rating"]:
        result += f"**Your rating:** {'⭐' * int(order['rating'])}\n"
    if order["refund_status"]:
        result += f"\n💰 **Refund:** {order['refund_status']}"
    
    tickets = get_tickets_by_order(params["order_id"])
    if tickets:
        result += f"\n📎 **Tickets:** " + ", ".join(f"`{t['ticket_id']}` ({t['status']})" for t in tickets)
    
    return result


def _exec_contact_driver(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"⚠️ Order `{params['order_id']}` not found."
    
    if "In Progress" not in order["status"]:
        return (
            f"📋 Order #{order['order_id']} is **{order['status']}**, so the driver is no longer on this delivery.\n\n"
            f"If you had a problem with the delivery, I can help you report it."
        )
    
    driver = order["driver"]
    return (
        f"📱 **Message sent to {driver}!**\n\n"
        f"**Your message:** \"{params['message']}\"\n\n"
        f"Your driver {driver} has been notified. They typically respond within a few minutes. "
        f"You can also reach them directly through the Uber Eats app's call button on the tracking screen."
    )


def _exec_apply_promo(params: dict, customer_id: str) -> str:
    code = params["promo_code"].strip().upper()
    # Simulate promo validation
    valid_promos = {
        "WELCOME10": ("10% off your next order (up to $5)", True),
        "FREEDELIVERY": ("Free delivery on your next 3 orders", True),
        "SAVE5": ("$5 off orders over $25", True),
    }
    if code in valid_promos:
        desc, _ = valid_promos[code]
        return f"🎉 **Promo code applied!**\n\n**Code:** `{code}`\n**Benefit:** {desc}\n\nThe discount will be applied automatically at checkout."
    else:
        return f"❌ Promo code `{code}` is invalid or expired. Please check the code and try again, or ask me about current promotions."


def _exec_cancel_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"⚠️ Order `{params['order_id']}` not found."
    
    confirm = params.get("confirm", "").lower().strip()
    if confirm not in ("yes", "y", "confirm"):
        return "No problem, your order has **not** been cancelled. Is there anything else I can help with?"
    
    if "In Progress" in order["status"]:
        return (
            f"✅ **Order #{order['order_id']} cancellation requested!**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Total:** {order['total']}\n\n"
            f"Since the restaurant may have already started preparing your food, a partial cancellation fee may apply. "
            f"You'll receive a confirmation email with refund details within a few minutes."
        )
    elif "Delivered" in order["status"]:
        return f"This order has already been delivered and cannot be cancelled. If there was an issue, I can help you report it."
    elif "Cancelled" in order["status"]:
        return f"This order has already been cancelled."
    else:
        return f"Order #{order['order_id']} status: **{order['status']}**. Please contact support for help."


# ── TICKET executors ─────────────────────────────────────────────────

def _exec_report_missing(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"⚠️ Order `{params['order_id']}` not found. Please check the order ID."
    
    # Auto-resolve: if the missing items match known items, issue immediate refund
    missing = params["missing_items"]
    order_items = ", ".join(order["items"])
    
    ticket = create_ticket(
        customer_id=customer_id,
        order_id=params["order_id"],
        ticket_type="Missing Items",
        details={"missing_items": missing, "order_items": order_items},
        status="Auto-Resolved",
        resolution=f"Automatic refund initiated for missing items: {missing}",
    )
    
    return (
        f"✅ **Missing items — resolved automatically!**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Order:** #{order['order_id']} from {order['restaurant']}\n"
        f"**Missing items:** {missing}\n\n"
        f"💰 **A refund has been automatically initiated** for the missing items. "
        f"You should see the credit on your statement within 3-5 business days.\n\n"
        f"📎 You can reference this ticket as `{ticket['ticket_id']}` if you need to follow up."
    )


def _exec_report_wrong(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"⚠️ Order `{params['order_id']}` not found."
    
    ticket = create_ticket(
        customer_id=customer_id,
        order_id=params["order_id"],
        ticket_type="Wrong Items",
        details={
            "wrong_items": params["wrong_items"],
            "expected_items": params["expected_items"],
        },
        status="Under Review",
        resolution=None,
    )
    
    return (
        f"📋 **Wrong items report submitted**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Order:** #{order['order_id']} from {order['restaurant']}\n"
        f"**Received:** {params['wrong_items']}\n"
        f"**Expected:** {params['expected_items']}\n\n"
        f"Our team will review this and you'll receive an update within 24 hours. "
        f"In most cases, a full refund or redelivery is issued.\n\n"
        f"📎 Reference: `{ticket['ticket_id']}`"
    )


def _exec_report_delivery(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"⚠️ Order `{params['order_id']}` not found."
    
    problem = params["problem_type"].lower()
    
    # Certain problems can be auto-resolved
    if "not delivered" in problem or "never arrived" in problem:
        ticket = create_ticket(
            customer_id=customer_id,
            order_id=params["order_id"],
            ticket_type="Order Not Delivered",
            details={"problem": params["problem_type"], "driver": order["driver"]},
            status="Investigating",
            resolution=None,
        )
        return (
            f"🔍 **Delivery investigation started**\n\n"
            f"**Ticket:** `{ticket['ticket_id']}`\n"
            f"**Order:** #{order['order_id']} from {order['restaurant']}\n"
            f"**Driver:** {order['driver']}\n"
            f"**Problem:** Order not delivered\n\n"
            f"We're checking GPS data and the driver's delivery confirmation. "
            f"If we confirm the order wasn't delivered, a **full refund of {order['total']}** will be issued automatically.\n"
            f"You'll hear back within **2 hours**.\n\n"
            f"📎 Reference: `{ticket['ticket_id']}`"
        )
    elif "damaged" in problem or "spilled" in problem:
        ticket = create_ticket(
            customer_id=customer_id,
            order_id=params["order_id"],
            ticket_type="Damaged Order",
            details={"problem": params["problem_type"]},
            status="Auto-Resolved",
            resolution=f"Automatic full refund of {order['total']} for damaged order",
        )
        return (
            f"✅ **Damaged order — resolved automatically!**\n\n"
            f"**Ticket:** `{ticket['ticket_id']}`\n"
            f"**Order:** #{order['order_id']} from {order['restaurant']}\n\n"
            f"💰 **A full refund of {order['total']} has been issued** to your original payment method. "
            f"We're sorry about the inconvenience.\n\n"
            f"📎 Reference: `{ticket['ticket_id']}`"
        )
    elif "late" in problem:
        ticket = create_ticket(
            customer_id=customer_id,
            order_id=params["order_id"],
            ticket_type="Late Delivery",
            details={"problem": params["problem_type"], "order_time": order["order_time"], "delivery_time": order["delivery_time"]},
            status="Auto-Resolved",
            resolution="$5.00 Uber Eats credit issued for late delivery",
        )
        return (
            f"✅ **Late delivery — resolved!**\n\n"
            f"**Ticket:** `{ticket['ticket_id']}`\n"
            f"**Order:** #{order['order_id']} from {order['restaurant']}\n"
            f"**Ordered:** {order['order_time']}\n"
            f"**Delivered:** {order['delivery_time'] or 'Unknown'}\n\n"
            f"🎁 **A $5.00 Uber Eats credit has been added to your account** as an apology for the delay.\n\n"
            f"📎 Reference: `{ticket['ticket_id']}`"
        )
    elif "cold" in problem:
        ticket = create_ticket(
            customer_id=customer_id,
            order_id=params["order_id"],
            ticket_type="Food Quality",
            details={"problem": params["problem_type"]},
            status="Auto-Resolved",
            resolution=f"Partial refund (50%) issued for cold food: ${float(order['total']) * 0.5:.2f}",
        )
        refund_amount = float(order["total"]) * 0.5
        return (
            f"✅ **Food quality issue — resolved!**\n\n"
            f"**Ticket:** `{ticket['ticket_id']}`\n"
            f"**Order:** #{order['order_id']} from {order['restaurant']}\n\n"
            f"💰 **A 50% refund (${refund_amount:.2f}) has been issued** for the food quality issue.\n\n"
            f"📎 Reference: `{ticket['ticket_id']}`"
        )
    else:
        ticket = create_ticket(
            customer_id=customer_id,
            order_id=params["order_id"],
            ticket_type="Delivery Problem",
            details={"problem": params["problem_type"]},
            status="Under Review",
        )
        return (
            f"📋 **Delivery problem reported**\n\n"
            f"**Ticket:** `{ticket['ticket_id']}`\n"
            f"**Order:** #{order['order_id']} from {order['restaurant']}\n"
            f"**Issue:** {params['problem_type']}\n\n"
            f"Our team will review this within 24 hours.\n\n"
            f"📎 Reference: `{ticket['ticket_id']}`"
        )


# ── ESCALATION executor ──────────────────────────────────────────────

def _exec_contact_support(params: dict, customer_id: str) -> str:
    queue_time = random.randint(2, 15)
    ticket = create_ticket(
        customer_id=customer_id,
        order_id="N/A",
        ticket_type="Support Escalation",
        details={"issue": params["issue_summary"], "urgency": params["urgency"]},
        status="Escalated to Agent",
    )
    return (
        f"📞 **Connected to support!**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Issue:** {params['issue_summary']}\n"
        f"**Priority:** {params['urgency'].capitalize()}\n"
        f"**Estimated wait:** {queue_time} minutes\n\n"
        f"A support specialist will reach out via in-app chat. Please keep the app open.\n\n"
        f"📎 Reference: `{ticket['ticket_id']}`"
    )


# ── TICKET LOOKUP ────────────────────────────────────────────────────

def _exec_lookup_ticket(params: dict, customer_id: str) -> str:
    ticket = get_ticket(params["ticket_id"])
    if not ticket:
        return f"⚠️ Ticket `{params['ticket_id']}` not found. Please check the ticket ID (format: UE-XXXXXX)."
    
    from datetime import datetime
    created = datetime.fromtimestamp(ticket["created_at"]).strftime("%Y-%m-%d %H:%M")
    updated = datetime.fromtimestamp(ticket["updated_at"]).strftime("%Y-%m-%d %H:%M")
    
    result = (
        f"📋 **Ticket {ticket['ticket_id']}**\n\n"
        f"**Type:** {ticket['ticket_type']}\n"
        f"**Status:** {ticket['status']}\n"
        f"**Order:** #{ticket['order_id']}\n"
        f"**Created:** {created}\n"
        f"**Last updated:** {updated}\n"
    )
    
    if ticket["details"]:
        for k, v in ticket["details"].items():
            result += f"**{k.replace('_', ' ').title()}:** {v}\n"
    
    if ticket["resolution"]:
        result += f"\n✅ **Resolution:** {ticket['resolution']}"
    elif ticket["status"] == "Investigating":
        result += "\n⏳ This ticket is being investigated. You'll receive an update soon."
    elif ticket["status"] == "Under Review":
        result += "\n⏳ This ticket is under review. You'll receive an update within 24 hours."
    elif ticket["status"] == "Escalated to Agent":
        result += "\n📞 A support specialist is handling this. They'll reach out shortly."
    
    return result
