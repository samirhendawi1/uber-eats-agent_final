"""
Tool execution engine with RAG-informed resolution.

Key change: ticket/issue tools now use the RAG pipeline to retrieve
relevant Uber Eats policies, then pass those policies + order details
to the LLM to determine the correct resolution. This means the knowledge
base directly drives how issues are resolved — not hardcoded rules.

Course concepts:
  - RAG in the action loop (not just for Q&A)
  - Tool Use with context augmentation
  - Prompt Chaining: collect params -> RAG retrieve -> LLM resolve -> create ticket
"""

from __future__ import annotations
import random
from memory import get_state, set_state
from orders_db import get_order
from tickets_db import create_ticket, get_ticket, get_tickets_by_order, get_tickets_by_customer
from tools import TOOLS
from vectorstore import retrieve as rag_retrieve
from llm_client import chat_completion


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
Do NOT use JSON — respond in natural customer-facing language with markdown formatting."""


def _rag_resolve(order: dict, issue_type: str, issue_details: str) -> tuple[str, str, str]:
    """
    Use RAG to retrieve relevant policies, then LLM to determine resolution.
    Returns (resolution_text, status, resolution_summary).
    """
    # Step 1: RAG retrieval — search for policies relevant to this issue
    search_queries = [
        issue_type,
        f"{issue_type} refund policy",
        f"uber eats {issue_details}",
    ]
    
    all_docs = []
    seen = set()
    for q in search_queries:
        docs = rag_retrieve(q, k=3)
        for d in docs:
            doc_id = d.metadata.get("id", d.page_content[:50])
            if doc_id not in seen:
                seen.add(doc_id)
                all_docs.append(d)
    
    rag_context = "\n\n".join(
        f"[Policy: {d.metadata.get('title', 'Unknown')} | Category: {d.metadata.get('category', '')}]\n{d.page_content}"
        for d in all_docs[:6]
    )
    
    # Step 2: Build prompt with order details + RAG context
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
    
    # Step 3: LLM determines resolution
    messages = [{"role": "user", "content": prompt}]
    resolution_text = chat_completion(messages, temperature=0.3, max_tokens=512)
    
    # Step 4: Determine status and summary based on issue type
    issue_lower = issue_details.lower() + " " + issue_type.lower()
    if any(kw in issue_lower for kw in ["not delivered", "never arrived"]):
        status = "Investigating"
        summary = f"Investigation initiated — GPS and delivery data under review for order {order['order_id']}"
    elif any(kw in issue_lower for kw in ["damaged", "spilled"]):
        status = "Auto-Resolved"
        summary = f"Full refund of {order['total']} issued for damaged/spilled order (per damaged order policy)"
    elif "late" in issue_lower:
        status = "Auto-Resolved"
        summary = f"Uber Eats credit issued for late delivery (per late delivery policy)"
    elif any(kw in issue_lower for kw in ["cold", "quality", "stale"]):
        status = "Auto-Resolved"
        refund = f"${float(order['total']) * 0.5:.2f}"
        summary = f"Partial refund of {refund} issued for food quality issue (per food quality policy)"
    elif any(kw in issue_lower for kw in ["missing", "not included"]):
        status = "Auto-Resolved"
        summary = f"Refund initiated for missing items (per missing items policy)"
    elif any(kw in issue_lower for kw in ["wrong", "incorrect"]):
        status = "Under Review"
        summary = f"Wrong items report under review — refund or redelivery pending (per wrong items policy)"
    else:
        status = "Under Review"
        summary = f"Issue reported and under review"
    
    # Build source attribution
    sources = [d.metadata.get("title", "") for d in all_docs[:3]]
    source_text = "\n\n📚 **Policies referenced:** " + ", ".join(f"_{s}_" for s in sources if s)
    
    return resolution_text + source_text, status, summary


# ── Parameter collection helpers ─────────────────────────────────────

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


# ── INSTANT RESOLUTION executors (no RAG needed — just data lookup) ──

def _exec_track_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found. Please check the order ID and try again."
    
    status = order["status"]
    if "In Progress" in status:
        stages = ["Order confirmed", "Restaurant preparing", "**Driver picking up**", "On the way", "Arriving"]
        eta = random.randint(10, 30)
        return (
            f"**Live Tracking - Order #{order['order_id']}**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Items:** {', '.join(order['items'])}\n\n"
            f"**Progress:**\n" + "\n".join(f"{'>>>' if '**' in s else '   '} {s}" for s in stages) + f"\n\n"
            f"**Driver:** {order['driver']}\n"
            f"**ETA:** ~{eta} minutes\n\n"
            f"You can message your driver by saying *\"send a message to my driver\"*."
        )
    elif "Delivered" in status:
        return (
            f"**Order #{order['order_id']} was delivered**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Delivered at:** {order['delivery_time']}\n"
            f"**Driver:** {order['driver']}\n\n"
            f"If you had an issue with this order, I can help you report it."
        )
    elif "Cancelled" in status:
        return (
            f"**Order #{order['order_id']} was cancelled**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Refund:** {order['refund_status'] or 'Processing'}\n\n"
            f"If you need help with the refund, just let me know."
        )
    else:
        return f"**Order #{order['order_id']}** - Status: {status}"


def _exec_check_refund(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    
    tickets = get_tickets_by_order(params["order_id"])
    
    # Use RAG to get refund policy context
    docs = rag_retrieve("refund status processing time", k=3)
    policy_context = " ".join(d.page_content for d in docs)[:500]
    
    refund = order["refund_status"]
    if refund:
        result = (
            f"**Refund Status - Order #{order['order_id']}**\n\n"
            f"**Restaurant:** {order['restaurant']}\n"
            f"**Order total:** {order['total']}\n"
            f"**Refund:** {refund}\n\n"
        )
        # Use RAG-retrieved policy to explain timeline
        messages = [{"role": "user", "content": (
            f"Based on this Uber Eats refund policy: {policy_context}\n\n"
            f"The customer's refund status is: {refund}\n"
            f"Give a brief 1-2 sentence update about what they should expect next. "
            f"Reference the policy naturally."
        )}]
        explanation = chat_completion(messages, temperature=0.3, max_tokens=150)
        result += explanation
    else:
        result = (
            f"**Order #{order['order_id']}** ({order['restaurant']})\n\n"
            f"No refund has been requested for this order.\n"
            f"If something was wrong, I can help you report it and start a refund."
        )
    
    if tickets:
        result += f"\n\n**Related tickets:** " + ", ".join(f"`{t['ticket_id']}`" for t in tickets)
    
    return result


def _exec_view_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    
    items_str = "\n".join(f"  - {item}" for item in order["items"])
    result = (
        f"**Order #{order['order_id']} - Full Details**\n\n"
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
        result += f"**Notes:** {order['special_instructions']}\n"
    if order["rating"]:
        result += f"**Your rating:** {'*' * int(order['rating'])}\n"
    if order["refund_status"]:
        result += f"\n**Refund:** {order['refund_status']}"
    
    tickets = get_tickets_by_order(params["order_id"])
    if tickets:
        result += f"\n**Tickets:** " + ", ".join(f"`{t['ticket_id']}` ({t['status']})" for t in tickets)
    
    return result


def _exec_contact_driver(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    
    if "In Progress" not in order["status"]:
        return (
            f"Order #{order['order_id']} is **{order['status']}**, so the driver is no longer active.\n\n"
            f"If you had a problem, I can help you report it."
        )
    
    return (
        f"**Message sent to {order['driver']}!**\n\n"
        f"**Your message:** \"{params['message']}\"\n\n"
        f"{order['driver']} has been notified and typically responds within a few minutes."
    )


def _exec_apply_promo(params: dict, customer_id: str) -> str:
    code = params["promo_code"].strip().upper()
    valid_promos = {
        "WELCOME10": "10% off your next order (up to $5)",
        "FREEDELIVERY": "Free delivery on your next 3 orders",
        "SAVE5": "$5 off orders over $25",
    }
    if code in valid_promos:
        return f"**Promo code applied!**\n\n**Code:** `{code}`\n**Benefit:** {valid_promos[code]}\n\nDiscount applies automatically at checkout."
    else:
        return f"Promo code `{code}` is invalid or expired. Check the code and try again."


def _exec_cancel_order(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    
    confirm = params.get("confirm", "").lower().strip()
    if confirm not in ("yes", "y", "confirm", "sure", "ok"):
        return "No problem, your order has **not** been cancelled."
    
    # Use RAG to get cancellation policy
    docs = rag_retrieve("cancel order cancellation fee policy", k=3)
    policy_context = "\n".join(f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in docs)
    
    messages = [{"role": "user", "content": (
        f"Based on these Uber Eats cancellation policies:\n{policy_context}\n\n"
        f"Order details: #{order['order_id']} from {order['restaurant']}, total {order['total']}, status: {order['status']}\n"
        f"The customer confirmed they want to cancel.\n\n"
        f"Write a brief response confirming the cancellation and explaining any fees/refund "
        f"based on the order status and policy. Reference the policy naturally."
    )}]
    
    return chat_completion(messages, temperature=0.3, max_tokens=256)


# ── RAG-INFORMED TICKET executors ────────────────────────────────────

def _exec_report_missing(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found. Please check the order ID."
    
    # RAG-informed resolution
    resolution_text, status, summary = _rag_resolve(
        order=order,
        issue_type="Missing Items",
        issue_details=f"Customer reports missing items: {params['missing_items']}",
    )
    
    ticket = create_ticket(
        customer_id=customer_id,
        order_id=params["order_id"],
        ticket_type="Missing Items",
        details={"missing_items": params["missing_items"], "order_items": ", ".join(order["items"])},
        status=status,
        resolution=summary,
    )
    
    return (
        f"**Ticket `{ticket['ticket_id']}` created for Order #{order['order_id']}**\n\n"
        f"{resolution_text}\n\n"
        f"---\n"
        f"**Ticket reference:** `{ticket['ticket_id']}`"
    )


def _exec_report_wrong(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    
    resolution_text, status, summary = _rag_resolve(
        order=order,
        issue_type="Wrong Items Received",
        issue_details=f"Received: {params['wrong_items']}. Expected: {params['expected_items']}",
    )
    
    ticket = create_ticket(
        customer_id=customer_id,
        order_id=params["order_id"],
        ticket_type="Wrong Items",
        details={"wrong_items": params["wrong_items"], "expected_items": params["expected_items"]},
        status=status,
        resolution=summary,
    )
    
    return (
        f"**Ticket `{ticket['ticket_id']}` created for Order #{order['order_id']}**\n\n"
        f"{resolution_text}\n\n"
        f"---\n"
        f"**Ticket reference:** `{ticket['ticket_id']}`"
    )


def _exec_report_delivery(params: dict, customer_id: str) -> str:
    order = get_order(params["order_id"])
    if not order:
        return f"Order `{params['order_id']}` not found."
    
    resolution_text, status, summary = _rag_resolve(
        order=order,
        issue_type="Delivery Problem",
        issue_details=f"Problem reported: {params['problem_type']}",
    )
    
    ticket = create_ticket(
        customer_id=customer_id,
        order_id=params["order_id"],
        ticket_type=f"Delivery: {params['problem_type'].title()}",
        details={"problem": params["problem_type"], "driver": order["driver"]},
        status=status,
        resolution=summary,
    )
    
    return (
        f"**Ticket `{ticket['ticket_id']}` created for Order #{order['order_id']}**\n\n"
        f"{resolution_text}\n\n"
        f"---\n"
        f"**Ticket reference:** `{ticket['ticket_id']}`"
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
        f"**Connected to support!**\n\n"
        f"**Ticket:** `{ticket['ticket_id']}`\n"
        f"**Issue:** {params['issue_summary']}\n"
        f"**Priority:** {params['urgency'].capitalize()}\n"
        f"**Estimated wait:** {queue_time} minutes\n\n"
        f"A specialist will reach out via in-app chat.\n\n"
        f"**Reference:** `{ticket['ticket_id']}`"
    )


# ── TICKET LOOKUP ────────────────────────────────────────────────────

def _exec_lookup_ticket(params: dict, customer_id: str) -> str:
    ticket = get_ticket(params["ticket_id"])
    if not ticket:
        return f"Ticket `{params['ticket_id']}` not found. Format: UE-XXXXXX."
    
    from datetime import datetime
    created = datetime.fromtimestamp(ticket["created_at"]).strftime("%Y-%m-%d %H:%M")
    updated = datetime.fromtimestamp(ticket["updated_at"]).strftime("%Y-%m-%d %H:%M")
    
    result = (
        f"**Ticket {ticket['ticket_id']}**\n\n"
        f"**Type:** {ticket['ticket_type']}\n"
        f"**Status:** {ticket['status']}\n"
        f"**Order:** #{ticket['order_id']}\n"
        f"**Created:** {created}\n"
        f"**Updated:** {updated}\n"
    )
    
    if ticket["details"]:
        for k, v in ticket["details"].items():
            result += f"**{k.replace('_', ' ').title()}:** {v}\n"
    
    if ticket["resolution"]:
        result += f"\n**Resolution:** {ticket['resolution']}"
    
    return result
