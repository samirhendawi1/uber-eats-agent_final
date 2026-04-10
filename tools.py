"""Formal tool definitions for the Uber Eats support agent.

Each tool has a name, description, required parameters, and an execute function.
Tools are categorized into:
  - INSTANT tools: resolve immediately (no ticket needed)
  - TICKET tools: create a support ticket for follow-up
  - ESCALATION tools: connect to a human agent
"""

from __future__ import annotations

TOOLS = {
    # ── INSTANT RESOLUTION TOOLS ──────────────────────────────────
    "track_order": {
        "name": "track_order",
        "description": "Track the real-time status of an active or recent order. Shows current delivery stage, driver info, and ETA.",
        "category": "instant",
        "required_params": ["order_id"],
        "param_prompts": {
            "order_id": "What's the order ID you'd like to track?",
        },
    },
    "check_refund_status": {
        "name": "check_refund_status",
        "description": "Check the status of a refund for a specific order. Shows refund amount, status, and timeline.",
        "category": "instant",
        "required_params": ["order_id"],
        "param_prompts": {
            "order_id": "What's the order ID you'd like to check the refund for?",
        },
    },
    "view_order_details": {
        "name": "view_order_details",
        "description": "View full details of a past order including items, restaurant, total, delivery time, and driver.",
        "category": "instant",
        "required_params": ["order_id"],
        "param_prompts": {
            "order_id": "What's the order ID you'd like to view?",
        },
    },
    "contact_driver": {
        "name": "contact_driver",
        "description": "Get driver contact info or send a message to the driver for an active delivery.",
        "category": "instant",
        "required_params": ["order_id", "message"],
        "param_prompts": {
            "order_id": "What's the order ID for the delivery you want to contact the driver about?",
            "message": "What message would you like to send to your driver? (e.g., 'I'm at the side entrance', 'Please ring the doorbell')",
        },
    },
    "apply_promo_code": {
        "name": "apply_promo_code",
        "description": "Apply a promotional code to the user's account.",
        "category": "instant",
        "required_params": ["promo_code"],
        "param_prompts": {
            "promo_code": "What promo code would you like to apply?",
        },
    },
    "cancel_order": {
        "name": "cancel_order",
        "description": "Cancel an active order. Full refund if restaurant hasn't started preparing.",
        "category": "instant",
        "required_params": ["order_id", "confirm"],
        "param_prompts": {
            "order_id": "What's the order ID you'd like to cancel?",
            "confirm": "Are you sure you want to cancel this order? Note: if the restaurant has already started preparing, cancellation fees may apply. (yes/no)",
        },
    },

    # ── TICKET TOOLS (require investigation) ──────────────────────
    "report_missing_items": {
        "name": "report_missing_items",
        "description": "Report items missing from a delivered order. Creates a ticket and processes automatic refund for confirmed missing items.",
        "category": "ticket",
        "required_params": ["order_id", "missing_items"],
        "param_prompts": {
            "order_id": "What's the order ID with missing items?",
            "missing_items": "Which items were missing from your order?",
        },
    },
    "report_wrong_items": {
        "name": "report_wrong_items",
        "description": "Report receiving wrong items in a delivered order. Creates a ticket for review.",
        "category": "ticket",
        "required_params": ["order_id", "wrong_items", "expected_items"],
        "param_prompts": {
            "order_id": "What's the order ID with wrong items?",
            "wrong_items": "What items did you receive that were wrong?",
            "expected_items": "What items did you originally order instead?",
        },
    },
    "report_delivery_problem": {
        "name": "report_delivery_problem",
        "description": "Report a delivery issue like food arrived damaged, spilled, cold, or order not delivered. Creates a ticket.",
        "category": "ticket",
        "required_params": ["order_id", "problem_type"],
        "param_prompts": {
            "order_id": "What's the order ID with the delivery problem?",
            "problem_type": "What type of delivery problem? (not delivered / late / damaged / spilled / cold food / driver issue)",
        },
    },

    # ── ESCALATION TOOL ───────────────────────────────────────────
    "contact_support": {
        "name": "contact_support",
        "description": "Escalate to a human support specialist. Only use when the issue cannot be resolved automatically.",
        "category": "escalation",
        "required_params": ["issue_summary", "urgency"],
        "param_prompts": {
            "issue_summary": "Could you briefly describe the issue you'd like to escalate?",
            "urgency": "How urgent is this? (low / medium / high)",
        },
    },
    "lookup_ticket": {
        "name": "lookup_ticket",
        "description": "Look up the status of an existing support ticket by ticket ID.",
        "category": "instant",
        "required_params": ["ticket_id"],
        "param_prompts": {
            "ticket_id": "What's your ticket ID? (e.g., UE-123456)",
        },
    },
}

# Mapping of tool names by category for easy filtering
INSTANT_TOOLS = [t for t in TOOLS.values() if t["category"] == "instant"]
TICKET_TOOLS = [t for t in TOOLS.values() if t["category"] == "ticket"]
ESCALATION_TOOLS = [t for t in TOOLS.values() if t["category"] == "escalation"]
