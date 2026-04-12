"""
RAG-First Agentic Orchestrator
================================
Architecture based on course concepts (RSM8430):
  - Augmented LLM: RAG + Tools + Memory (Lecture: Agentic Building Blocks)
  - ReAct-style reasoning: Retrieve → Reason → Act (Lecture: Agents)
  - Evaluator-Optimizer: Response quality check loop (Lecture: Agentic Workflows)
  - Guardrails: Input/output validation (Lecture: Guardrails)
  - Prompt Chaining: Multi-step tool execution (Lecture: Prompt Chaining)

Flow:
  1. Input Guardrails (reject unsafe/injection)
  2. RAG Retrieval (always — provides context for ALL decisions)
  3. Augmented LLM decides: answer from context OR call a tool
  4. If tool → execute tool → feed result back to LLM for final response
  5. Output Guardrails (validate response)
  6. Evaluator check (quality gate)
  7. Save to memory
"""

from __future__ import annotations
import json, re
from guardrails import run_guardrails, check_output_guardrails
from memory import add_message, get_history, get_state, set_state
from tools import TOOLS
from actions import execute_tool, get_next_missing_param, get_param_prompt, store_param, clear_tool
from vectorstore import retrieve
from llm_client import chat_completion, chat_completion_json


# ── Tool descriptions for the LLM (< 200 tokens each, per course best practices) ──

def _build_tool_descriptions() -> str:
    """Build concise tool descriptions for the system prompt."""
    lines = []
    for name, tool in TOOLS.items():
        params = ", ".join(tool["required_params"])
        lines.append(f"- {name}: {tool['description']} [params: {params}] [category: {tool['category']}]")
    return "\n".join(lines)


TOOL_DESCRIPTIONS = _build_tool_descriptions()


# ── System prompt: Augmented LLM with RAG + Tools + Memory ──

SYSTEM_PROMPT = f"""You are an Uber Eats customer support agent. You are an Augmented LLM with access to:
1. **Knowledge Base (RAG)**: Retrieved documents about Uber Eats policies and features
2. **Tools**: Actions you can execute on behalf of the customer
3. **Memory**: Conversation history for context continuity

## Available Tools
{TOOL_DESCRIPTIONS}

## Decision Framework (ReAct Pattern)
For each user message, you receive RAG-retrieved context. Use this framework:

1. **Retrieve**: Context documents are provided automatically
2. **Reason**: Analyze the user's intent using context + conversation history  
3. **Act**: Either:
   a) ANSWER directly using retrieved knowledge (for policy/feature questions)
   b) CALL a tool (for actions on specific orders, accounts, tickets)
   c) ASK for clarification (if ambiguous)

## Response Format
Respond with JSON only:

For direct answers (from RAG knowledge):
{{"action": "answer", "response": "your helpful response here", "sources_used": ["article title 1", "article title 2"]}}

For tool calls:
{{"action": "tool_call", "tool": "tool_name", "extracted_params": {{"param1": "value1"}}, "reasoning": "why this tool"}}

For clarification:
{{"action": "clarify", "response": "your clarifying question"}}

For greetings:
{{"action": "greeting"}}

For out-of-scope:
{{"action": "out_of_scope", "response": "polite redirect to Uber Eats topics"}}

## Important Rules
- If the user's request doesn't have a tool call in availible tools, create a ticket for the user.
- ALWAYS use retrieved context to inform your decisions
- **CRITICAL: If the user mentions an order ID/number AND describes a problem, ALWAYS use a tool. Never just "answer" when an action is needed.**
- If the user mentions an order number + missing items → tool_call report_missing_items (extract both order_id and missing_items)
- If the user mentions an order number + damage/spill/cold → tool_call report_delivery_problem
- If the user mentions an order number + wrong items → tool_call report_wrong_items
- If the user says "yes", "confirm", "proceed", "go ahead" → look at the conversation history to understand what they're confirming, then call the appropriate tool
- "Where is my order?" → track_order (NOT a delivery problem report)
- "What's your refund policy?" → answer from knowledge (NOT check_refund_status tool)
- "My food arrived cold" (no order ID) → report_delivery_problem tool (you'll collect the order_id next)
- When citing knowledge, reference the source article title
- Be empathetic, concise, and helpful
- If a tool requires parameters you don't have, extract what you can and the system will prompt for the rest
- NEVER ask the user to confirm before starting a tool — just start collecting parameters and execute
"""


# ── Evaluator: LLM-as-Judge quality check (course concept) ──

EVALUATOR_PROMPT = """You are a quality evaluator for an Uber Eats support agent.
Rate this response on a 1-5 scale and flag any issues.

Criteria:
- Accuracy: Does it match the retrieved context? No hallucinations?
- Helpfulness: Does it address the user's actual need?
- Safety: No harmful, off-topic, or misleading content?
- Tone: Empathetic and professional?

User message: {user_msg}
Agent response: {response}

Respond with JSON only: {{"score": 1-5, "pass": true/false, "issue": "brief issue description or null"}}
A score >= 3 passes. Below 3, the response should be regenerated."""


def _evaluate_response(user_msg: str, response: str) -> dict:
    """LLM-as-Judge evaluator (course: Evaluator-Optimizer pattern)."""
    try:
        result = chat_completion_json(
            [{"role": "user", "content": EVALUATOR_PROMPT.format(user_msg=user_msg, response=response)}],
            temperature=0.1,
        )
        return {"score": result.get("score", 4), "pass": result.get("pass", True), "issue": result.get("issue")}
    except Exception:
        return {"score": 4, "pass": True, "issue": None}  # fail-open


# ── RAG-Augmented LLM Decision ──

def _rag_decision(user_msg: str, history: list[dict], rag_context: str) -> dict:
    """Send user message + RAG context + history to LLM for decision."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (short-term memory)
    for h in history[-8:]:
        messages.append(h)

    # User message with RAG context
    augmented_msg = (
        f"## Retrieved Knowledge Base Context\n{rag_context}\n\n"
        f"## User Message\n{user_msg}\n\n"
        f"Respond with JSON following the decision framework."
    )
    messages.append({"role": "user", "content": augmented_msg})

    return chat_completion_json(messages, temperature=0.2)


# ── Format greeting ──

def _greeting_response() -> str:
    return (
        "👋 Hi there! I'm your Uber Eats support assistant powered by our knowledge base. I can help with:\n\n"
        "**💡 Ask me anything:**\n"
        "• Questions about Uber Eats policies, fees, features\n"
        "• Track your order or check delivery status\n"
        "• Check refund status or report issues\n"
        "• Contact your driver or cancel an order\n"
        "• Report missing items, wrong orders, or delivery problems\n"
        "• Look up existing support tickets\n"
        "• Apply promo codes\n\n"
        "I use our knowledge base to give you accurate answers and can take action on your account. How can I help?"
    )


# ── Handle tool flow ──

def _handle_tool_call(session_id: str, tool_name: str, extracted_params: dict, user_msg: str, customer_id: str) -> str:
    """Initiate or continue a tool call with parameter collection."""
    if tool_name not in TOOLS:
        return "I'm sorry, that action isn't available. Could you rephrase what you need?"

    set_state(session_id, "current_tool", tool_name)

    # Store any params the LLM already extracted
    for param, value in extracted_params.items():
        if value and param in TOOLS[tool_name]["required_params"]:
            store_param(session_id, tool_name, param, str(value))

    # Also try regex extraction from user message
    _try_extract_params(session_id, tool_name, user_msg)

    # Check if we need more params
    next_param = get_next_missing_param(session_id, tool_name)
    if next_param:
        return get_param_prompt(session_id, tool_name)

    # All params ready → execute
    result = execute_tool(session_id, tool_name, customer_id)
    return result


def _try_extract_params(session_id: str, tool_name: str, user_msg: str):
    """Regex-based param extraction as fallback."""
    msg_lower = user_msg.lower().strip()
    tool = TOOLS[tool_name]

    # Order ID
    order_match = re.search(r"\b(\d{5,7})\b", user_msg)
    if order_match and "order_id" in tool["required_params"]:
        if get_next_missing_param(session_id, tool_name) == "order_id":
            store_param(session_id, tool_name, "order_id", order_match.group(1))

    # Ticket ID
    ticket_match = re.search(r"(UE-\d{4,8})", user_msg, re.IGNORECASE)
    if ticket_match and "ticket_id" in tool["required_params"]:
        store_param(session_id, tool_name, "ticket_id", ticket_match.group(1).upper())

    # Problem type
    if "problem_type" in tool["required_params"]:
        for keyword, ptype in [
            ("not delivered", "not delivered"), ("never arrived", "not delivered"),
            ("didn't arrive", "not delivered"), ("late", "late"),
            ("damaged", "damaged"), ("spilled", "spilled"),
            ("cold", "cold food"), ("driver", "driver issue"),
        ]:
            if keyword in msg_lower:
                store_param(session_id, tool_name, "problem_type", ptype)
                break

    # Urgency
    if "urgency" in tool["required_params"]:
        for level in ["high", "medium", "low"]:
            if level in msg_lower:
                store_param(session_id, tool_name, "urgency", level)
                break

    # Confirm
    if "confirm" in tool["required_params"]:
        for word in ["yes", "y", "confirm", "sure", "ok"]:
            if word in msg_lower.split():
                store_param(session_id, tool_name, "confirm", "yes"); break
        for word in ["no", "n", "cancel", "nevermind"]:
            if word in msg_lower.split():
                store_param(session_id, tool_name, "confirm", "no"); break

    # Promo code
    if "promo_code" in tool["required_params"]:
        promo_match = re.search(r"\b([A-Z0-9]{4,20})\b", user_msg)
        if promo_match:
            store_param(session_id, tool_name, "promo_code", promo_match.group(1))


def _pre_route_tool(user_msg: str, history: list[dict]) -> tuple | None:
    """
    Deterministic pre-routing for obvious tool cases.
    Returns (tool_name, extracted_params) or None.
    Bypasses the LLM decision when the intent is unambiguous.
    """
    msg_lower = user_msg.lower()

    # ── Skip pre-routing for QUESTIONS (let RAG handle these) ──
    question_indicators = [
        "what is", "what's", "whats", "what are", "how do", "how does", "how can", "how long",
        "when do", "when does", "when can", "why do", "why does", "why is",
        "can i", "can you", "do you", "is there", "are there",
        "policy", "policies", "explain", "tell me about", "how to", "what happens",
        "do i get", "am i eligible", "how much", "what if", "does uber",
        "i want to know", "i want the", "i need to know", "information about",
        "?",  # Any question mark = likely a question, not an action
    ]
    if any(q in msg_lower for q in question_indicators):
        return None  # Let the LLM + RAG handle knowledge questions

    # Extract order ID if present
    order_match = re.search(r"\b(\d{5,7})\b", user_msg)
    order_id = order_match.group(1) if order_match else None

    # Extract ticket ID if present
    ticket_match = re.search(r"(UE-\d{4,8})", user_msg, re.IGNORECASE)

    # ── Order ID + issue keywords → direct tool call ──
    if order_id:
        # Missing items
        if any(kw in msg_lower for kw in ["missing", "forgot", "not included", "didn't get", "left out", "without"]):
            # Try to extract what's missing (everything after the keyword)
            missing = re.split(r"missing|forgot|not included|didn't get|left out|without", msg_lower, maxsplit=1)
            missing_text = missing[-1].strip().strip(".,!") if len(missing) > 1 else ""
            params = {"order_id": order_id}
            if missing_text and len(missing_text) > 1:
                params["missing_items"] = missing_text
            return ("report_missing_items", params)

        # Wrong items
        if any(kw in msg_lower for kw in ["wrong item", "incorrect item", "received wrong", "wrong order", "not what i ordered"]):
            return ("report_wrong_items", {"order_id": order_id})

        # Delivery problems
        if any(kw in msg_lower for kw in ["not delivered", "never arrived", "didn't arrive", "never received"]):
            return ("report_delivery_problem", {"order_id": order_id, "problem_type": "not delivered"})
        if any(kw in msg_lower for kw in ["damaged", "spilled", "crushed", "broken"]):
            return ("report_delivery_problem", {"order_id": order_id, "problem_type": "damaged"})
        if any(kw in msg_lower for kw in ["cold food", "arrived cold", "food was cold", "not hot"]):
            return ("report_delivery_problem", {"order_id": order_id, "problem_type": "cold food"})
        if any(kw in msg_lower for kw in ["late", "took too long", "slow delivery"]):
            return ("report_delivery_problem", {"order_id": order_id, "problem_type": "late"})

        # Tracking
        if any(kw in msg_lower for kw in ["where is", "track", "status", "eta", "how long"]):
            return ("track_order", {"order_id": order_id})

        # Refund check
        if any(kw in msg_lower for kw in ["refund", "money back", "reimburs"]):
            return ("check_refund_status", {"order_id": order_id})

        # Cancel
        if any(kw in msg_lower for kw in ["cancel"]):
            return ("cancel_order", {"order_id": order_id})

        # View details
        if any(kw in msg_lower for kw in ["details", "receipt", "show me order", "view order", "order info"]):
            return ("view_order_details", {"order_id": order_id})

    # ── Ticket lookup ──
    if ticket_match:
        return ("lookup_ticket", {"ticket_id": ticket_match.group(1).upper()})

    # ── No order ID but clear ACTION request (requires possessive language) ──
    if any(kw in msg_lower for kw in ["my order has missing", "my items are missing", "i have missing items", "i got missing items", "my food is missing"]):
        return ("report_missing_items", {})
    if any(kw in msg_lower for kw in ["my order never arrived", "never got my order", "my food never came", "i never received my order"]):
        return ("report_delivery_problem", {"problem_type": "not delivered"})
    if any(kw in msg_lower for kw in ["my food arrived damaged", "my order was damaged", "my food was spilled", "my order arrived broken"]):
        return ("report_delivery_problem", {"problem_type": "damaged"})

    # ── Confirmation from history ──
    confirms = {"yes", "y", "ok", "sure", "confirm", "proceed", "go ahead", "yeah", "yep", "do it", "please", "yes please", "yes please assist me"}
    if msg_lower.strip().rstrip(".!,") in confirms and len(history) >= 2:
        # Look at what the assistant last suggested
        last_bot = [h["content"] for h in history if h["role"] == "assistant"]
        if last_bot:
            last = last_bot[-1].lower()
            # Re-extract order ID and issue from the last assistant message
            prev_order = re.search(r"\b(\d{5,7})\b", last_bot[-1])
            if prev_order:
                oid = prev_order.group(1)
                if any(kw in last for kw in ["missing", "report"]):
                    return ("report_missing_items", {"order_id": oid})
                if any(kw in last for kw in ["deliver", "arrived", "damage"]):
                    return ("report_delivery_problem", {"order_id": oid})
                if any(kw in last for kw in ["refund"]):
                    return ("check_refund_status", {"order_id": oid})
                if any(kw in last for kw in ["cancel"]):
                    return ("cancel_order", {"order_id": oid})
                if any(kw in last for kw in ["escalat", "support", "specialist"]):
                    return ("contact_support", {})

    return None


def _handle_followup(session_id: str, user_msg: str, customer_id: str) -> str:
    """Continue multi-turn tool parameter collection."""
    current_tool = get_state(session_id, "current_tool")
    if not current_tool:
        return "I'm not sure what you're referring to. How can I help you?"

    _try_extract_params(session_id, current_tool, user_msg)

    next_param = get_next_missing_param(session_id, current_tool)
    if next_param:
        # If extraction didn't work, store raw message as param value
        still_missing = get_next_missing_param(session_id, current_tool)
        if still_missing == next_param:
            store_param(session_id, current_tool, next_param, user_msg.strip())

    next_param = get_next_missing_param(session_id, current_tool)
    if next_param:
        return get_param_prompt(session_id, current_tool)

    return execute_tool(session_id, current_tool, customer_id)


# ── Main entry point ─────────────────────────────────────────────────

def process_message(session_id: str, user_message: str, customer_id: str = "") -> dict:
    """
    Main agent loop implementing the RAG-first agentic pattern:
    Input Guardrails → RAG Retrieval → LLM Decision → Tool/Answer → Output Guardrails → Evaluator
    """
    # ── Step 1: Save to memory (short-term) ──
    add_message(session_id, "user", user_message)
    history = get_history(session_id)

    # ── Step 2: Input Guardrails ──
    rejection = run_guardrails(user_message)
    if rejection:
        add_message(session_id, "assistant", rejection)
        return {"response": rejection, "intent": "guardrail_blocked", "confidence": 1.0, "sources": [], "eval_score": None}

    # ── Step 3: Check if mid-tool (multi-turn parameter collection) ──
    current_tool = get_state(session_id, "current_tool")
    if current_tool:
        response = _handle_followup(session_id, user_message, customer_id)
        add_message(session_id, "assistant", response)
        return {"response": response, "intent": f"tool:{current_tool}", "confidence": 1.0, "sources": [], "eval_score": None}

    # ── Step 3b: Short confirmation with no active tool — check history ──
    # If user says "yes"/"ok"/"sure" and the last assistant message suggested a tool action,
    # re-run the LLM with explicit history so it picks up context
    short_confirms = {"yes", "y", "ok", "sure", "confirm", "proceed", "go ahead", "yeah", "yep", "do it", "please"}
    if user_message.strip().lower().rstrip(".!") in short_confirms and len(history) >= 2:
        # Inject a hint into the user message so the LLM has context
        last_assistant = [h for h in history if h["role"] == "assistant"]
        if last_assistant:
            user_message_augmented = f"The user said '{user_message}' to confirm. Previous assistant message was: {last_assistant[-1]['content'][:300]}. Determine what action to take based on this context."
        else:
            user_message_augmented = user_message
    else:
        user_message_augmented = user_message

    # ── Step 3c: Pre-routing — deterministic tool matching for obvious cases ──
    # If message clearly matches a tool pattern, skip the LLM decision entirely.
    # This prevents the LLM from choosing "answer" when it should call a tool.
    pre_route = _pre_route_tool(user_message, history)
    if pre_route:
        tool_name, extracted_params = pre_route
        # Still do RAG retrieval so the tool execution has policy context
        docs = retrieve(user_message, k=5)
        sources = [
            {"title": getattr(d, "metadata", {}).get("title", ""), "url": getattr(d, "metadata", {}).get("source_url", ""), "category": getattr(d, "metadata", {}).get("category", "")}
            for d in docs
        ]
        intent = f"tool:{tool_name}"
        try:
            response = _handle_tool_call(session_id, tool_name, extracted_params, user_message, customer_id)
        except Exception as e:
            response = f"I encountered an error: {str(e)[:100]}. Please try again."

        add_message(session_id, "assistant", response)
        return {"response": response, "intent": intent, "confidence": 1.0, "sources": sources, "eval_score": None}

    # ── Step 4: RAG Retrieval (ALWAYS — this is the core) ──
    docs = retrieve(user_message, k=5)
    rag_context = "\n\n".join(
        f"[Source: {getattr(d, 'metadata', {}).get('title', 'Unknown')} | Category: {getattr(d, 'metadata', {}).get('category', '')} | URL: {getattr(d, 'metadata', {}).get('source_url', '')}]\n{getattr(d, 'page_content', str(d))}"
        for d in docs
    )
    sources = [
        {"title": getattr(d, "metadata", {}).get("title", ""), "url": getattr(d, "metadata", {}).get("source_url", ""), "category": getattr(d, "metadata", {}).get("category", "")}
        for d in docs
    ]

    # ── Step 5: Augmented LLM Decision (ReAct: Reason + Act) ──
    decision = _rag_decision(user_message_augmented, history, rag_context)

    action = decision.get("action", "answer")
    intent = action
    response = ""

    if action == "greeting":
        response = _greeting_response()
        intent = "greeting"
        sources = []

    elif action == "out_of_scope":
        response = decision.get("response", "I can only help with Uber Eats topics. How can I assist you?")
        intent = "out_of_scope"
        sources = []

    elif action == "clarify":
        response = decision.get("response", "Could you tell me more about what you need help with?")
        intent = "clarification"

    elif action == "tool_call":
        tool_name = decision.get("tool", "")
        extracted = decision.get("extracted_params", {})
        reasoning = decision.get("reasoning", "")
        intent = f"tool:{tool_name}"
        try:
            response = _handle_tool_call(session_id, tool_name, extracted, user_message, customer_id)
        except Exception as e:
            response = f"I encountered an error processing your request. Could you try again? (Error: {str(e)[:100]})"

    elif action == "answer":
        response = decision.get("response", "")
        # Ensure sources_used from LLM maps to actual source docs
        used_titles = decision.get("sources_used", [])
        if used_titles:
            sources = [s for s in sources if s["title"] in used_titles] or sources[:3]
        intent = "knowledge_answer"

    else:
        # Fallback: treat as answer
        response = decision.get("response", str(decision))
        intent = "fallback"

    if not response:
        response = "I'm sorry, I wasn't able to process that. Could you rephrase your question?"

    # Sanitize: if response looks like raw JSON from the LLM, re-generate naturally
    if response.strip().startswith('{"action"') or response.strip().startswith("{'action"):
        try:
            response = chat_completion(
                [{"role": "user", "content": f"Based on this context, respond naturally to the customer (NOT JSON):\n\nUser: {user_message}\nDecision: {response[:500]}\n\nProvide a helpful, natural language response."}],
                temperature=0.3, max_tokens=512,
            )
        except Exception:
            response = "I'm processing your request. Could you please try again?"

    # ── Step 6: Output Guardrails ──
    output_issue = check_output_guardrails(response)
    if output_issue:
        response = output_issue

    # ── Step 7: Evaluator-Optimizer (LLM-as-Judge) ──
    eval_result = _evaluate_response(user_message, response)
    eval_score = eval_result.get("score", 4)

    # If evaluator fails the response, try once more with explicit instruction
    if not eval_result.get("pass", True) and eval_result.get("issue"):
        retry_msgs = [
            {"role": "system", "content": "You are a helpful Uber Eats customer support agent. Respond naturally in plain language. Do NOT respond with JSON."},
            {"role": "user", "content": (
                f"Context:\n{rag_context}\n\n"
                f"User: {user_message}\n\n"
                f"Your previous response was flagged: {eval_result['issue']}\n"
                f"Please provide an improved, natural language response."
            )},
        ]
        response = chat_completion(retry_msgs, temperature=0.3)
        eval_score = 3  # mark as retried

    # ── Step 8: Save to memory ──
    add_message(session_id, "assistant", response)

    return {
        "response": response,
        "intent": intent,
        "confidence": 0.9,
        "sources": sources,
        "eval_score": eval_score,
    }