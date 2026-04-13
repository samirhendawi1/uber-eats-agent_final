"""
RAG-First Agentic Orchestrator
================================
The LLM is the sole decision maker. Every message flows through:
  1. Input Guardrails → 2. Mid-tool check → 3. RAG Retrieval →
  4. LLM decides (answer or tool) → 5. Tool executes if selected →
  6. Output Guardrails → 7. Evaluator → 8. Memory

No pre-router. No regex routing. The LLM sees RAG context + tool
descriptions + conversation history and chooses what to do.
Tools execute deterministically once selected.
"""

from __future__ import annotations
import json, re
from guardrails import run_guardrails, check_output_guardrails
from memory import add_message, get_history, get_state, set_state
from tools import TOOLS
from actions import execute_tool, get_next_missing_param, get_param_prompt, store_param, clear_tool
from vectorstore import retrieve
from llm_client import chat_completion, chat_completion_json


# ── Tool descriptions for the LLM ───────────────────────────────────

def _build_tool_descriptions() -> str:
    lines = []
    for name, tool in TOOLS.items():
        params = ", ".join(tool["required_params"])
        lines.append(f"- {name}: {tool['description']} [params: {params}] [category: {tool['category']}]")
    return "\n".join(lines)


TOOL_DESCRIPTIONS = _build_tool_descriptions()


# ── System prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an Uber Eats customer support agent. You are an Augmented LLM with access to:
1. **Knowledge Base (RAG)**: Retrieved documents about Uber Eats policies
2. **Tools**: Actions you can execute on behalf of the customer
3. **Memory**: Conversation history for context continuity

## Available Tools
{TOOL_DESCRIPTIONS}

## Decision Framework (ReAct Pattern)
For each user message, you receive RAG-retrieved context. Follow this:

1. **Retrieve**: Context documents are provided automatically
2. **Reason**: Analyze the user's intent using context + conversation history
3. **Act**: Choose ONE action:
   - ANSWER: for policy questions, general info, "how does X work?"
   - TOOL_CALL: for actions on orders, tickets, accounts
   - CLARIFY: if truly ambiguous
   - GREETING: for hello/hi/hey
   - OUT_OF_SCOPE: for non-Uber-Eats topics

## When to use a tool vs answer
- User mentions an ORDER ID + a problem → TOOL (even if phrased as a question)
- User mentions a TICKET ID (UE-XXXXXX) → lookup_ticket tool ALWAYS
- "Where is my order 100004?" → tool_call track_order
- "My order 100006 had missing fries" → tool_call report_missing_items
- "Check refund on order 100002" → tool_call check_refund_status
- "What's the refund policy?" (no order ID) → answer from RAG
- "How do I cancel?" (no order ID) → answer from RAG
- "Cancel order 100004" → tool_call cancel_order
- "I need to talk to someone" → tool_call contact_support
- "What's the info on ticket UE-123456?" → tool_call lookup_ticket
- User says "yes"/"ok"/"sure" after you suggested an action → look at conversation history, call the appropriate tool

## Response Format (JSON only)

For answers: {{"action": "answer", "response": "your response", "sources_used": ["article title"]}}
For tools: {{"action": "tool_call", "tool": "tool_name", "extracted_params": {{"param": "value"}}, "reasoning": "why"}}
For clarification: {{"action": "clarify", "response": "your question"}}
For greetings: {{"action": "greeting"}}
For out-of-scope: {{"action": "out_of_scope", "response": "polite redirect"}}

## Critical Rules
- If user provides an order ID AND describes an issue → ALWAYS use a tool. Never just "answer".
- If a ticket ID (UE-XXXXXX) is mentioned → ALWAYS use lookup_ticket. No exceptions.
- Extract as many parameters as you can from the message. The system will ask for any missing ones.
- Be empathetic, concise, and reference policy when answering from RAG.
- NEVER ask the user to confirm before starting — just call the tool.
"""


# ── Evaluator: LLM-as-Judge ──────────────────────────────────────────

EVALUATOR_PROMPT = """You are a quality evaluator for an Uber Eats support agent.
Rate this response on a 1-5 scale.

Criteria:
- Accuracy: Does it match the retrieved context? No hallucinations?
- Helpfulness: Does it address the user's actual need?
- Safety: No harmful or misleading content?
- Tone: Empathetic and professional?

User message: {user_msg}
Agent response: {response}

Respond with JSON only: {{"score": 1-5, "pass": true/false, "issue": "brief issue or null"}}
Score >= 3 passes. Below 3 should be regenerated."""


def _evaluate_response(user_msg: str, response: str) -> dict:
    try:
        result = chat_completion_json(
            [{"role": "user", "content": EVALUATOR_PROMPT.format(user_msg=user_msg, response=response)}],
            temperature=0.1,
        )
        return {"score": result.get("score", 4), "pass": result.get("pass", True), "issue": result.get("issue")}
    except Exception:
        return {"score": 4, "pass": True, "issue": None}


# ── RAG-Augmented LLM Decision ───────────────────────────────────────

def _rag_decision(user_msg: str, history: list[dict], rag_context: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-8:]:
        messages.append(h)
    augmented_msg = (
        f"## Retrieved Knowledge Base Context\n{rag_context}\n\n"
        f"## User Message\n{user_msg}\n\n"
        f"Respond with JSON following the decision framework."
    )
    messages.append({"role": "user", "content": augmented_msg})
    return chat_completion_json(messages, temperature=0.2)


# ── Greeting ─────────────────────────────────────────────────────────

def _greeting_response() -> str:
    return (
        "Hi there! I'm your Uber Eats support assistant. I can help with:\n\n"
        "- Questions about policies, fees, and features\n"
        "- Track orders or check delivery status\n"
        "- Check refund status or report issues\n"
        "- Report missing items, wrong orders, or delivery problems\n"
        "- Cancel orders or contact your driver\n"
        "- Look up existing support tickets\n"
        "- Escalate to a human support specialist\n\n"
        "How can I help you today?"
    )


# ── Tool call handling ───────────────────────────────────────────────

def _handle_tool_call(session_id: str, tool_name: str, extracted_params: dict, user_msg: str, customer_id: str) -> str:
    """Initiate or continue a tool call with parameter collection."""
    if tool_name not in TOOLS:
        return "I'm sorry, that action isn't available. Could you rephrase what you need?"

    set_state(session_id, "current_tool", tool_name)

    # Store any params the LLM extracted
    for param, value in extracted_params.items():
        if param in TOOLS[tool_name]["required_params"] and value:
            store_param(session_id, tool_name, param, str(value))

    # Also try regex extraction from raw message
    _try_extract_params(session_id, tool_name, user_msg)

    # Check if all params ready
    next_param = get_next_missing_param(session_id, tool_name)
    if next_param:
        return get_param_prompt(session_id, tool_name)

    # All params ready → execute
    return execute_tool(session_id, tool_name, customer_id)


def _try_extract_params(session_id: str, tool_name: str, user_msg: str):
    """Best-effort regex extraction of common parameter types."""
    # Order ID
    order_match = re.search(r"\b(\d{5,7})\b", user_msg)
    if order_match and "order_id" in TOOLS[tool_name]["required_params"]:
        next_p = get_next_missing_param(session_id, tool_name)
        if next_p == "order_id":
            store_param(session_id, tool_name, "order_id", order_match.group(1))

    # Ticket ID
    ticket_match = re.search(r"(UE-\d{4,8})", user_msg, re.IGNORECASE)
    if ticket_match and "ticket_id" in TOOLS[tool_name]["required_params"]:
        store_param(session_id, tool_name, "ticket_id", ticket_match.group(1).upper())

    # Problem type keywords
    if "problem_type" in TOOLS[tool_name]["required_params"]:
        msg_lower = user_msg.lower()
        for keyword, problem in [
            ("not delivered", "not delivered"), ("never arrived", "not delivered"),
            ("damaged", "damaged"), ("spilled", "damaged"),
            ("late", "late"), ("cold", "cold food"),
        ]:
            if keyword in msg_lower:
                store_param(session_id, tool_name, "problem_type", problem)
                break

    # Missing items
    if "missing_items" in TOOLS[tool_name]["required_params"]:
        missing_match = re.split(r"missing|forgot|without|left out|didn't get", user_msg.lower(), maxsplit=1)
        if len(missing_match) > 1:
            items = missing_match[-1].strip().strip(".,!?")
            if items and len(items) > 1:
                store_param(session_id, tool_name, "missing_items", items)


# ── Multi-turn follow-up ─────────────────────────────────────────────

def _handle_followup(session_id: str, user_msg: str, customer_id: str) -> str:
    """Continue multi-turn tool parameter collection."""
    current_tool = get_state(session_id, "current_tool")
    if not current_tool:
        return "I'm not sure what you're referring to. How can I help you?"

    _try_extract_params(session_id, current_tool, user_msg)

    next_param = get_next_missing_param(session_id, current_tool)
    if next_param:
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
    Main agent loop:
    Guardrails → Mid-tool → RAG → LLM Decision → Tool/Answer → Output Guardrails → Evaluator → Memory
    """
    # ── Step 1: Save to memory ──
    add_message(session_id, "user", user_message)
    history = get_history(session_id)

    # ── Step 2: Input Guardrails ──
    rejection = run_guardrails(user_message)
    if rejection:
        add_message(session_id, "assistant", rejection)
        return {"response": rejection, "intent": "guardrail_blocked", "confidence": 1.0, "sources": [], "eval_score": None}

    # ── Step 3: Mid-tool check (multi-turn parameter collection) ──
    current_tool = get_state(session_id, "current_tool")
    if current_tool:
        response = _handle_followup(session_id, user_message, customer_id)
        add_message(session_id, "assistant", response)
        return {"response": response, "intent": f"tool:{current_tool}", "confidence": 1.0, "sources": [], "eval_score": None}

    # ── Step 4: RAG Retrieval (ALWAYS runs) ──
    docs = retrieve(user_message, k=5)
    rag_context = "\n\n".join(
        f"[Source: {getattr(d, 'metadata', {}).get('title', 'Unknown')} | Category: {getattr(d, 'metadata', {}).get('category', '')} | URL: {getattr(d, 'metadata', {}).get('source_url', '')}]\n{getattr(d, 'page_content', str(d))}"
        for d in docs
    )
    sources = [
        {"title": getattr(d, "metadata", {}).get("title", ""), "url": getattr(d, "metadata", {}).get("source_url", ""), "category": getattr(d, "metadata", {}).get("category", "")}
        for d in docs
    ]

    # ── Step 5: LLM Decision (sole decision maker) ──
    decision = _rag_decision(user_message, history, rag_context)

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
        intent = f"tool:{tool_name}"
        try:
            response = _handle_tool_call(session_id, tool_name, extracted, user_message, customer_id)
        except Exception as e:
            response = f"I encountered an error processing your request. Could you try again? (Error: {str(e)[:100]})"

    elif action == "answer":
        response = decision.get("response", "")
        used_titles = decision.get("sources_used", [])
        if used_titles:
            sources = [s for s in sources if s["title"] in used_titles] or sources[:3]
        intent = "knowledge_answer"

    else:
        response = decision.get("response", str(decision))
        intent = "fallback"

    if not response:
        response = "I'm sorry, I wasn't able to process that. Could you rephrase your question?"

    # Sanitize raw JSON responses
    if response.strip().startswith('{"action"') or response.strip().startswith("{'action"):
        try:
            response = chat_completion(
                [{"role": "user", "content": f"Respond naturally to the customer (NOT JSON):\n\nUser: {user_message}\nContext: {response[:500]}\n\nProvide a helpful response."}],
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

    if not eval_result.get("pass", True) and eval_result.get("issue"):
        retry_msgs = [
            {"role": "system", "content": "You are a helpful Uber Eats support agent. Respond naturally. Do NOT use JSON."},
            {"role": "user", "content": (
                f"Context:\n{rag_context}\n\n"
                f"User: {user_message}\n\n"
                f"Previous response was flagged: {eval_result['issue']}\n"
                f"Provide an improved natural language response."
            )},
        ]
        response = chat_completion(retry_msgs, temperature=0.3)
        eval_score = 3

    # ── Step 8: Save to memory ──
    add_message(session_id, "assistant", response)

    return {
        "response": response,
        "intent": intent,
        "confidence": 0.9,
        "sources": sources,
        "eval_score": eval_score,
    }
