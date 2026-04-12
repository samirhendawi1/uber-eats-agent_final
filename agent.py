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
- ALWAYS use retrieved context to inform your decisions
- For order-specific requests (track, refund, report), use the appropriate tool
- "Where is my order?" → track_order (NOT a delivery problem report)
- "What's your refund policy?" → answer from knowledge (NOT check_refund_status tool)
- "My food arrived cold" → report_delivery_problem tool
- When citing knowledge, reference the source article title
- Be empathetic, concise, and helpful
- If a tool requires parameters you don't have, indicate what you have and what's missing
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
        reasoning = decision.get("reasoning", "")
        intent = f"tool:{tool_name}"
        response = _handle_tool_call(session_id, tool_name, extracted, user_message, customer_id)

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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Context:\n{rag_context}\n\n"
                f"User: {user_message}\n\n"
                f"Your previous response was flagged: {eval_result['issue']}\n"
                f"Please provide an improved response. Respond naturally (not JSON)."
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
