# 🍔 Uber Eats Customer Support Agent

**RSM8430H — Group 17 Final Project**

A conversational support agent powered by a RAG-first agentic architecture where the LLM uses retrieved knowledge to reason about and select tools.

## Architecture — RAG-First Agentic Pattern

```
User Message
     │
     ▼
┌─────────────────┐
│ Input Guardrails │──→ Block injection + unsafe content
└────────┬────────┘
         ▼
┌─────────────────┐
│  RAG Retrieval   │──→ ALWAYS retrieve top-5 relevant docs
│  (ChromaDB)      │    from 60-article knowledge base
└────────┬────────┘
         ▼
┌──────────────────────────────────────────┐
│         Augmented LLM (ReAct)            │
│  Receives: RAG context + tool list +     │
│            conversation history           │
│  Decides:  answer | tool_call | clarify  │
└────────┬─────────────┬───────────────────┘
         │             │
    ┌────▼────┐   ┌───▼──────┐
    │ Answer  │   │ Tool     │
    │ from    │   │ Execution│──→ Instant / Ticket / Escalation
    │ context │   └───┬──────┘
    └────┬────┘       │
         │            │
         ▼            ▼
┌─────────────────────────┐
│   Output Guardrails     │──→ Validate response
└────────┬────────────────┘
         ▼
┌─────────────────────────┐
│ Evaluator (LLM-as-Judge)│──→ Score 1-5, retry if < 3
└────────┬────────────────┘
         ▼
┌─────────────────┐
│  Memory (SQLite) │──→ Save to conversation history
└─────────────────┘
```

## Course Concepts Implemented (RSM8430)


| Concept                         | Implementation                                            | File                         |
| ------------------------------- | --------------------------------------------------------- | ---------------------------- |
| **RAG Pipeline**                | Ingestion → chunking → ChromaDB embedding → MMR retrieval | `vectorstore.py`             |
| **Augmented LLM**               | LLM with RAG + Tools + Memory (the 3 augmentations)       | `agent.py`                   |
| **ReAct Pattern**               | Retrieve → Reason → Act decision loop                     | `agent.py`                   |
| **Tool Use / Function Calling** | 11 tools the LLM selects via structured JSON output       | `tools.py`, `actions.py`     |
| **Agentic Workflows**           | Multi-turn parameter collection, prompt chaining          | `agent.py`                   |
| **Evaluator-Optimizer**         | LLM-as-Judge scores responses, retries if score < 3       | `agent.py`                   |
| **Guardrails**                  | Input (injection/safety) + Output (validation) layers     | `guardrails.py`              |
| **Memory**                      | Short-term (conversation) + Long-term (SQLite sessions)   | `memory.py`                  |
| **Evaluation**                  | Rule-based checks + LLM-as-Judge scoring                  | `evaluation.py`              |
| **Source Attribution**          | RAG retrieval includes source titles + URLs               | `agent.py`, `vectorstore.py` |


## Key Design: RAG Drives Tool Selection

Unlike typical chatbots with hardcoded intent classification, our agent uses RAG to inform all decisions:

1. **Every message** retrieves relevant knowledge base documents
2. The LLM sees the retrieved context alongside tool descriptions
3. The LLM **reasons** about whether to answer from context or call a tool
4. This means the agent naturally handles edge cases where a question is partially about policy AND partially about a specific order

**Example:** "I read you offer refunds for missing items — my order 100001 is missing fries"
→ RAG retrieves the refund policy → LLM sees the policy AND the report_missing_items tool → correctly calls the tool with context from the policy

## Tool Categories


| Category          | Tools                                                                                                               | Resolution                  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **Instant**       | track_order, check_refund_status, view_order_details, contact_driver, apply_promo_code, cancel_order, lookup_ticket | Immediate response          |
| **Auto-resolve**  | report_missing_items, report_delivery_problem (damaged/late/cold)                                                   | Automatic refund + ticket   |
| **Investigation** | report_delivery_problem (not delivered), report_wrong_items                                                         | Ticket with 2-24hr response |
| **Escalation**    | contact_support                                                                                                     | Human agent queue           |


## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install chromadb sentence-transformers streamlit requests pydantic
streamlit run app.py
```

## Evaluation

```bash
# Rule-based evaluation (15 test cases)
python evaluation.py

# With LLM-as-Judge scoring (slower, uses LLM calls)
python evaluation.py --judge
```

## Files


| File             | Purpose                                             |
| ---------------- | --------------------------------------------------- |
| `agent.py`       | RAG-first agentic orchestrator (core)               |
| `vectorstore.py` | ChromaDB vector store (no LangChain)                |
| `tools.py`       | 11 tool definitions (instant/ticket/escalation)     |
| `actions.py`     | Tool execution with auto-resolve logic              |
| `tickets_db.py`  | SQLite ticket persistence + lookup                  |
| `guardrails.py`  | Input + output guardrails                           |
| `memory.py`      | SQLite conversation memory                          |
| `llm_client.py`  | qwen3-30b endpoint wrapper                          |
| `orders_db.py`   | 60 sample orders (CSV)                              |
| `users_db.py`    | 6 demo accounts                                     |
| `evaluation.py`  | Rule-based + LLM-as-Judge eval suite                |
| `app.py`         | Streamlit frontend (chat, orders, tickets, account) |


## Team

Group 17 — Samir Hendawi, Yuyan Zhang, Jiaer Jiang, Ce Shen, Junyan Yue