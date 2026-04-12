# Uber Eats Customer Support Agent

**RSM8430H: Applications of Large Language Models — Group 17 Final Project**

A production-quality conversational support agent for Uber Eats built on a RAG-first agentic architecture. The agent uses retrieved knowledge base documents to both answer policy questions and make tool-use decisions, combining retrieval-augmented generation with multi-step reasoning, tool execution, and safety guardrails.

**Live Demo:** [https://uber-eats.streamlit.app/](https://uber-eats.streamlit.app/)

**Team:** Samir Hendawi · Yuyan Zhang · Jiaer Jiang · Ce Shen · Junyan Yue

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Knowledge Base & Data Scraping](#knowledge-base--data-scraping)
- [RAG Pipeline & Chunking Strategy](#rag-pipeline--chunking-strategy)
- [Agent Decision Loop](#agent-decision-loop)
- [Tools & Actions](#tools--actions)
- [Guardrails](#guardrails)
- [Evaluation](#evaluation)
- [Memory & State Persistence](#memory--state-persistence)
- [User Accounts & Order Database](#user-accounts--order-database)
- [Frontend (Streamlit UI)](#frontend-streamlit-ui)
- [File Reference](#file-reference)
- [Setup — Running Locally](#setup--running-locally)
- [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
- [Demo Accounts](#demo-accounts)
- [Course Concepts Implemented](#course-concepts-implemented)
- [Known Limitations](#known-limitations)

---

## Overview

This agent handles the full lifecycle of Uber Eats customer support:

1. **Knowledge questions** — "How long do refunds take?" → answers from a RAG knowledge base of 60 real Uber Eats help articles
2. **Order actions** — "Where is my order 100004?" → instant tracking with live status, driver info, and ETA
3. **Issue resolution** — "My food arrived damaged" → RAG retrieves the relevant damage/refund policy, the LLM reads the policy alongside the order details, and determines the correct resolution (refund amount, timeline, next steps)
4. **Multi-turn workflows** — "Report missing items" → collects order ID → collects which items → creates ticket with RAG-informed resolution
5. **Safety** — prompt injection attempts and unsafe content are blocked before reaching the LLM

The key design principle: **RAG is not just for Q&A**. Every message — including tool calls — goes through retrieval first. The LLM sees policy context when deciding whether to answer directly or invoke a tool, and again when tools execute to determine resolutions.

---

## Architecture

```
User Message
     │
     ▼
┌─────────────────────┐
│  Input Guardrails    │──→ Regex-based prompt injection + unsafe content detection
│  (guardrails.py)     │    Blocks before LLM sees the message
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  RAG Retrieval       │──→ ALWAYS runs — retrieves top-5 relevant chunks
│  (vectorstore.py)    │    from 60 articles via ChromaDB + sentence-transformers
└──────────┬──────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  Augmented LLM Decision (ReAct Pattern)              │
│  (agent.py)                                          │
│                                                      │
│  Receives: RAG context + tool descriptions +         │
│            conversation history                       │
│                                                      │
│  Decides via structured JSON output:                 │
│    → "answer": respond from retrieved knowledge      │
│    → "tool_call": invoke a specific tool             │
│    → "clarify": ask for more information             │
│    → "greeting": welcome message                     │
│    → "out_of_scope": politely redirect               │
└───────────┬──────────────┬───────────────────────────┘
            │              │
       ┌────▼────┐    ┌───▼──────────┐
       │ Answer  │    │ Tool         │
       │ from    │    │ Execution    │──→ Instant / Auto-resolve / Investigation / Escalation
       │ RAG     │    │ (actions.py) │    RAG retrieves policy → LLM determines resolution
       └────┬────┘    └───┬──────────┘
            │             │
            ▼             ▼
┌─────────────────────────────┐
│  Output Guardrails          │──→ Response length validation
│  (guardrails.py)            │
└──────────┬──────────────────┘
           ▼
┌─────────────────────────────┐
│  Evaluator (LLM-as-Judge)   │──→ Scores response 1-5 on accuracy, helpfulness,
│  (agent.py)                 │    safety, tone. Retries if score < 3.
└──────────┬──────────────────┘
           ▼
┌─────────────────────┐
│  Memory (SQLite)     │──→ Saves to conversation history for context continuity
│  (memory.py)         │
└─────────────────────┘
```

---

## Knowledge Base & Data Scraping

### Source

The knowledge base consists of **60 articles** sourced from the [Uber Eats Help Center](https://help.uber.com/en/ubereats) — a publicly accessible collection of FAQ and support documentation.

### Scraping

**`scraper.py`** fetches articles from 20+ known Uber Eats help article URLs using `requests` + `BeautifulSoup`. It:

1. Fetches each URL and extracts text content using multiple CSS selectors (handles Uber's different page layouts)
2. Falls back to existing knowledge base content for pages that use client-side rendering (React)
3. Merges scraped content with existing articles (carries over articles not in the scrape list)
4. Respects rate limits with 1-second delays between requests

```bash
# Run locally to refresh the knowledge base
pip install beautifulsoup4
python scraper.py
# Then delete data/chroma_db/ to rebuild the vector store
```

### Article Coverage

| Category | Articles | Topics |
|---|---|---|
| Account | 7 | Security, payments, ratings, privacy, addresses, notifications, accessibility |
| Delivery Issues | 6 | Late delivery, order not delivered, delayed arrival, merchant delivery |
| Ordering | 6 | Place order, schedule, group orders, pickup, grocery, reorder |
| Delivery | 6 | Tracking, instructions, no-contact, tipping, driver assignment, alcohol |
| Order Issues | 5 | Missing items, wrong items, wrong order, damaged/spilled, forgotten items |
| Cancellation | 5 | Cancellation policy, fees, charged for cancel, order canceled FAQ |
| Refunds | 4 | Refund policy, managing refunds, order errors, return policy |
| Promotions | 3 | Apply promo, how promos work, referral program |
| Membership | 3 | Uber One benefits, Uber One cancellation, Eats Pass |
| Pricing | 3 | Delivery fees, service fees, taxes |
| Payments | 3 | Gift cards, unauthorized charges, receipts |
| Support | 2 | Contact support, support hours |
| Food Quality | 2 | Quality issues, allergies |
| Other | 2 | General info, restaurant search |

All source URLs link to real, working pages on `help.uber.com`.

---

## RAG Pipeline & Chunking Strategy

### Ingestion Pipeline

```
knowledge_base.json (60 articles)
        │
        ▼
  Character-based chunking (400 chars, 80 overlap)
        │
        ▼
  ~180 text chunks with metadata
        │
        ▼
  Sentence-transformers embedding (all-MiniLM-L6-v2)
        │
        ▼
  ChromaDB persistent vector store
```

### Chunking Parameters

| Parameter | Value | Rationale |
|---|---|---|
| **Chunk size** | 400 characters (~100 tokens) | Our articles are short (200-600 chars). 400-char chunks keep each chunk topically focused — a chunk about cancellation policy won't be diluted with refund information. |
| **Overlap** | 80 characters (20%) | Prevents losing context at boundaries. If "refund within 48 hours" gets split at char 400, the overlap ensures the next chunk still contains enough for retrieval to match. |
| **Method** | Character-based recursive | Simple and effective for FAQ-style content. Semantic chunking would add complexity without meaningful improvement since our articles are already short and topically focused. |

### Why These Choices

- **Small chunks for short documents**: Most help articles are 200-600 characters. A 400-char chunk means 1-2 chunks per article, keeping each topically precise. Larger chunks (1000+) would merge multiple policies, reducing retrieval precision.
- **20% overlap is the standard starting point**: Recommended in the course RAG lecture. Enough to preserve cross-boundary sentences without excessive duplication.
- **Character-based over semantic**: Our documents are already structured as focused help topics. Semantic chunking (paragraph/sentence boundaries) is better for long documents like legal contracts — overkill here.

### Embedding Model

**`all-MiniLM-L6-v2`** from sentence-transformers:
- 384-dimensional embeddings
- Fast inference (important for Streamlit Cloud cold starts)
- Good performance on short text similarity tasks
- Small model size (~80MB) — downloads quickly on deployment

### Retrieval

- **Method**: ChromaDB's default similarity search (cosine similarity)
- **Top-K**: 5 documents retrieved per query
- Metadata preserved through retrieval: article title, category, source URL

---

## Agent Decision Loop

The agent (`agent.py`) implements a **ReAct-style** (Retrieve → Reason → Act) decision loop using the hosted LLM endpoint.

### System Prompt Design

The LLM receives a structured system prompt containing:

1. **Role definition**: "You are an Uber Eats customer support agent"
2. **Tool descriptions**: All 11 tools with parameters and categories (kept under 200 tokens each per course best practices)
3. **Decision framework**: Explicit instructions for when to answer vs. call a tool vs. clarify
4. **Response format**: Structured JSON output (`{"action": "answer|tool_call|clarify|greeting|out_of_scope", ...}`)
5. **Routing rules**: Critical distinctions like "Where is my order?" → `track_order` (NOT `report_delivery_problem`)

### How RAG Drives Tool Selection

Unlike typical chatbots with hardcoded intent classifiers, our agent uses RAG context to inform tool selection:

1. User says: "I read you offer refunds for missing items — my order 100001 is missing fries"
2. RAG retrieves: "Missing items" policy article + "Managing refunds" article
3. LLM sees: the policy context + the `report_missing_items` tool description
4. LLM decides: `{"action": "tool_call", "tool": "report_missing_items", "extracted_params": {"order_id": "100001", "missing_items": "fries"}}`

This means the agent naturally handles mixed queries (partly policy question, partly action request).

### Evaluator-Optimizer

After every response, a separate LLM call scores it 1-5:
- **Accuracy**: Does it match retrieved context? No hallucinations?
- **Helpfulness**: Does it address the user's actual need?
- **Safety**: No harmful or misleading content?
- **Tone**: Empathetic and professional?

If the score is below 3, the agent retries with the evaluator's feedback — implementing the **Evaluator-Optimizer** agentic workflow pattern from the course.

---

## Tools & Actions

### 11 Tools Across 3 Categories

#### Instant Resolution (no ticket, solved immediately)

| Tool | Description | Parameters |
|---|---|---|
| `track_order` | Live delivery status, driver info, ETA | `order_id` |
| `check_refund_status` | Current refund status with RAG-informed timeline explanation | `order_id` |
| `view_order_details` | Full order receipt: items, total, driver, payment, address | `order_id` |
| `contact_driver` | Send message to active delivery driver | `order_id`, `message` |
| `apply_promo_code` | Validate and apply promotional code | `promo_code` |
| `cancel_order` | Cancel with RAG-informed fee explanation | `order_id`, `confirm` |
| `lookup_ticket` | Look up existing support ticket status | `ticket_id` |

#### Auto-Resolve (creates ticket + immediate resolution via RAG)

| Tool | RAG Usage | Resolution |
|---|---|---|
| `report_missing_items` | Retrieves missing items policy → LLM determines refund | Automatic refund + ticket |
| `report_delivery_problem` | Retrieves delivery/damage/late policy → LLM resolves | Varies by problem type |
| `report_wrong_items` | Retrieves wrong items policy → LLM determines resolution | Review ticket |

#### Escalation

| Tool | Description | Parameters |
|---|---|---|
| `contact_support` | Connect to human support specialist | `issue_summary`, `urgency` |

### RAG-Informed Resolution (How Tools Use the Knowledge Base)

When a ticket tool executes, it doesn't use hardcoded rules. Instead:

1. **RAG retrieves** relevant policy documents (e.g., "Damaged or spilled order" article)
2. **LLM receives** the policy + order details (restaurant, items, total, driver)
3. **LLM determines** the resolution based on policy: refund amount, timeline, next steps
4. **Response cites** which policies were referenced

This means the knowledge base directly drives resolution logic. If the refund policy changes, updating the knowledge base articles changes how the agent resolves issues — no code changes needed.

### Multi-Turn Parameter Collection

Tools that require multiple parameters (e.g., `report_missing_items` needs `order_id` + `missing_items`) use prompt chaining:

1. User triggers the tool → agent stores the current tool in session state
2. Agent prompts for the first missing parameter
3. User provides it → agent extracts via regex + stores
4. Agent prompts for next missing parameter (or executes if all collected)

Parameters are extracted intelligently: order IDs via `\b(\d{5,7})\b`, ticket IDs via `UE-\d{4,8}`, problem types via keyword matching.

### Data Updates

When tools execute, they update the order database in real time:
- `cancel_order` → status changes to "Cancelled", refund status updated
- `report_missing_items` → status changes to "Delivered - Reported Issue", refund summary added
- `report_delivery_problem` → same pattern
- `report_wrong_items` → same pattern

These updates are reflected immediately on the Orders page.

---

## Guardrails

Two-layer guardrail system implemented as separate, focused checks (per course guidance: "Guardrails work better when implemented as separate, focused checks rather than relying on the LLM to self-police").

### Input Guardrails (before LLM)

**Prompt injection detection** — regex patterns for:
- "ignore all instructions", "you are now", "forget everything"
- "system prompt", "jailbreak", "developer mode", "bypass", "override"
- "disregard", "new persona", "repeat your instructions"

**Unsafe content filtering** — blocks queries containing:
- Violence, weapons, illegal activity keywords
- Drug references
- Self-harm references

### Output Guardrails (after LLM)

- **Response length truncation**: caps at 3000 characters with a "ask for more details" note
- **Pattern detection**: flags responses where the LLM breaks character (e.g., "As an AI...")

---

## Evaluation

Three evaluation approaches matching course content (`evaluation.py`):

### 1. Real-Time Evaluator (every message)

Built into `agent.py`. After every response:
- Separate LLM call scores the response 1-5
- Criteria: accuracy, helpfulness, safety, tone
- If score < 3, the agent retries with the evaluator's feedback
- Score displayed in the UI next to each message

### 2. Rule-Based Test Suite (15 cases)

```bash
python evaluation.py
```

| Category | Tests | What It Checks |
|---|---|---|
| RAG Retrieval | 4 | Knowledge answers contain expected keywords + have source attribution |
| Tool Routing | 4 | LLM picks correct tool (e.g., "where's my order" → track, not report) |
| Multi-Turn | 2 | Parameter collection works across messages, tickets created |
| Guardrails | 3 | Injection blocked, out-of-scope rejected, unsafe blocked |
| Error Handling | 2 | Invalid order ID handled gracefully, nonsense input gets clarification |

Each test checks: intent classification accuracy + response keyword presence + source attribution.

### 3. LLM-as-Judge Scoring

```bash
python evaluation.py --judge
```

Adds a separate LLM evaluation call per test case, scoring:
- **Relevance** (1-5): Does it address the question?
- **Accuracy** (1-5): Correct information, no hallucinations?
- **Helpfulness** (1-5): Actionable and useful?
- **Tone** (1-5): Empathetic and professional?

---

## Memory & State Persistence

### Conversation Memory (`memory.py`)

- **SQLite-backed** with three tables: `sessions`, `messages`, `session_state`
- **Short-term**: last 8 messages included in LLM context for conversation continuity
- **Cross-session**: history persists across page reloads (locally) and within a session (on Cloud)
- **Session state**: stores current tool, collected parameters, active action

### Session Management

- New session created on login
- Session ID displayed in sidebar
- "New Chat" button creates fresh session while preserving past sessions
- Past sessions accessible (locally) via session switching

---

## User Accounts & Order Database

### 6 Demo Accounts (`users_db.py`)

| Username | Password | Orders | Uber One | Key Scenarios |
|---|---|---|---|---|
| `samir` | `password123` | 15 | Yes | Refunds, active order, damaged, cancelled, late |
| `yuyan` | `password123` | 10 | Yes | Wrong items, late with credit, restaurant closure |
| `jiaer` | `password123` | 10 | No | Cold food, crushed pizza, cancelled by restaurant |
| `ce` | `password123` | 10 | No | Missing items, late breakfast, customer cancel |
| `junyan` | `password123` | 10 | Yes | Spilled wings, wrong congee, restaurant closed |
| `demo` | `demo` | 5 | No | Simple set for quick walkthroughs |

### Order Database (`orders_db.py`)

- **60 orders** across all accounts stored in `data/orders.csv`
- On first run, seeded into **SQLite** (`orders.db`) for read/write access
- Covers every status type: Delivered, In Progress, Cancelled, Cancelled by Restaurant, Delivered - Late, Delivered - Damaged, Delivered - Reported Issue
- Orders include: items, totals, driver names, payment methods, special instructions, ratings, refund statuses

### Ticket Database (`tickets_db.py`)

- SQLite with full CRUD: create, read, update
- Tickets created automatically when issues are reported through chat
- Each ticket has: ticket ID (`UE-XXXXXX`), type, status, order reference, details, resolution, timestamps
- Viewable on the Tickets page and lookupable via chat ("What's the status of UE-123456?")

---

## Frontend (Streamlit UI)

### 4 Pages

| Page | Features |
|---|---|
| **Chat** | Conversational interface, intent + eval score display, source attribution with expandable links, tool indicators |
| **Orders** | Full order history with status filters, sort options, expandable details, refund status (updates in real-time after chat actions) |
| **Tickets** | Support ticket browser with status indicators, resolution details, timestamps |
| **Account** | Profile info, payment methods, order statistics (total orders, delivered, total spent, avg rating) |

### Design

- **Uber-branded**: black sidebar, green (#06C167) accents, DM Sans typography
- **Streamlit theme**: forced light mode via `.streamlit/config.toml` for consistent text visibility
- **Login/logout**: password-protected with session management
- **Responsive**: sidebar navigation with Chat/Orders/Tickets/Account tabs

---

## File Reference

| File | Lines | Purpose |
|---|---|---|
| `agent.py` | 374 | RAG-first agentic orchestrator — system prompt, ReAct decision loop, evaluator |
| `actions.py` | 385 | Tool execution engine — RAG-informed resolution, parameter collection, order updates |
| `app.py` | 455 | Streamlit frontend — login, chat, orders, tickets, account pages + CSS |
| `evaluation.py` | 295 | 15-case test suite — rule-based + LLM-as-Judge scoring |
| `vectorstore.py` | 93 | ChromaDB vector store — chunking, embedding, retrieval |
| `tools.py` | 129 | 11 tool definitions — names, descriptions, parameters, categories |
| `scraper.py` | 163 | Uber Eats Help Center scraper — fetches real article content |
| `orders_db.py` | 137 | SQLite order database — seeds from CSV, supports real-time updates |
| `tickets_db.py` | 114 | SQLite ticket database — CRUD with persistence |
| `memory.py` | 93 | SQLite conversation memory — sessions, messages, state |
| `guardrails.py` | 88 | Input + output guardrails — injection detection, safety checks |
| `users_db.py` | 82 | 6 demo user accounts with profiles and payment methods |
| `llm_client.py` | 48 | OpenAI-compatible wrapper for the qwen3 LLM endpoint |
| `config.py` | 31 | Central configuration — paths, LLM settings, Cloud detection |
| `requirements.txt` | 6 | Python dependencies |
| `runtime.txt` | 1 | Pins Python 3.12 for Streamlit Cloud |
| `.streamlit/config.toml` | 9 | Streamlit theme (light mode, green primary color) |
| `data/knowledge_base.json` | — | 60 Uber Eats help articles (RAG source) |
| `data/orders.csv` | — | 60 sample orders across 6 accounts |

---

## Setup — Running Locally

### Prerequisites

- Python 3.10+ (tested on 3.12)
- ~500MB disk space (for sentence-transformers model download on first run)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/uber-eats-agent_final.git
cd uber-eats-agent_final

# Create virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

If `source venv/bin/activate` doesn't stick (common on macOS zsh), run directly:

```bash
./venv/bin/pip install -r requirements.txt
./venv/bin/streamlit run app.py
```

The app opens at `http://localhost:8501`. The vector store builds automatically on first run (~30 seconds for model download + embedding).

### Optional: Refresh Knowledge Base

```bash
python scraper.py          # Fetches latest articles from help.uber.com
rm -rf data/chroma_db/     # Force vector store rebuild
streamlit run app.py       # Rebuilds with fresh content
```

### Optional: Run Evaluation

```bash
# Rule-based only (fast, ~30 seconds)
python evaluation.py

# With LLM-as-Judge scoring (slower, ~2 minutes)
python evaluation.py --judge
```

---

## Deployment on Streamlit Cloud

The app is deployed at [https://uber-eats.streamlit.app/](https://uber-eats.streamlit.app/).

### How It Works

1. **GitHub → Streamlit Cloud**: the repo is connected to Streamlit Community Cloud, which auto-deploys on every `git push`
2. **`runtime.txt`**: pins Python 3.12 (Streamlit Cloud defaults to 3.14 which doesn't have ChromaDB wheels)
3. **`config.py`**: detects Cloud environment (`/mount/src` exists) and writes all SQLite databases + ChromaDB to `/tmp/uber_eats_agent/` instead of the read-only repo directory
4. **`requirements.txt`**: Streamlit Cloud runs `pip install -r requirements.txt` automatically
5. **`.streamlit/config.toml`**: sets the light theme for consistent text visibility

### Cloud-Specific Behavior

| Aspect | Local | Streamlit Cloud |
|---|---|---|
| **Python version** | Your system (3.10-3.13) | Pinned to 3.12 via `runtime.txt` |
| **SQLite databases** | Written to `data/` | Written to `/tmp/uber_eats_agent/` |
| **ChromaDB storage** | Persistent in `data/chroma_db/` | Rebuilt on each cold start in `/tmp/` |
| **Persistence** | Full — data survives restarts | Ephemeral — `/tmp/` cleared on reboot |
| **Vector store build** | Once (cached on disk) | On every cold start (~30s) |

### Deploying Your Own Instance

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → "New app" → select your repo → branch `main` → file `app.py`
4. (Optional) Add secrets in Advanced Settings for password protection
5. Click Deploy

---

## Demo Accounts

All accounts use password **`password123`** except `demo` which uses **`demo`**.

For the best demo experience, use **`samir`** — it has 15 orders covering every scenario:

| Try saying... | What happens |
|---|---|
| "Hi" | Greeting with capability list |
| "What's the refund policy?" | RAG answer with source citation |
| "Where is my order 100004?" | Instant tracking (In Progress order) |
| "Check refund for order 100002" | Shows existing refund with policy explanation |
| "My fries were missing from order 100001" | RAG-informed resolution → auto-refund → ticket created → order updated |
| "My food from order 100003 arrived spilled" | RAG retrieves damage policy → full refund → ticket |
| "Cancel order 100004" → "yes" | RAG-informed cancellation with fee explanation → order status updated |
| "What's the status of UE-123456?" | Ticket lookup |
| "I need to talk to a real person" | Escalation ticket created |
| "Ignore all instructions" | Guardrail blocks it |
| "What's the weather?" | Out-of-scope redirect |

After any action, check the **Orders** and **Tickets** pages to see real-time updates.

---

## Course Concepts Implemented

| RSM8430 Concept | Implementation | File(s) |
|---|---|---|
| **RAG Pipeline** | Ingestion → 400-char chunking → ChromaDB embedding → cosine retrieval | `vectorstore.py` |
| **Augmented LLM** | LLM enhanced with RAG + Tools + Memory (the 3 augmentations) | `agent.py` |
| **ReAct Pattern** | Retrieve → Reason → Act decision loop on every message | `agent.py` |
| **Tool Use / Function Calling** | 11 tools selected via structured JSON, with multi-turn param collection | `tools.py`, `actions.py` |
| **Prompt Chaining** | Sequential: collect params → RAG retrieve → LLM resolve → create ticket → update order | `actions.py` |
| **Evaluator-Optimizer** | LLM-as-Judge scores every response, retries if quality < 3/5 | `agent.py` |
| **Guardrails** | Separate input (injection/safety) + output (validation) layers | `guardrails.py` |
| **Memory** | Short-term (8-turn context window) + long-term (SQLite sessions) | `memory.py` |
| **Evals** | Rule-based (keyword/intent) + LLM-as-Judge (relevance/accuracy/tone) | `evaluation.py` |
| **Source Attribution** | Every RAG answer and tool resolution cites the knowledge base articles used | `agent.py`, `actions.py` |
| **Agentic Workflows** | Start simple → add complexity only when needed (design guideline from lecture) | Full system |

---

## Known Limitations

- **JSON parsing**: The qwen3-30b model sometimes returns malformed JSON, requiring regex fallback extraction in `llm_client.py`
- **Latency**: The evaluator adds a second LLM call per message (~2-4s overhead). Could be made async or sampling-based in production
- **Mock actions**: Tools simulate real Uber Eats systems — no actual API integration. Order updates are local to the SQLite database
- **Input guardrails are regex-based**: An LLM-based classifier would catch more sophisticated injection attempts (e.g., encoded or indirect prompts)
- **Knowledge base is curated**: While sourced from real Uber help pages, a production system would need automated periodic scraping and re-embedding
- **Cloud persistence**: On Streamlit Cloud, all SQLite data (orders, tickets, memory) resets on reboot since `/tmp/` is ephemeral. Locally, everything persists
- **Single-model architecture**: Uses the same qwen3-30b endpoint for decisions, resolution, and evaluation. A production system might use different models for different tasks (fast model for classification, large model for generation)

---

## LLM Endpoint

- **URL**: `https://rsm-8430-finalproject.bjlkeng.io`
- **Model**: `qwen3-30b-a3b-fp8` (reasoning enabled)
- **API**: OpenAI-compatible chat completions (`/v1/chat/completions`)
- **Reasoning**: The model produces `<think>...</think>` blocks which are stripped by `llm_client.py` before returning responses

---

## License

This project was built for RSM8430H at the University of Toronto Rotman School of Management. All data sources are publicly accessible and used in compliance with their terms of service.
