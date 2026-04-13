# Uber Eats Customer Support Agent

**RSM8430H: Applications of Large Language Models — Group 17 Final Project**

A production-quality conversational support agent for Uber Eats built on a RAG-first agentic architecture. The LLM is the sole decision maker — it sees RAG context, tool descriptions, and conversation history, then decides whether to answer from policy or execute a tool. Tools execute deterministically; RAG provides policy context for customer-facing messages.

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
3. **Issue resolution** — "My food arrived damaged" → code creates ticket and issues refund, RAG retrieves damage policy, LLM writes customer message referencing the policy
4. **Multi-turn workflows** — "Report missing items" → collects order ID → collects which items → creates ticket → directs to help form for photos
5. **Safety** — prompt injection attempts and unsafe content are blocked before reaching the LLM

The key design principles:

- **The LLM is the sole decision maker.** No pre-router, no regex routing. Every message goes through RAG retrieval, then the LLM decides to answer or call a tool based on context.
- **Code executes, RAG explains.** Tools create tickets, issue refunds, and update orders deterministically. RAG retrieves relevant policy. The LLM writes the customer message referencing that policy.
- **RAG runs twice on tool calls.** First to inform the LLM's decision, then inside the tool to provide policy context for the response message.

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
│  Mid-Tool Check      │──→ If user is mid-tool (multi-turn param collection),
│  (agent.py)          │    store param and continue — no LLM call needed
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  RAG Retrieval       │──→ ALWAYS runs — retrieves top-5 relevant chunks
│  (vectorstore.py)    │    from 60 articles via ChromaDB + remote embeddings
└──────────┬──────────┘
           ▼
┌──────────────────────────────────────────────────────┐
│  LLM Decision — Sole Decision Maker (ReAct Pattern)  │
│  (agent.py)                                          │
│                                                      │
│  Receives: RAG context + 10 tool descriptions +      │
│            conversation history (last 8 messages)     │
│                                                      │
│  Decides via structured JSON output:                 │
│    → "answer": respond from retrieved knowledge      │
│    → "tool_call": invoke a specific tool             │
│    → "clarify": ask for more information             │
│    → "greeting": welcome message                     │
│    → "out_of_scope": politely redirect               │
└───────────┬──────────────┬───────────────────────────┘
            │              │
       ┌────▼────┐    ┌───▼──────────────────────────┐
       │ Answer  │    │ Tool Execution               │
       │ from    │    │ (actions.py)                  │
       │ RAG     │    │                              │
       └────┬────┘    │ 1. Code executes action      │
            │         │ 2. RAG retrieves policy       │
            │         │ 3. LLM writes policy message  │
            │         └───┬──────────────────────────┘
            │             │
            ▼             ▼
┌─────────────────────────────┐
│  Output Guardrails          │──→ Response length + JSON sanitizer
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

**`scraper.py`** fetches articles from 20 known Uber Eats help article URLs using `requests` + `BeautifulSoup`. For each URL it:

1. Sends an HTTP request with browser-like headers (User-Agent, Accept)
2. Parses the HTML and tries multiple CSS selectors to find article content (`div.article-content`, `main article`, `<p>` tags)
3. Takes the first selector returning 50+ characters, cleans whitespace, truncates to 2000 chars
4. Falls back to existing knowledge base content for pages using client-side rendering (React)
5. Merges scraped content with the 40 additional curated articles not in the scrape list
6. Respects rate limits with 1-second delays between requests

**Why only 20 URLs?** Uber's help center is a React single-page app. When `requests` fetches a page, it gets raw HTML before JavaScript runs — most pages return empty `<div>` containers. These 20 URLs are the ones confirmed to return server-side rendered content. A headless browser (Selenium/Playwright) would expand coverage but adds ~500MB in dependencies and can't run on Streamlit Cloud.

```bash
# Run locally to refresh the knowledge base
pip install beautifulsoup4
python scraper.py
# Then delete chroma_db to force vector store rebuild
rm -rf data/chroma_db/
streamlit run app.py
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
  Remote embedding API (all-MiniLM-L6-v2)
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

### Embedding

**Remote API** at `https://rsm-8430-a2.bjlkeng.io/v1/embeddings` using `all-MiniLM-L6-v2`:
- 384-dimensional embeddings
- Called via `RemoteEmbeddingFunction` class in `vectorstore.py` — implements ChromaDB's `__call__`, `embed_documents`, and `embed_query` interfaces
- Batches requests in groups of 64 to avoid API limits
- No local model download needed — faster cold starts on Streamlit Cloud

**At build time**: every chunk is sent to the API and the returned vectors are stored in ChromaDB alongside the text and metadata.

**At query time**: the user's message is sent to the same API, ChromaDB finds the 5 closest vectors via cosine similarity, and returns those chunks as RAG context.

### Retrieval

- **Method**: ChromaDB cosine similarity search
- **Top-K**: 5 documents retrieved per query
- Metadata preserved through retrieval: article title, category, source URL

---

## Agent Decision Loop

The agent (`agent.py`) implements a **ReAct-style** (Retrieve → Reason → Act) decision loop. The LLM is the sole decision maker — no pre-router, no regex routing.

### System Prompt Design

The LLM receives a structured system prompt containing:

1. **Role definition**: "You are an Uber Eats customer support agent"
2. **Tool descriptions**: All 10 tools with parameters and categories
3. **Decision framework**: When to use a tool vs. answer from RAG
4. **Explicit routing examples**: "order ID + problem → tool_call", "policy question → answer", "ticket ID → lookup_ticket"
5. **Response format**: Structured JSON output (`{"action": "answer|tool_call|clarify|greeting|out_of_scope", ...}`)

### How Decisions Work

The LLM sees: system prompt + 5 RAG chunks with source metadata + last 8 conversation messages + user message. It returns structured JSON choosing an action:

- `"answer"` — respond using RAG knowledge (policy questions with no order ID)
- `"tool_call"` — select a tool and extract parameters (order-specific actions)
- `"clarify"` — ask for more info (genuinely ambiguous)
- `"greeting"` — welcome message
- `"out_of_scope"` — redirect non-Uber-Eats topics

### RAG Runs Twice on Tool Calls

1. **Before the LLM decides** — RAG retrieval at Step 4 gives the LLM policy context to inform its tool selection
2. **Inside the tool** — `actions.py` calls `_get_policy()` with a targeted query (e.g., "missing items refund policy 48 hours") and passes the retrieved policy to a second LLM call that writes the customer-facing message

This means updating the knowledge base articles changes both how the agent answers questions AND how it communicates resolutions — no code changes needed.

### Evaluator-Optimizer

After every response, a separate LLM call scores it 1-5:
- **Accuracy**: Does it match retrieved context? No hallucinations?
- **Helpfulness**: Does it address the user's actual need?
- **Safety**: No harmful or misleading content?
- **Tone**: Empathetic and professional?

If the score is below 3, the agent retries with the evaluator's feedback using a natural language system prompt (not the JSON one, to avoid getting JSON back).

---

## Tools & Actions

### 10 Tools Across 3 Categories

#### Instant (5 tools — data lookup, no ticket)

| Tool | What It Does |
|---|---|
| `track_order` | Shows live delivery status, driver info, ETA. Different responses for In Progress / Delivered / Cancelled. |
| `check_refund_status` | Shows refund amount and status. RAG retrieves refund timeline policy for the explanation. |
| `view_order_details` | Full order receipt: items, total, driver, payment, address, special instructions, rating. |
| `contact_driver` | Sends message to active delivery driver. Blocked if order already delivered. |
| `lookup_ticket` | Shows ticket status, type, resolution, timestamps. Status-specific next-step messages. |

#### Action (4 tools — create ticket, update order, follow real Uber policy)

| Tool | What It Does | Real Uber Behavior |
|---|---|---|
| `cancel_order` | Cancels order + determines refund based on status | Free if not accepted; partial refund if preparing; blocked if delivered |
| `report_missing_items` | Creates ticket, auto-refunds missing items, updates order | Refund for item sales price + tax; links to help form; 48-hour window |
| `report_wrong_items` | Creates ticket under review, requests photos | Not auto-resolved — needs photo verification; links to app help flow |
| `report_delivery_issue` | Creates ticket with problem-specific resolution | See breakdown below |

#### Delivery Issue Resolution by Type

| Problem | Status | Resolution | Real Uber Behavior |
|---|---|---|---|
| Not delivered | Investigating | GPS + delivery confirmation check | Full refund if confirmed; 2-hour response |
| Damaged/spilled | Auto-Resolved | Full refund | Immediate refund; photo submission optional |
| Late (20+ min) | Auto-Resolved | 20% Uber Cash credit | Per "latest arrival by" policy |
| Cold food | Auto-Resolved | 50% refund | Can request full via app help form |
| Driver issue | Escalated to Agent | Safety team review | 24-hour specialist contact |

#### Escalation (1 tool)

| Tool | What It Does |
|---|---|
| `contact_support` | Creates escalation ticket with priority and wait time. Provides real Uber Eats phone number (1-800-253-6882) and help URL. |

### How Tool Execution Works (Code Executes, RAG Explains)

When the LLM selects a tool, `_handle_tool_call()` in `agent.py`:

1. **Stores extracted params** — whatever the LLM extracted from the message
2. **Regex extraction pass** — `_try_extract_params()` catches order IDs, ticket IDs, problem keywords the LLM may have missed
3. **Missing param check** — if params are incomplete, sets `current_tool` in session state and prompts for the next one. Next message enters mid-tool flow.
4. **Executes** — `execute_tool()` in `actions.py` runs the tool

Inside each action tool in `actions.py`:
1. **Code executes deterministically** — creates ticket in SQLite, updates order status and refund, determines resolution amount
2. **RAG retrieves targeted policy** — e.g., `_get_policy("missing items refund policy 48 hours")`
3. **LLM writes customer message** — `_policy_message()` sends the action taken + policy + order details to the LLM with instruction: "Confirm what was done, reference the policy. Do NOT ask the customer to do anything — the action is complete."
4. **Response assembled** — ticket ID, resolution details, policy message, help form link, ticket reference

### Multi-Turn Parameter Collection

Tools that need multiple parameters use prompt chaining:
1. LLM selects tool, extracts what it can → stored in session state
2. System prompts for missing param → user provides it → stored
3. Repeat until all params collected → execute

Parameters are extracted via regex from the raw message: order IDs (`\b\d{5,7}\b`), ticket IDs (`UE-\d{4,8}`), problem types (keyword matching), missing items (text after "missing"/"forgot" keywords).

---

## Guardrails

Two-layer guardrail system implemented as separate, focused checks.

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

- **Response length truncation**: caps at 3000 characters
- **JSON sanitizer**: if the response accidentally contains raw JSON (`{"action":...}`), re-generates a natural language response with a separate LLM call

---

## Evaluation

Three evaluation approaches matching course content:

### 1. Real-Time Evaluator (every message)

Built into `agent.py`. After every response:
- Separate LLM call scores the response 1-5
- Criteria: accuracy, helpfulness, safety, tone
- If score < 3, retries with evaluator's feedback (natural language prompt, not JSON)
- Score displayed in the UI next to each message

### 2. Rule-Based Test Suite (15 cases)

```bash
python evaluation.py
```

| Category | Tests | What It Checks |
|---|---|---|
| RAG Retrieval | 4 | Knowledge answers contain expected keywords + source attribution |
| Tool Routing | 4 | LLM picks correct tool (e.g., "where's my order" → track, not report) |
| Multi-Turn | 2 | Parameter collection across messages, tickets created |
| Guardrails | 3 | Injection blocked, out-of-scope rejected, unsafe blocked |
| Error Handling | 2 | Invalid order ID handled gracefully, nonsense input gets clarification |

### 3. Metrics Runner (`eval_metrics_run.py`)

```bash
python eval_metrics_run.py
python eval_metrics_run.py --json-out eval_results.json
```

Measures five quantitative metrics:
- **Intent accuracy**: correct intent routing
- **Tool selection rate**: right tool chosen when a tool is needed
- **Argument correctness**: correct order ID extracted, ticket created in DB
- **End-to-end success**: full pipeline pass (intent + tool + args + keywords + sources)
- **Latency (p50/p95)**: wall-clock time per response including evaluator

---

## Memory & State Persistence

### Conversation Memory (`memory.py`)

- **SQLite-backed** with three tables: `sessions`, `messages`, `session_state`
- **Short-term**: last 8 messages included in LLM context for conversation continuity
- **Cross-session**: history persists across page reloads (locally) and within a session (on Cloud)
- **Session state**: stores current tool, collected parameters, active action

### Session Management

- New session created on login
- "New Chat" button creates fresh session
- Past sessions accessible locally via session switching

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
- Orders update in real-time after tool actions (status, refund fields)
- Covers: Delivered, In Progress, Cancelled, Cancelled by Restaurant, Delivered - Late, Delivered - Damaged

### Ticket Database (`tickets_db.py`)

- SQLite with full CRUD: create, read, update
- Tickets created automatically when issues are reported through chat
- Each ticket has: ticket ID (`UE-XXXXXX`), type, status, order reference, details, resolution, timestamps
- Viewable on the Tickets page and lookupable via chat

---

## Frontend (Streamlit UI)

### 4 Pages

| Page | Features |
|---|---|
| **Chat** | Conversational interface, intent + eval score display, source attribution with expandable links |
| **Orders** | Full order history with status filters, expandable details, refund status (updates in real-time) |
| **Tickets** | Support ticket browser with status indicators, resolution details, timestamps |
| **Account** | Profile info, payment methods, order statistics |

### Design

- **Uber-branded**: black sidebar, green (#06C167) accents
- **Streamlit theme**: forced light mode via `.streamlit/config.toml`
- **Login/logout**: password-protected with session management

---

## File Reference

| File | Purpose |
|---|---|
| `agent.py` | RAG-first agentic orchestrator — LLM sole decision maker, ReAct loop, evaluator |
| `actions.py` | Tool execution — code executes + RAG explains, real Uber Eats behavior |
| `app.py` | Streamlit frontend — login, chat, orders, tickets, account pages + CSS |
| `evaluation.py` | 15-case test suite — rule-based + LLM-as-Judge scoring |
| `eval_metrics_run.py` | Advanced metrics — intent, tool, argument, E2E, latency |
| `vectorstore.py` | ChromaDB vector store — chunking, remote embedding, retrieval |
| `tools.py` | 10 tool definitions — names, descriptions, parameters, categories |
| `scraper.py` | Uber Eats Help Center scraper — fetches real article content |
| `orders_db.py` | SQLite order database — seeds from CSV, supports real-time updates |
| `tickets_db.py` | SQLite ticket database — CRUD with persistence |
| `memory.py` | SQLite conversation memory — sessions, messages, state |
| `guardrails.py` | Input + output guardrails — injection detection, safety, JSON sanitizer |
| `users_db.py` | 6 demo user accounts with profiles and payment methods |
| `llm_client.py` | OpenAI-compatible wrapper for the qwen3 LLM endpoint |
| `config.py` | Central configuration — paths, LLM/embedding endpoints, Cloud detection |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Pins Python 3.12 for Streamlit Cloud |
| `.streamlit/config.toml` | Streamlit theme (light mode, green primary color) |
| `data/knowledge_base.json` | 60 Uber Eats help articles (RAG source) |
| `data/orders.csv` | 60 sample orders across 6 accounts |

---

## Setup — Running Locally

### Prerequisites

- Python 3.10+ (tested on 3.12)
- Internet access (for remote LLM and embedding endpoints)

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

The app opens at `http://localhost:8501`. The vector store builds automatically on first run (embeds ~180 chunks via the remote API).

### Reset Data

```bash
rm -f data/orders.db data/tickets.db data/memory.db
rm -rf data/chroma_db/
streamlit run app.py    # Rebuilds everything fresh from CSV + JSON
```

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

# Full metrics suite
python eval_metrics_run.py
```

---

## Deployment on Streamlit Cloud

The app is deployed at [https://uber-eats.streamlit.app/](https://uber-eats.streamlit.app/).

### How It Works

1. **GitHub → Streamlit Cloud**: the repo is connected to Streamlit Community Cloud, which auto-deploys on every `git push`
2. **`runtime.txt`**: pins Python 3.12 (Streamlit Cloud defaults to 3.14 which doesn't have ChromaDB wheels)
3. **`config.py`**: detects Cloud environment (`/mount/src` exists) and writes all SQLite databases + ChromaDB to `/tmp/uber_eats_agent/` instead of the read-only repo directory
4. **Auto-rebuild**: every deploy wipes `/tmp/`, so ChromaDB rebuilds from scratch using the remote embedding API on first request
5. **`.streamlit/config.toml`**: sets the light theme for consistent text visibility

### Cloud-Specific Behavior

| Aspect | Local | Streamlit Cloud |
|---|---|---|
| **Python version** | Your system (3.10-3.13) | Pinned to 3.12 via `runtime.txt` |
| **Embeddings** | Remote API call | Remote API call (same) |
| **SQLite databases** | Written to `data/` | Written to `/tmp/uber_eats_agent/` |
| **ChromaDB storage** | Persistent in `data/chroma_db/` | Rebuilt on each cold start in `/tmp/` |
| **Persistence** | Full — data survives restarts | Ephemeral — `/tmp/` cleared on reboot |

### Deploying Your Own Instance

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub → "New app" → select your repo → branch `main` → file `app.py`
4. Click Deploy

---

## Demo Accounts

All accounts use password **`password123`** except `demo` which uses **`demo`**.

For the best demo experience, use **`samir`** — it has 15 orders covering every scenario:

| Try saying... | What happens |
|---|---|
| "Hi" | Greeting with capability list |
| "What's the refund policy?" | RAG answer with source citation |
| "Where is my order 100004?" | Instant tracking (In Progress order) |
| "Check refund for order 100002" | Shows existing refund with RAG-informed timeline |
| "My fries were missing from order 100001" | Creates ticket, auto-refund, links to help form, order updated |
| "My food from order 100003 arrived spilled" | Full refund issued, ticket created, order updated |
| "My order 100005 was late" | 20% Uber Cash credit issued per late delivery policy |
| "Cancel order 100004" → "yes" | Cancellation with policy-based fee explanation |
| "What's the status of UE-123456?" | Ticket lookup with status-specific next steps |
| "I need to talk to a real person" | Escalation ticket + phone number + help URL |
| "Ignore all instructions" | Guardrail blocks it |
| "What's the weather?" | Out-of-scope redirect |

After any action, check the **Orders** and **Tickets** pages to see real-time updates.

---

## Course Concepts Implemented

| RSM8430 Concept | Implementation | File(s) |
|---|---|---|
| **RAG Pipeline** | Ingestion → 400-char chunking → remote embedding → ChromaDB cosine retrieval | `vectorstore.py` |
| **Augmented LLM** | LLM enhanced with RAG + Tools + Memory (the 3 augmentations) | `agent.py` |
| **ReAct Pattern** | Retrieve → Reason → Act — LLM as sole decision maker | `agent.py` |
| **Tool Use** | 10 tools selected via structured JSON, with multi-turn param collection | `tools.py`, `actions.py` |
| **Prompt Chaining** | Collect params → RAG retrieve → code executes → LLM writes policy message | `actions.py` |
| **Evaluator-Optimizer** | LLM-as-Judge scores every response, retries if quality < 3/5 | `agent.py` |
| **Guardrails** | Separate input (injection/safety) + output (length/JSON sanitizer) layers | `guardrails.py` |
| **Memory** | Short-term (8-turn context window) + long-term (SQLite sessions) | `memory.py` |
| **Evals** | Rule-based (keyword/intent) + LLM-as-Judge + metrics runner (intent/tool/E2E/latency) | `evaluation.py`, `eval_metrics_run.py` |
| **Source Attribution** | Every RAG answer cites articles; every tool resolution references policy | `agent.py`, `actions.py` |

---

## Known Limitations

- **JSON parsing**: The qwen3-30b model sometimes returns malformed JSON, requiring regex fallback extraction. A JSON sanitizer catches cases where raw JSON leaks into responses.
- **Latency**: The evaluator adds a second LLM call per message (~2-4s overhead). Could be async or sampling-based in production.
- **Mock actions**: Tools simulate real Uber Eats systems — no actual API integration. Order/ticket updates are local to SQLite.
- **Regex guardrails**: An LLM-based classifier would catch encoded or indirect prompt injections.
- **Scraper coverage**: Uber's help center uses client-side rendering for most pages, limiting the scraper to ~20 direct URLs. Selenium would expand this.
- **Cloud persistence**: On Streamlit Cloud, all data resets on reboot since `/tmp/` is ephemeral. Production would need a managed database.
- **Single-model architecture**: Uses the same qwen3-30b endpoint for decisions, resolution messages, and evaluation. Production might use different models for different tasks.

---

## Endpoints

### LLM

- **URL**: `https://rsm-8430-finalproject.bjlkeng.io`
- **Model**: `qwen3-30b-a3b-fp8` (reasoning enabled)
- **API**: OpenAI-compatible chat completions (`/v1/chat/completions`)
- **Reasoning**: The model produces `<think>...</think>` blocks which are stripped by `llm_client.py`

### Embedding

- **URL**: `https://rsm-8430-a2.bjlkeng.io`
- **Model**: `all-MiniLM-L6-v2`
- **API**: OpenAI-compatible embeddings (`/v1/embeddings`)
- **Output**: 384-dimensional vectors

---

## License

This project was built for RSM8430H at the University of Toronto Rotman School of Management. All data sources are publicly accessible and used in compliance with their terms of service.
