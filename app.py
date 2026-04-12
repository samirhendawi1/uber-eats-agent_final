"""Streamlit frontend for the Uber Eats Support Agent - polished UI."""

import streamlit as st
from agent import process_message
from memory import create_session, get_history
from vectorstore import build_vectorstore
from users_db import authenticate
from orders_db import get_orders_by_customer
from tickets_db import get_tickets_by_customer
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Uber Eats Support", page_icon="🍔", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────
# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── SIDEBAR (black background, white text) ── */
section[data-testid="stSidebar"] {
    background: #000000 !important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stCaption * {
    color: #999999 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #333 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: #222 !important;
    color: #fff !important;
    border: 1px solid #444 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #06C167 !important;
    border-color: #06C167 !important;
}
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #555 !important;
    color: #aaa !important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #E54B4B !important;
    border-color: #E54B4B !important;
    color: #fff !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* ── FORMS (login) ── */
[data-testid="stForm"] {
    background: white !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08) !important;
    border: 1px solid #eee !important;
}
[data-testid="stForm"] .stButton > button {
    background: #06C167 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}
[data-testid="stForm"] .stButton > button:hover {
    background: #048848 !important;
}

/* ── INPUTS ── */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 2px solid #E2E2E2 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: #06C167 !important;
    box-shadow: 0 0 0 3px rgba(6,193,103,0.1) !important;
}

/* ── EXPANDERS ── */
[data-testid="stExpander"] {
    background: white !important;
    border-radius: 14px !important;
    border: 1px solid #eee !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    margin-bottom: 0.5rem !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: white !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    border: 1px solid #eee !important;
}

/* ── BUTTONS (main area) ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── CUSTOM COMPONENTS ── */
.uber-header {
    background: linear-gradient(135deg, #000 0%, #1a1a1a 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
}
.uber-header h1 { color: white !important; margin: 0 !important; font-size: 2rem !important; }
.uber-header p { color: #ccc !important; margin: 0.5rem 0 0 0 !important; }

.uber-badge {
    display: inline-block;
    background: #06C167;
    color: white !important;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.login-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.login-header h1 { font-size: 2.5rem !important; margin-bottom: 0.25rem !important; }
.login-header p { color: #7D7D7D; font-size: 16px; }
</style>
""", unsafe_allow_html=True)



# ── Initialize vector store ───────────────────────────────────────────
@st.cache_resource
def init_vectorstore():
    return build_vectorstore()

with st.spinner("Loading knowledge base..."):
    init_vectorstore()

# ── Session state defaults ────────────────────────────────────────────
for key, default in {
    "logged_in": False, "user": None, "session_id": None,
    "messages": [], "page": "chat",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ═══════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.markdown("")  # spacer
    st.markdown("""
    <div class="login-header">
        <h1>🍔 Uber Eats Support</h1>
        <p>Sign in to get help with your orders</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        st.markdown("")
        st.caption("**Demo accounts:** `samir` . `yuyan` . `jiaer` . `ce` . `junyan` . `demo`")

    if submitted:
        user = authenticate(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user = user
            st.session_state.session_id = create_session()
            st.session_state.messages = []
            st.rerun()
        else:
            with col2:
                st.error("Invalid username or password.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════
# LOGGED IN - SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
user = st.session_state.user

with st.sidebar:
    st.markdown(f"### 👤 {user['full_name']}")
    st.caption(user['email'])
    if user["uber_one"]:
        st.markdown('<span class="uber-badge">✦ Uber One</span>', unsafe_allow_html=True)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.page = "chat"; st.rerun()
    with c2:
        if st.button("📦 Orders", use_container_width=True):
            st.session_state.page = "orders"; st.rerun()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("🎫 Tickets", use_container_width=True):
            st.session_state.page = "tickets"; st.rerun()
    with c4:
        if st.button("👤 Account", use_container_width=True):
            st.session_state.page = "account"; st.rerun()

    st.divider()

    if st.session_state.page == "chat":
        st.caption(f"Session `{st.session_state.session_id}`")
        if st.button("✨ New Chat", use_container_width=True):
            st.session_state.session_id = create_session()
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.markdown("**🛠 Tools**")
        st.markdown(
            "📍 Track order . 💰 Refund status\n\n"
            "📋 Order details . 📱 Contact driver\n\n"
            "🏷 Promo code . ❌ Cancel order\n\n"
            "📦 Missing items . 🔄 Wrong items\n\n"
            "🚗 Delivery issues . 📞 Support\n\n"
            "🔍 Ticket lookup"
        )

    st.divider()
    if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
        for k in ["logged_in", "user", "session_id", "messages", "page"]:
            st.session_state[k] = {"logged_in": False, "user": None, "session_id": None, "messages": [], "page": "chat"}[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# ORDERS PAGE
# ═══════════════════════════════════════════════════════════════════════
if st.session_state.page == "orders":
    st.markdown("""<div class="uber-header">
        <h1>📦 My Orders</h1>
        <p>View and manage your order history</p>
    </div>""", unsafe_allow_html=True)

    orders = get_orders_by_customer(user["user_id"])
    if not orders:
        st.info("No orders yet.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        status_filter = st.selectbox("Filter", ["All"] + sorted(set(o["status"] for o in orders)))
    with c2:
        sort_order = st.selectbox("Sort", ["Newest first", "Oldest first"])

    filtered = orders if status_filter == "All" else [o for o in orders if o["status"] == status_filter]
    if sort_order == "Oldest first":
        filtered = list(reversed(filtered))

    st.caption(f"Showing {len(filtered)} of {len(orders)} orders")

    for o in filtered:
        emoji = "✅" if "Delivered" in o["status"] and "Issue" not in o["status"] and "Late" not in o["status"] and "Damaged" not in o["status"] else (
            "🔄" if "Progress" in o["status"] else (
            "❌" if "Cancel" in o["status"] else "⚠️"))

        with st.expander(f"{emoji}  **{o['restaurant']}** - `#{o['order_id']}`  .  {o['order_time'][:10]}  .  **{o['total']}**"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Order ID:** `{o['order_id']}`")
                st.markdown(f"**Status:** {o['status']}")
                st.markdown(f"**Items:** {', '.join(o['items'])}")
                st.markdown(f"**Total:** {o['total']}")
                if o["rating"]:
                    st.markdown(f"**Rating:** {'⭐' * int(o['rating'])}")
            with c2:
                st.markdown(f"**Ordered:** {o['order_time']}")
                st.markdown(f"**Delivered:** {o['delivery_time'] or '-'}")
                st.markdown(f"**Driver:** {o['driver']}")
                st.markdown(f"**Payment:** {o['payment_method']}")
                if o["special_instructions"]:
                    st.markdown(f"**Notes:** _{o['special_instructions']}_")
            if o["refund_status"]:
                st.success(f"💰 {o['refund_status']}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════
# TICKETS PAGE
# ═══════════════════════════════════════════════════════════════════════
if st.session_state.page == "tickets":
    st.markdown("""<div class="uber-header">
        <h1>🎫 Support Tickets</h1>
        <p>Track your open and resolved issues</p>
    </div>""", unsafe_allow_html=True)

    tickets = get_tickets_by_customer(user["user_id"])
    if not tickets:
        st.info("No tickets yet. Report an issue via chat and a ticket will be created automatically.")
        st.stop()

    st.caption(f"{len(tickets)} ticket(s)")
    for t in tickets:
        emoji = {"Auto-Resolved": "✅", "Under Review": "⏳", "Investigating": "🔍", "Escalated to Agent": "📞", "Open": "🟡"}.get(t["status"], "📋")
        created = datetime.fromtimestamp(t["created_at"]).strftime("%b %d, %Y at %H:%M")
        with st.expander(f"{emoji}  `{t['ticket_id']}` - **{t['ticket_type']}**  .  Order #{t['order_id']}  .  {created}"):
            st.markdown(f"**Ticket:** `{t['ticket_id']}`")
            st.markdown(f"**Type:** {t['ticket_type']}")
            st.markdown(f"**Status:** {t['status']}")
            st.markdown(f"**Order:** #{t['order_id']}")
            st.markdown(f"**Created:** {created}")
            if t["details"]:
                for k, v in t["details"].items():
                    st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
            if t["resolution"]:
                st.success(f"✅ **Resolution:** {t['resolution']}")
            elif t["status"] == "Investigating":
                st.warning("⏳ Being investigated - update coming soon.")
            elif t["status"] == "Under Review":
                st.warning("⏳ Under review - expect an update within 24 hours.")
            elif t["status"] == "Escalated to Agent":
                st.info("📞 A specialist is handling this.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════
# ACCOUNT PAGE
# ═══════════════════════════════════════════════════════════════════════
if st.session_state.page == "account":
    st.markdown("""<div class="uber-header">
        <h1>👤 My Account</h1>
        <p>Your profile and order statistics</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Profile")
        st.markdown(f"**Name:** {user['full_name']}")
        st.markdown(f"**Email:** {user['email']}")
        st.markdown(f"**Phone:** {user['phone']}")
        st.markdown(f"**Address:** {user['default_address']}")
        if user["uber_one"]:
            st.markdown("**Uber One:** ✅ Active")
        else:
            st.markdown("**Uber One:** Not subscribed")
    with c2:
        st.markdown("### Payment Methods")
        for pm in user["payment_methods"]:
            st.markdown(f"💳 {pm}")

    st.divider()
    orders = get_orders_by_customer(user["user_id"])
    st.markdown("### Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Orders", len(orders))
    with c2:
        st.metric("Delivered", sum(1 for o in orders if "Delivered" in o["status"]))
    with c3:
        st.metric("Spent", f"${sum(float(o['total']) for o in orders):.0f}")
    with c4:
        ratings = [int(o["rating"]) for o in orders if o["rating"]]
        st.metric("Avg Rating", f"{sum(ratings)/len(ratings):.1f} ⭐" if ratings else "-")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════
# CHAT PAGE
# ═══════════════════════════════════════════════════════════════════════
st.markdown(f"""<div class="uber-header">
    <h1>🍔 Uber Eats Support</h1>
    <p>Hi {user['full_name'].split()[0]}! How can I help you today?</p>
</div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"].get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["metadata"]["sources"]:
                    st.markdown(f"[{src['title']}]({src['url']})")

if user_input := st.chat_input("Ask about orders, refunds, deliveries..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(""):
            result = process_message(
                st.session_state.session_id,
                user_input,
                customer_id=user["user_id"],
            )
        st.markdown(result["response"])

        # Subtle metadata
        intent_colors = {
            "track_order": "📍", "check_refund_status": "💰", "view_order_details": "📋",
            "contact_driver": "📱", "cancel_order": "❌", "report_missing_items": "📦",
            "report_wrong_items": "🔄", "report_delivery_problem": "🚗",
            "contact_support": "📞", "knowledge_question": "📖", "greeting": "👋",
            "lookup_ticket": "🔍", "followup": "↩️", "out_of_scope": "🚫",
            "guardrail_blocked": "🛡️", "clarification_needed": "❓",
            "apply_promo_code": "🏷️",
        }
        intent = result['intent']
        icon = intent_colors.get(intent, ".")
        eval_score = result.get('eval_score')
        eval_text = f" | eval: {eval_score}/5" if eval_score else ""
        st.caption(f"{icon} {intent}{eval_text}")

        if result["sources"]:
            with st.expander("📚 Sources"):
                for src in result["sources"]:
                    st.markdown(f"[{src['title']}]({src['url']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["response"], "metadata": result}
    )
