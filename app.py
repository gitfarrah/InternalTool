import os
import html
import logging

import streamlit as st
from dotenv import load_dotenv

from src.storage.cache_manager import (
    get_cached_search_results,
    cache_search_results,
    get_cache_manager
)
from src.agent import create_pm_agent
from src.auth import get_session_manager
from src.handler.oauth_handler import (
    get_oauth_url,
    exchange_code_for_token,
    get_user_info,
    is_token_valid,
)

# Setup logger
logger = logging.getLogger(__name__)

# =========================
# Environment & Config
# =========================

load_dotenv()

# Page config (do this as early as possible)
st.set_page_config(
    page_title="Internal PM Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Slack Authentication (using Farah's oauth_handler approach)
# =========================

import time

def _is_slack_authenticated_cached() -> bool:
    """Check if Slack token is valid with caching to avoid repeated API calls."""
    token = st.session_state.get("slack_token", "")
    if not token:
        st.session_state["slack_token_valid"] = False
        return False
    now = time.time()
    last_checked = st.session_state.get("slack_token_checked_at", 0)
    # Revalidate at most every 30 minutes
    if st.session_state.get("slack_token_valid") is not None and (now - last_checked) < 1800:
        return bool(st.session_state.get("slack_token_valid"))
    valid = False
    try:
        valid = is_token_valid(token)
    except Exception:
        valid = False
    st.session_state["slack_token_valid"] = valid
    st.session_state["slack_token_checked_at"] = now
    return valid

# OAuth callback handling
try:
    query_params = st.query_params
except Exception:
    query_params = {}

if "code" in query_params:
    try:
        oauth_code = query_params.get("code", "")
        
        # Check if we already have a valid token (avoid reprocessing)
        if st.session_state.get("slack_token") and _is_slack_authenticated_cached():
            logger.info("Already have valid Slack token, skipping OAuth processing")
            try:
                st.query_params.clear()
            except Exception:
                pass
        else:
            # Validate state to avoid CSRF; if expected missing (e.g., cloud restart), allow once
            returned_state = query_params.get("state", "")
            expected_state = st.session_state.get("slack_oauth_state", "")
            if expected_state and returned_state != expected_state:
                st.error("❌ Authentication failed: invalid_state. Please try again.")
                try:
                    st.query_params.clear()
                except Exception:
                    pass
                st.stop()

            # Check if this code was already processed
            processed_code = st.session_state.get("processed_oauth_code", "")
            if processed_code == oauth_code:
                logger.info("OAuth code already processed, clearing params and continuing")
                try:
                    st.query_params.clear()
                except Exception:
                    pass
            else:
                # Process the OAuth code
                logger.info("Processing OAuth code...")
                token = exchange_code_for_token(oauth_code)
                user_info_dict = get_user_info(token)
                st.session_state["slack_token"] = token
                st.session_state["slack_user_name"] = user_info_dict.get("name", "User")
                st.session_state["slack_user_display_name"] = user_info_dict.get("display_name") or user_info_dict.get("name", "User")
                st.session_state["slack_user_id"] = user_info_dict.get("id", "")
                st.session_state["slack_user_email"] = user_info_dict.get("email", "")
                st.session_state["slack_user_real_name"] = user_info_dict.get("real_name", "")
                st.session_state["processed_oauth_code"] = oauth_code  # Mark code as processed
                
                logger.info(f"Slack OAuth successful for user: {st.session_state['slack_user_display_name']}")
                
                # Clear URL params BEFORE showing success message
                try:
                    st.query_params.clear()
                except Exception:
                    pass
                
                st.success(f"✅ Connected as {st.session_state['slack_user_display_name']}")
                # Rerun to reload page without OAuth params
                st.rerun()
    except Exception as e:
        msg = str(e)
        logger.error(f"OAuth error: {msg}", exc_info=True)
        if "invalid_code" in msg.lower() or "already_used" in msg.lower():
            # Clear the processed code flag so user can try again
            if "processed_oauth_code" in st.session_state:
                del st.session_state["processed_oauth_code"]
            st.warning("OAuth code expired or already used. Please click Connect to Slack again.")
            try:
                st.query_params.clear()
            except Exception:
                pass
            if "slack_oauth_state" in st.session_state:
                del st.session_state["slack_oauth_state"]
        else:
            st.error(f"❌ Authentication failed: {msg}")
            logger.error(f"OAuth authentication failed: {msg}", exc_info=True)
            try:
                st.query_params.clear()
            except Exception:
                pass

# Auth gate: require Slack token to proceed
if "slack_token" not in st.session_state or not _is_slack_authenticated_cached():
    # Show login UI
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
            background-color: white;
            padding: 3rem 2rem;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<div style='text-align: center; font-size: 4rem; margin-bottom: 1rem;'>💬</div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #1a202c; margin-bottom: 0.5rem;'>Internal PM Chat</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #718096; margin-bottom: 2rem;'>Your unified workspace for Slack conversations, Confluence docs, and team knowledge</p>", unsafe_allow_html=True)

        try:
            oauth_url = get_oauth_url()
            st.markdown(f"""
                <div style='text-align: center; margin: 2rem 0;'>
                    <a href="{oauth_url}"
                       style='display: inline-block;
                              background: #4A154B;
                              color: white;
                              padding: 12px 32px;
                              border-radius: 8px;
                              text-decoration: none;
                              font-weight: 600;
                              font-size: 1rem;
                              transition: background 0.3s;'>
                        Sign in with Slack
                    </a>
                </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.error("OAuth configuration missing")

        st.write("")
        st.markdown("---")

        st.markdown("### Why Sign in with Slack?")

        col_a, col_b = st.columns([1, 9])
        with col_a:
            st.write("🔒")
        with col_b:
            st.markdown("**Private Channel Access**  \nSearch your private channels and conversations securely")

        col_a, col_b = st.columns([1, 9])
        with col_a:
            st.write("🎯")
        with col_b:
            st.markdown("**Personalized Results**  \nGet search results tailored to your permissions")

        col_a, col_b = st.columns([1, 9])
        with col_a:
            st.write("⚡")
        with col_b:
            st.markdown("**Unified Search**  \nSearch across Slack, Confluence, and documentation")

        st.write("")
        st.caption("🔐 Secured by Slack OAuth 2.0")

    st.stop()

# Get authenticated user info for session management
user_info = {
    "id": st.session_state.get("slack_user_id", ""),
    "name": st.session_state.get("slack_user_name", ""),
    "display_name": st.session_state.get("slack_user_display_name", ""),
    "real_name": st.session_state.get("slack_user_real_name", ""),
    "email": st.session_state.get("slack_user_email", ""),
}

# Initialize session manager
session_manager = get_session_manager()

# Create or update session
if "auth_session_id" not in st.session_state:
    st.session_state.auth_session_id = session_manager.create_session(user_info)

# Update session activity
session_manager.update_session_activity(st.session_state.auth_session_id)

# Validate session is still active
if not session_manager.is_session_valid(st.session_state.auth_session_id):
    st.warning("Your session has expired. Please log in again.")
    # Clear Slack token
    if "slack_token" in st.session_state:
        del st.session_state["slack_token"]
    st.rerun()

# Constants / Tunables
LLM_NAME = "Gemini 2.0 Flash Experimental"

# =========================
# Secrets & Keys
# =========================

def _get_secret_or_env(name: str, default: str = "") -> str:
    if name in st.secrets:
        return st.secrets.get(name, default)
    return os.getenv(name, default)

GEMINI_API_KEY = _get_secret_or_env("GEMINI_API_KEY")

# =========================
# CSS (consolidated)
# =========================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.25rem;
        font-weight: 800;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #444;
        font-size: 1rem;
        margin-bottom: 1.25rem;
    }
    .citation-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 8px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .citation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        border-left-color: #0d5aa7;
    }
    .citation-title {
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 8px;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .citation-evidence {
        font-style: italic;
        color: #444;
        margin-bottom: 8px;
        background: #f0f4f8;
        padding: 10px;
        border-radius: 6px;
        border-left: 3px solid #28a745;
        font-size: 13px;
        line-height: 1.4;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .citation-card a {
        color: #1f77b4;
        text-decoration: none;
        transition: color 0.2s ease;
    }
    .citation-card a:hover {
        color: #0d5aa7;
        text-decoration: underline;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Header with logo
# =========================

st.markdown('<div class="main-header">Internal PM Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions across Confluence, Slack, Docs, Zendesk, and Jira</div>', unsafe_allow_html=True)

# =========================
# Sidebar
# =========================

with st.sidebar:
    # Show authenticated Slack user widget
    if user_info and user_info.get("id"):
        st.markdown("### 👤 Signed in as")
        col1, col2 = st.columns([1, 3])
        with col1:
            # User avatar placeholder (Farah's oauth_handler doesn't return image)
            st.markdown("👤")
        with col2:
            display_name = user_info.get("display_name") or user_info.get("real_name") or user_info.get("name") or "User"
            st.write(f"**{display_name}**")
            if user_info.get("email"):
                st.caption(user_info["email"])

        if st.button("🚪 Sign Out", use_container_width=True):
            # Clear Slack token
            if "slack_token" in st.session_state:
                del st.session_state["slack_token"]
            if "slack_user_name" in st.session_state:
                del st.session_state["slack_user_name"]
            if "slack_user_display_name" in st.session_state:
                del st.session_state["slack_user_display_name"]
            if "slack_user_id" in st.session_state:
                del st.session_state["slack_user_id"]
            if "slack_user_email" in st.session_state:
                del st.session_state["slack_user_email"]
            if "slack_user_real_name" in st.session_state:
                del st.session_state["slack_user_real_name"]
            st.rerun()

        st.markdown("---")

    st.header("Chat Configuration")

    st.info("""
    **Agent searches across:**
    - 📚 Docs (Qdrant)
    - 💬 Slack Messages
    - 📖 Confluence Pages
    - 🎫 Zendesk Tickets (Incorta)
    - 📋 Jira Issues (Incorta)
    """)

    st.markdown("---")
    st.subheader("System Info")

    # Show API key rotation info if available
    api_key_info = "Single API Key"
    if st.session_state.get("agent_executor") is not None:
        executor = st.session_state["agent_executor"]
        # Check if it's a RetryAgentExecutor with api_manager
        if hasattr(executor, 'api_manager'):
            num_keys = len(executor.api_manager.api_keys)
            current_idx = executor.api_manager.current_index
            api_key_info = f"{num_keys} API Keys (Current: #{current_idx + 1})"

    st.info(f"""
    **AI Model:** {LLM_NAME}
    **Vector DB:** Qdrant
    **Search:** Multi-source analysis
    **API Keys:** {api_key_info}
    """)

    # Cache statistics
    st.markdown("---")
    st.subheader("Cache Statistics")
    cache_stats = get_cache_manager().get_stats()
    st.metric("Active Cached Items", cache_stats["active_items"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Cache", use_container_width=True):
            get_cache_manager().clear()
            st.success("Cache cleared!")
            st.rerun()

    with col2:
        if st.button("New Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["chat_history"] = []
            st.success("Chat reset!")
            st.rerun()

    st.markdown("---")
    st.subheader("Usage Tips")
    st.markdown("""
    - Ask follow-up questions naturally
    - Reference previous answers with "it", "that", etc.
    - Use specific keywords for best results
    - All sources searched automatically
    """)

    # Show conversation count
    if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
        st.markdown("---")
        st.metric("Messages in Chat", len(st.session_state["messages"]))

# =========================
# Session state initialization
# =========================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = None

# =========================
# Guardrails
# =========================

def _ensure_gemini_key_if_needed():
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Add it to Streamlit secrets or your environment.")
        st.stop()

# =========================
# Helper Functions
# =========================

def _clean_slack_text(text: str) -> str:
    """Clean Slack message formatting."""
    import re

    text = re.sub(r'<#[A-Z0-9]+\|([^>]+)>', r'#\1', text)
    text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
    text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
    text = re.sub(r'<@[A-Z0-9]+\|([^>]+)>', r'@\1', text)
    text = re.sub(r'<@([A-Z0-9]+)>', r'@unknown_user', text)
    text = text.replace('<!channel>', '@channel')
    text = text.replace('<!here>', '@here')
    text = text.replace('<!everyone>', '@everyone')

    return text


def render_sources(sources):
    """Render source citations grouped by type in separate expanders."""
    if not sources:
        return

    import re

    # Categorize sources by type
    slack_messages = []
    confluence_pages = []
    docs_results = []
    zendesk_results = []
    jira_results = []
    other_sources = []

    # DEBUG: Log each source for categorization analysis
    logger.info(f"DEBUG render_sources - Processing {len(sources)} sources")
    for idx, source in enumerate(sources):
        # Determine source type
        source_type = source.get("source", "unknown")
        url = source.get("url", "") or source.get("permalink", "")
        title = source.get("title", "NO_TITLE")
        has_space = "space" in source

        logger.info(f"DEBUG render_sources [{idx}] - title: {title}, url: {url}, source_type: {source_type}, has_space: {has_space}")

        # Prioritize source field check first for accurate categorization
        if "channel" in source or "permalink" in source or source_type == "slack":
            slack_messages.append(source)
        elif source_type == "confluence" or ("confluence" in url.lower() and source_type != "knowledge_base"):
            confluence_pages.append(source)
            logger.info(f"DEBUG render_sources - CATEGORIZED AS CONFLUENCE: {title}")
        elif source_type == "zendesk" or ("zendesk" in url.lower() and source_type != "knowledge_base"):
            zendesk_results.append(source)
        elif source_type == "jira" or ("jira" in url.lower() and source_type != "knowledge_base"):
            jira_results.append(source)
        elif source_type == "knowledge_base" or (url.startswith("http") and source_type not in ["slack", "confluence", "zendesk", "jira"]):
            docs_results.append(source)
            logger.info(f"DEBUG render_sources - CATEGORIZED AS KNOWLEDGE_BASE: {title}")
        else:
            other_sources.append(source)
            logger.info(f"DEBUG render_sources - CATEGORIZED AS OTHER: {title}")

    logger.info(f"DEBUG render_sources - Final counts: Slack={len(slack_messages)}, Confluence={len(confluence_pages)}, Docs={len(docs_results)}, Zendesk={len(zendesk_results)}, Jira={len(jira_results)}, Other={len(other_sources)}")

    # Render Slack Messages
    with st.expander(f"💬 Slack Messages ({len(slack_messages)} found)", expanded=False):
        if not slack_messages:
            st.info("No Slack results found.")
        else:
            for idx, m in enumerate(slack_messages, 1):
                channel = m.get('channel', 'Unknown')
                channel_label = f"#{channel}"
                if m.get("is_private"):
                    channel_label += " (private)"
                username = m.get('username', 'Unknown')
                timestamp = m.get('date') or m.get('ts', 'Unknown')
                text = _clean_slack_text(m.get('text', '').strip())
                permalink = m.get('permalink', '')
                score = m.get('score', m.get('relevance_score', 0.0)) or 0.0

                text_escaped = html.escape(text)[:500]
                if len(text) > 500:
                    text_escaped += "..."

                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4A90E2;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #1f1f1f;">{channel_label}</span>
                        <span style="color: #666; font-size: 0.9em;">{timestamp}</span>
                    </div>
                    <div style="color: #4A90E2; font-size: 0.9em; margin-bottom: 10px;">@{username}</div>
                    <div style="color: #1f1f1f; line-height: 1.6; margin-bottom: 10px; white-space: pre-wrap;">{text_escaped}</div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <a href="{permalink}" target="_blank" style="color: #4A90E2; text-decoration: none; font-size: 0.9em;">View in Slack</a>
                        <span style="color: #888; font-size: 0.85em;">Score: {score:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Render Confluence Pages
    with st.expander(f"📖 Confluence Pages ({len(confluence_pages)} found)", expanded=False):
        if not confluence_pages:
            st.info("No Confluence results found.")
        else:
            for idx, p in enumerate(confluence_pages, 1):
                title = p.get('title', 'Untitled')
                space = p.get('space', 'Unknown')
                last_modified = p.get('last_modified', 'Unknown')
                excerpt = (p.get('excerpt') or p.get('text', '')).strip()
                url = p.get('url', '')
                score = p.get('score', 0.0)

                # Clean excerpt from HTML tags and highlight markers
                cleaned_excerpt = re.sub(r'@@@hl@@@(.*?)@@@endhl@@@', r'\1', excerpt)
                cleaned_excerpt = re.sub(r'<[^>]+>', '', cleaned_excerpt)
                cleaned_excerpt = cleaned_excerpt.replace('&nbsp;', ' ').replace('&amp;', '&')
                cleaned_excerpt = cleaned_excerpt.replace('&lt;', '<').replace('&gt;', '>')
                if len(cleaned_excerpt) > 300:
                    cleaned_excerpt = cleaned_excerpt[:300] + '...'

                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6B46C1;">
                    <div style="font-weight: 600; color: #1f1f1f; font-size: 1.1em; margin-bottom: 8px;">{html.escape(title)}</div>
                    <div style="display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em;">
                        <span style="color: #6B46C1; font-weight: 500;">{html.escape(space)}</span>
                        <span style="color: #666;">{html.escape(last_modified)}</span>
                        <span style="color: #888;">Score: {score:.2f}</span>
                    </div>
                    <div style="color: #444; line-height: 1.6; margin-bottom: 10px; font-style: italic;">{html.escape(cleaned_excerpt)}</div>
                    <a href="{url}" target="_blank" style="color: #6B46C1; text-decoration: none; font-size: 0.9em;">Open Page</a>
                </div>
                """, unsafe_allow_html=True)

    # Render Knowledge Base Docs
    with st.expander(f"📚 Knowledge Base Docs ({len(docs_results)} found)", expanded=False):
        if not docs_results:
            st.info("No docs results found.")
        else:
            for idx, d in enumerate(docs_results, 1):
                title = d.get('title', 'Untitled')
                url = d.get('url', '')
                text = (d.get('text') or '').strip()
                steps = d.get('steps') or []
                score = d.get('relevance_score', d.get('score', 0.0)) or 0.0
                score_raw = d.get('score_raw')
                source_label = d.get('source', 'Knowledge Base')
                collection = d.get('collection')

                # Prepare meta badges
                score_label = f"{score:.3f}"
                if score_raw is not None:
                    score_label = f"{score:.3f} (raw {score_raw:.3f})"

                meta_badges = [
                    f"<span style='background-color: #e6f4ea; color: #1b5e20; padding: 2px 10px; border-radius: 12px; font-size: 0.8em;'>Score {score_label}</span>",
                    f"<span style='background-color: #e8eaf6; color: #283593; padding: 2px 10px; border-radius: 12px; font-size: 0.8em;'>{html.escape(source_label)}</span>",
                ]
                if collection:
                    meta_badges.append(
                        f"<span style='background-color: #f1f3f4; color: #3c4043; padding: 2px 10px; border-radius: 12px; font-size: 0.8em;'>{html.escape(collection)}</span>"
                    )
                meta_html = " ".join(meta_badges)

                # Build body content (steps preferred, otherwise concise summary)
                if steps:
                    steps_items = ''.join(f"<li>{html.escape(step.strip())}</li>" for step in steps[:6] if step.strip())
                    body_html = (
                        "<div style='color: #444; line-height: 1.6; margin-bottom: 10px;'>"
                        "<strong>Steps:</strong>"
                        f"<ol style='margin: 6px 0 0 20px;'>{steps_items}</ol>"
                        "</div>"
                    )
                else:
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    summary_items = []
                    for sentence in sentences:
                        clean_sentence = sentence.strip()
                        if len(clean_sentence) < 12:
                            continue
                        summary_items.append(html.escape(clean_sentence))
                        if len(summary_items) >= 3:
                            break
                    if summary_items:
                        summary_list = ''.join(f"<li>{item}</li>" for item in summary_items)
                        body_html = (
                            "<div style='color: #444; line-height: 1.6; margin-bottom: 10px;'>"
                            "<strong>Summary:</strong>"
                            f"<ul style='margin: 6px 0 0 18px;'>{summary_list}</ul>"
                            "</div>"
                        )
                    else:
                        truncated = html.escape(text[:300] + ("..." if len(text) > 300 else ""))
                        body_html = (
                            "<div style='color: #444; line-height: 1.6; margin-bottom: 10px;'>"
                            f"{truncated}"
                            "</div>"
                        )

                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28a745;">
                    <div style="font-weight: 600; color: #1f1f1f; font-size: 1.1em; margin-bottom: 8px;">{html.escape(title)}</div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; font-size: 0.9em; color: #666;">{meta_html}</div>
                    {body_html}
                    <a href="{url}" target="_blank" style="color: #28a745; text-decoration: none; font-size: 0.9em;">Open Document</a>
                </div>
                """, unsafe_allow_html=True)

    # Render Zendesk Tickets
    if zendesk_results:
        with st.expander(f"🎫 Zendesk Tickets ({len(zendesk_results)} found)", expanded=False):
            for idx, z in enumerate(zendesk_results, 1):
                title = z.get('title', 'Untitled')
                url = z.get('url', '')
                text = (z.get('text') or z.get('excerpt', '')).strip()
                score = z.get('score', 0.0)

                if len(text) > 300:
                    text = text[:300] + '...'

                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #dc3545;">
                    <div style="font-weight: 600; color: #1f1f1f; font-size: 1.1em; margin-bottom: 8px;">{html.escape(title)}</div>
                    <div style="color: #666; font-size: 0.9em; margin-bottom: 10px;">Score: {score:.2f}</div>
                    <div style="color: #444; line-height: 1.6; margin-bottom: 10px;">{html.escape(text)}</div>
                    <a href="{url}" target="_blank" style="color: #dc3545; text-decoration: none; font-size: 0.9em;">View Ticket</a>
                </div>
                """, unsafe_allow_html=True)

    # Render Jira Issues
    if jira_results:
        with st.expander(f"📋 Jira Issues ({len(jira_results)} found)", expanded=False):
            for idx, j in enumerate(jira_results, 1):
                title = j.get('title', 'Untitled')
                url = j.get('url', '')
                text = (j.get('text') or j.get('excerpt', '')).strip()
                score = j.get('score', 0.0)

                if len(text) > 300:
                    text = text[:300] + '...'

                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #0052CC;">
                    <div style="font-weight: 600; color: #1f1f1f; font-size: 1.1em; margin-bottom: 8px;">{html.escape(title)}</div>
                    <div style="color: #666; font-size: 0.9em; margin-bottom: 10px;">Score: {score:.2f}</div>
                    <div style="color: #444; line-height: 1.6; margin-bottom: 10px;">{html.escape(text)}</div>
                    <a href="{url}" target="_blank" style="color: #0052CC; text-decoration: none; font-size: 0.9em;">View Issue</a>
                </div>
                """, unsafe_allow_html=True)


def build_conversation_context():
    """Build conversation context string for agent."""
    if not st.session_state["chat_history"]:
        return ""

    # Get last 6 messages (3 exchanges)
    recent_history = st.session_state["chat_history"][-6:]
    context_lines = []

    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:200]  # Truncate long messages
        context_lines.append(f"{role}: {content}")

    return "\n".join(context_lines)


def process_query(query: str):
    """Process a user query and return response with sources."""
    _ensure_gemini_key_if_needed()

    # Build conversation context
    conversation_context = build_conversation_context()

    # Enhanced query with context for follow-ups
    enhanced_query = query
    if conversation_context:
        enhanced_query = f"Previous conversation:\n{conversation_context}\n\nCurrent question: {query}"

    # Check cache for agentic search results
    cache_filters = {
        "mode": "agentic",
        "query": query,
        "context": conversation_context[:100]  # Include short context in cache key
    }
    cached_results = get_cached_search_results(query, cache_filters)

    if cached_results:
        return {
            "answer": cached_results.get("final_answer"),
            "sources": cached_results.get("all_sources", []),
            "tools_used": cached_results.get("tools_used", []),
            "from_cache": True
        }

    # Initialize agent executor if not exists
    if st.session_state["agent_executor"] is None:
        with st.spinner("🤖 Initializing agent..."):
            try:
                st.session_state["agent_executor"] = create_pm_agent(api_key=None)
            except Exception as e:
                st.error(f"Failed to initialize agent: {e}")
                st.stop()

    # Run agent
    agent_executor = st.session_state["agent_executor"]

    # Collect agent execution details
    final_answer = None
    all_sources = []
    tools_used = set()
    step_sources = []  # list of lists: sources per observation step

    try:
        # Collect all chunks
        all_chunks = []
        for chunk in agent_executor.stream({"input": enhanced_query}):
            all_chunks.append(chunk)

            # Collect tool usage
            if "actions" in chunk:
                for action in chunk["actions"]:
                    tools_used.add(action.tool)

            if "steps" in chunk:
                for step in chunk["steps"]:
                    # Extract sources from observations
                    try:
                        obs_data = step.observation
                        collected = []
                        
                        # Handle string representation of list (JSON serialized)
                        if isinstance(obs_data, str):
                            try:
                                import json
                                obs_data = json.loads(obs_data)
                            except (json.JSONDecodeError, ValueError):
                                # If not JSON, try eval (less safe but might work for list strings)
                                try:
                                    obs_data = eval(obs_data) if obs_data.strip().startswith('[') else obs_data
                                except:
                                    pass
                        
                        if isinstance(obs_data, list):
                            # DEBUG: Log observation data
                            logger.info(f"DEBUG - Observation contains {len(obs_data)} items")
                            confluence_items = []
                            knowledge_base_items = []
                            for item in obs_data:
                                # Check for source items - include "excerpt" for Confluence results
                                if isinstance(item, dict) and any(k in item for k in ["url", "title", "text", "excerpt", "permalink"]):
                                    all_sources.append(item)
                                    collected.append(item)
                                    # Track Confluence items specifically - check for source field or space field
                                    url = item.get("url", "") or item.get("permalink", "")
                                    source_type = item.get("source", "")
                                    if "confluence" in url.lower() or item.get("space") or source_type == "confluence":
                                        confluence_items.append(item.get("title", "NO_TITLE"))
                                        # Ensure source field is set if missing
                                        if "source" not in item:
                                            item["source"] = "confluence"
                                    # Track knowledge_base items specifically
                                    elif source_type == "knowledge_base":
                                        knowledge_base_items.append(item.get("title", "NO_TITLE"))
                                        # Ensure source field is set if missing
                                        if "source" not in item:
                                            item["source"] = "knowledge_base"
                            if confluence_items:
                                logger.info(f"DEBUG - Found {len(confluence_items)} Confluence items: {confluence_items}")
                            if knowledge_base_items:
                                logger.info(f"DEBUG - Found {len(knowledge_base_items)} Knowledge Base items: {knowledge_base_items}")
                        if collected:
                            step_sources.append(collected)
                    except Exception as e:
                        logger.error(f"DEBUG - Error extracting sources: {e}")
                        pass

            if "output" in chunk:
                final_answer = chunk["output"]

        # Determine which sources were actually used in the answer
        import re
        
        def normalize_url(u: str) -> str:
            try:
                return (u or "").strip().lower()
            except:
                return (u or "").lower()

        def normalize_title(t: str) -> str:
            try:
                return (t or "").strip().lower()
            except:
                return (t or "").lower()

        # Extract keywords from the answer that might reference sources
        mentioned_sources = []
        if isinstance(final_answer, str) and final_answer:
            answer_lower = final_answer.lower()
            answer_words = set(answer_lower.split())
            
            # Find sources that match by:
            # 1. URL domain/partial match in answer
            # 2. Title keywords in answer
            # 3. Source-specific identifiers (ticket numbers, page titles, etc.)
            for s in all_sources:
                url = normalize_url(s.get("url") or s.get("permalink") or "")
                title = normalize_title(s.get("title") or "")
                
                # Check if URL domain or partial URL appears in answer
                if url:
                    # Extract domain or key parts
                    url_parts = url.replace("https://", "").replace("http://", "").split("/")
                    if url_parts and url_parts[0]:
                        domain = url_parts[0].split(".")[0]  # e.g., "confluence" from "confluence.incorta.com"
                        if domain in answer_lower or any(part in answer_lower for part in url_parts[:3] if part):
                            mentioned_sources.append(s)
                            continue
                
                # Check if title keywords appear in answer
                if title and len(title) > 3:
                    title_words = set(title.split())
                    # If 2+ title words appear in answer, likely referenced
                    common_words = title_words.intersection(answer_words)
                    if len(common_words) >= 2:
                        mentioned_sources.append(s)
                        continue
                
                # Check for Slack channel/username mentions
                if "channel" in s:
                    channel = s.get("channel", "").lower()
                    if channel and f"#{channel}" in answer_lower:
                        mentioned_sources.append(s)
                        continue
                
                # Check for ticket/issue numbers (Zendesk/Jira)
                if "ticket" in answer_lower or "issue" in answer_lower:
                    # Look for patterns like "PROD-123" or ticket IDs
                    ticket_patterns = re.findall(r'[A-Z]+-\d+|\d{5,}', answer_lower)
                    if ticket_patterns and url:
                        # If answer mentions ticket patterns and we have a URL, likely related
                        mentioned_sources.append(s)
                        continue

        # Use only sources that were actually mentioned/referenced
        used_sources = mentioned_sources if mentioned_sources else []

        # If still no sources, try a more lenient approach: use only the LAST observation step
        # (the one that directly led to the final answer)
        if not used_sources and step_sources:
            last_step_sources = step_sources[-1]
            # Deduplicate
            seen = set()
            used_sources = []
            for s in last_step_sources:
                key = (s.get("url") or s.get("permalink") or s.get("title") or "")
                if key and key not in seen:
                    seen.add(key)
                    used_sources.append(s)
            
        # Final fallback: if agent used tools, only show sources if we have very few
        # (likely all were used) - otherwise show nothing rather than wrong sources
        if not used_sources:
            if len(all_sources) <= 5:
                # If we have 5 or fewer sources, likely all were used
                seen = set()
                used_sources = []
                for s in all_sources:
                    key = (s.get("url") or s.get("permalink") or s.get("title") or "")
                    if key and key not in seen:
                        seen.add(key)
                        used_sources.append(s)
            else:
                # Too many sources, don't show any rather than showing wrong ones
                used_sources = []

        # Filter sources by minimum relevance score
        # Only keep sources with score > 0.3 (if score is available)
        MIN_RELEVANCE_SCORE = 0.2  # Lowered from 0.3 to avoid filtering good results

        # DEBUG: Log before filtering
        confluence_before = [s for s in used_sources if "confluence" in (s.get("url", "") or "").lower() or s.get("space") or s.get("source") == "confluence"]
        logger.info(f"DEBUG - Before filtering: {len(used_sources)} total sources, {len(confluence_before)} Confluence sources")
        if confluence_before:
            logger.info(f"DEBUG - Confluence titles before filter: {[s.get('title', 'NO_TITLE') for s in confluence_before]}")

        filtered_sources = []
        for source in used_sources:
            score = source.get("score", 1.0)  # Default to 1.0 if no score
            if isinstance(score, (int, float)) and score >= MIN_RELEVANCE_SCORE:
                filtered_sources.append(source)
            elif not isinstance(score, (int, float)):
                # Keep sources without numeric scores
                filtered_sources.append(source)
        used_sources = filtered_sources

        # DEBUG: Log after filtering
        confluence_after = [s for s in used_sources if "confluence" in (s.get("url", "") or "").lower() or s.get("space") or s.get("source") == "confluence"]
        logger.info(f"DEBUG - After filtering: {len(used_sources)} total sources, {len(confluence_after)} Confluence sources")
        if confluence_after:
            logger.info(f"DEBUG - Confluence titles after filter: {[s.get('title', 'NO_TITLE') for s in confluence_after]}")

        # Check if answer indicates no relevant information found
        # If so, don't show sources (they're not actually relevant)
        if final_answer and isinstance(final_answer, str):
            answer_lower = final_answer.lower()
            no_results_phrases = [
                "no relevant information",
                "couldn't find",
                "could not find",
                "no information found",
                "no results",
                "unable to find",
                "don't have any information",
                "no specific information",
                "no documentation",
                "no confluence pages",
                "no slack messages",
                "no docs found",
                "sorry, i couldn't",
                "sorry, i could not",
                "i don't have",
                "i couldn't locate"
            ]
            if any(phrase in answer_lower for phrase in no_results_phrases):
                used_sources = []

        # Cache results
        try:
            cache_search_results(query, cache_filters, {
                "final_answer": final_answer,
                "all_sources": used_sources,
                "tools_used": list(tools_used),
                "mode": "agentic"
            })
        except Exception as e:
            pass  # Don't fail on cache errors

        return {
            "answer": final_answer,
            "sources": used_sources,
            "tools_used": list(tools_used),
            "from_cache": False
        }

    except Exception as e:
        st.error(f"Agent execution failed: {e}")
        return {
            "answer": f"Sorry, I encountered an error: {str(e)}",
            "sources": [],
            "tools_used": [],
            "from_cache": False
        }


# =========================
# Display chat history
# =========================

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Render sources if available
        if message.get("sources"):
            render_sources(message["sources"])

        # Show cache indicator
        if message.get("from_cache"):
            st.caption("📦 From cache")

# =========================
# Chat input
# =========================

if prompt := st.chat_input("Ask a question about your PM tools and processes..."):
    # Add user message to chat
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["chat_history"].append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and searching..."):
            import time
            start_time = time.time()
            response = process_query(prompt)
            response_time_ms = int((time.time() - start_time) * 1000)

            # Log query for audit trail
            sources_used = []
            if response.get("sources"):
                for source in response["sources"]:
                    source_type = source.get("source_type", "unknown")
                    if source_type not in sources_used:
                        sources_used.append(source_type)

            session_manager.log_query(
                session_id=st.session_state.auth_session_id,
                user_id=user_info['id'],
                email=user_info['email'],
                query_text=prompt,
                sources_used=sources_used,
                response_time_ms=response_time_ms
            )

        # Display answer
        st.markdown(response["answer"])

        # Display sources
        if response["sources"]:
            # DEBUG: Log what's being passed to render_sources
            confluence_in_response = [s for s in response["sources"] if "confluence" in (s.get("url", "") or "").lower() or s.get("space") or s.get("source") == "confluence"]
            logger.info(f"DEBUG - Passing to render_sources: {len(response['sources'])} total, {len(confluence_in_response)} Confluence")
            if confluence_in_response:
                logger.info(f"DEBUG - Confluence titles in render_sources: {[s.get('title', 'NO_TITLE') for s in confluence_in_response]}")
            render_sources(response["sources"])

        # Show cache indicator
        if response["from_cache"]:
            st.caption("📦 From cache")

        # Add assistant message to chat
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "from_cache": response["from_cache"]
        })
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": response["answer"]
        })

# =========================
# Welcome message
# =========================

if len(st.session_state["messages"]) == 0:
    st.markdown("---")
    st.markdown("### 👋 Welcome! Ask me anything about:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📚 Documentation**
        - Product features
        - Configuration guides
        - Best practices
        """)

    with col2:
        st.markdown("""
        **🎫 Customer Issues**
        - Zendesk tickets
        - Common problems
        - Support patterns
        """)

    with col3:
        st.markdown("""
        **📋 Development**
        - Jira tickets
        - Roadmap items
        - Feature status
        """)

    st.markdown("---")
    st.markdown("**Example questions:**")
    st.markdown("""
    - "What is SAML authentication in Incorta?"
    - "Show me recent Zendesk tickets about performance issues"
    - "What's the status of the new dashboard feature in Jira?"
    - "How do I configure materialized views?"
    """)

# =========================
# Footer
# =========================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>Internal PM Chat • Conversational Search Assistant • Powered by AI</small>
</div>
""", unsafe_allow_html=True)
