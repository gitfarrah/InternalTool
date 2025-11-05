from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path for package imports on all platforms
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import json
from datetime import datetime

import streamlit as st

from src.handler.confluence_handler import search_confluence_optimized
from src.handler.gemini_handler import ask_gemini, answer_with_citations
from src.handler.intent_analyzer import analyze_user_intent, validate_intent
from src.handler.slack_handler import search_slack
from src.storage.cache_manager import (
    get_cached_search_results, cache_search_results,
    get_cached_intent_analysis, cache_intent_analysis,
    get_cache_manager
)
from src.agent.pm_agent import search_docs_plain, fetch_schema_details, fetch_table_data  # Integrated tools from other code
from src.handler.oauth_handler import (
    get_oauth_url,
    exchange_code_for_token,
    get_user_info,
    is_token_valid,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="Incorta AI Search Assistant", page_icon="incorta.png", layout="wide")

def _validate_env() -> List[str]:
    """Return a list of missing required environment variables."""
    missing = []
    if not st.secrets.get("SLACK_USER_TOKEN"):
        missing.append("SLACK_USER_TOKEN")
    if not st.secrets.get("CONFLUENCE_URL"):
        missing.append("CONFLUENCE_URL")
    if not st.secrets.get("CONFLUENCE_EMAIL"):
        missing.append("CONFLUENCE_EMAIL")
    if not st.secrets.get("CONFLUENCE_API_TOKEN"):
        missing.append("CONFLUENCE_API_TOKEN")
    if not st.secrets.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    # Added from other code for new sources
    if not st.secrets.get("QDRANT_URL"):
        missing.append("QDRANT_URL")
    if not st.secrets.get("INCORTA_ENV_URL"):
        missing.append("INCORTA_ENV_URL")
    if not st.secrets.get("INCORTA_TENANT"):
        missing.append("INCORTA_TENANT")
    if not st.secrets.get("INCORTA_USERNAME"):
        missing.append("INCORTA_USERNAME")
    if not st.secrets.get("INCORTA_PASSWORD"):
        missing.append("INCORTA_PASSWORD")
    return missing

def _clean_slack_text(text: str) -> str:
    import re
    
    text = re.sub(r'<#[A-Z0-9]+\|([^>]+)>', r'#\1', text)
    text = re.sub(r'<(https?://[^|>]+)\|([^>]+)>', r'\2 (\1)', text)
    text = re.sub(r'<(https?://[^>]+)>', r'\1', text)
    text = re.sub(r'<@[A-Z0-9]+\|([^>]+)>', r'@\1', text)
    text = re.sub(r'<@([A-Z0-9]+)>', r'@unknown_user', text)
    # Handle special mentions
    text = text.replace('<!channel>', '@channel')
    text = text.replace('<!here>', '@here')
    text = text.replace('<!everyone>', '@everyone')
    
    return text

def _format_context(slack_messages: List[dict], confluence_pages: List[dict], docs_results: List[dict], zendesk_results: dict, jira_results: dict) -> str:
    parts: List[str] = []

    if slack_messages:
        parts.append("=== Slack Messages ===")
        for i, m in enumerate(slack_messages, start=1):
            timestamp = m.get('date', 'Unknown date')
            meta = f"[# {m.get('channel','?')} | @{m.get('username','?')} | {timestamp}]"
            clean_text = _clean_slack_text(m.get('text','').strip())
            line = f"{i}. {meta}\n{clean_text}\nSource: {m.get('permalink','')}\n"
            parts.append(line)

    if confluence_pages:
        parts.append("=== Confluence Pages ===")
        for i, p in enumerate(confluence_pages, start=1):
            meta = f"[{p.get('space','?')} | last modified: {p.get('last_modified','?')}]"
            excerpt = (p.get('excerpt') or '').strip()
            line = f"{i}. {p.get('title','Untitled')} {meta}\nExcerpt: {excerpt}\nSource: {p.get('url','')}\n"
            parts.append(line)

    # Added from other code
    if docs_results:
        parts.append("=== Knowledge Base (Docs) ===")
        for i, d in enumerate(docs_results, start=1):
            meta = f"[Score: {d.get('score', 0.0):.2f}]"
            text = (d.get('text') or '').strip()
            line = f"{i}. {d.get('title','Untitled')} {meta}\nExcerpt: {text[:300]}\nSource: {d.get('url','')}\n"
            parts.append(line)

    if zendesk_results and 'data' in zendesk_results:
        parts.append("=== Zendesk Tickets ===")
        data = zendesk_results['data']
        columns = data.get('columns', [])
        rows = data.get('rows', [])
        if columns and rows:
            table_str = "Columns: " + ", ".join(columns) + "\n"
            for i, row in enumerate(rows, 1):
                table_str += f"Row {i}: " + str(row) + "\n"
            parts.append(table_str)

    if jira_results and 'data' in jira_results:
        parts.append("=== Jira Issues ===")
        data = jira_results['data']
        columns = data.get('columns', [])
        rows = data.get('rows', [])
        if columns and rows:
            table_str = "Columns: " + ", ".join(columns) + "\n"
            for i, row in enumerate(rows, 1):
                table_str += f"Row {i}: " + str(row) + "\n"
            parts.append(table_str)

    return "\n".join(parts)


def _group_slack_by_channel_date(slack_results: List[dict]) -> dict:
    grouped = {}
    for m in slack_results or []:
        channel = f"#{(m.get('channel') or 'unknown').lstrip('#')}"
        date_str = m.get('date') or m.get('ts') or 'Unknown date'
        date_key = date_str.split(' ')[0] if isinstance(date_str, str) else str(date_str)
        text = (m.get('text') or '').strip().replace('\n', ' ')
        if len(text) > 200:
            text = text[:200] + '...'
        grouped.setdefault(channel, {}).setdefault(date_key, []).append(text)
    for channel in grouped:
        grouped[channel] = dict(sorted(grouped[channel].items()))
    return dict(sorted(grouped.items()))


def _group_confluence_by_space(pages: List[dict]) -> dict:
    grouped = {"Confluence": {}}
    buckets = {}
    for p in pages or []:
        parent = p.get('space') or 'General'
        child = p.get('title') or 'Untitled'
        buckets.setdefault(parent, set()).add(child)
    for parent, children in buckets.items():
        grouped["Confluence"][parent] = {
            "title": parent,
            "children": sorted(list(children))
        }
    return grouped


def _format_grouped_response(summary: str, slack_results: List[dict], conf_results: List[dict]) -> str:
    slack_grouped = _group_slack_by_channel_date(slack_results)
    conf_grouped = _group_confluence_by_space(conf_results)

    lines = []
    if summary and summary.strip():
        lines.append(f"**Summary:** {summary.strip()}")
        lines.append("")

    lines.append("**Slack Messages:**")
    lines.append("")
    if slack_grouped:
        idx = 1
        for channel, dates in slack_grouped.items():
            lines.append(f"{idx}. **Channel: {channel}**")
            for date_key, msgs in dates.items():
                preview = ", ".join(msgs[:6])
                lines.append(f"   - {date_key}: {preview}")
            lines.append("")
            idx += 1
    else:
        lines.append("(no Slack messages found)")
        lines.append("")

    lines.append("**Confluence Pages:**")
    lines.append("")
    if conf_grouped.get("Confluence"):
        lines.append("3. **Confluence**")
        for parent, info in conf_grouped["Confluence"].items():
            lines.append(f"   - Parent Wiki: {parent}")
            for child in info.get("children", [])[:10]:
                lines.append(f"     - Child: {child}")
        lines.append("")
    else:
        lines.append("(no Confluence pages found)")

    return "\n".join(lines)


def _render_sources(slack_messages: List[dict], confluence_pages: List[dict], docs_results: List[dict], zendesk_results: dict, jira_results: dict) -> None:
    with st.expander(f"Slack Messages ({len(slack_messages)} found)", expanded=False):
        if not slack_messages:
            st.info("No Slack results found.")
        else:
            for idx, m in enumerate(slack_messages, 1):
                channel = m.get('channel', 'Unknown')
                username = m.get('username', 'Unknown')
                timestamp = m.get('date', m.get('ts', ''))
                text = _clean_slack_text(m.get('text', '').strip())
                permalink = m.get('permalink', '')
                relevance_score = m.get('relevance_score', 0.0)
                import html
                text_escaped = html.escape(text)
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4A90E2;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #1f1f1f;">#{channel}</span>
                        <div style="display: flex; gap: 10px; align-items: center;">
                            <span style="color: #666; font-size: 0.9em;">{timestamp}</span>
                            <span style="color: #888; font-size: 0.9em;">Score: {relevance_score:.2f}</span>
                        </div>
                    </div>
                    <div style="color: #4A90E2; font-size: 0.9em; margin-bottom: 10px;">@{username}</div>
                    <div style="color: #1f1f1f; line-height: 1.6; margin-bottom: 10px; white-space: pre-wrap;">{text_escaped}</div>
                    <a href="{permalink}" target="_blank" style="color: #4A90E2; text-decoration: none; font-size: 0.9em;">View in Slack</a>
                </div>
                """, unsafe_allow_html=True)

    with st.expander(f"Confluence Pages ({len(confluence_pages)} found)", expanded=False):
        if not confluence_pages:
            st.info("No Confluence results found.")
        else:
            for idx, p in enumerate(confluence_pages, 1):
                title = p.get('title', 'Untitled')
                space = p.get('space', 'Unknown')
                last_modified = p.get('last_modified', 'Unknown')
                excerpt = (p.get('excerpt') or '').strip()
                url = p.get('url', '')
                score = p.get('score', 0.0)
                import re
                cleaned_excerpt = re.sub(r'@@@hl@@@(.*?)@@@endhl@@@', r'\1', excerpt)
                cleaned_excerpt = re.sub(r'<[^>]+>', '', cleaned_excerpt)
                cleaned_excerpt = cleaned_excerpt.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                if len(cleaned_excerpt) > 300:
                    cleaned_excerpt = cleaned_excerpt[:300] + '...'
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6B46C1;">
                    <div style="font-weight: 600; color: #1f1f1f; font-size: 1.1em; margin-bottom: 8px;">{title}</div>
                    <div style="display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em;">
                        <span style="color: #6B46C1; font-weight: 500;">{space}</span>
                        <span style="color: #666;">{last_modified}</span>
                        <span style="color: #888;">Score: {score:.2f}</span>
                    </div>
                    <div style="color: #444; line-height: 1.6; margin-bottom: 10px; font-style: italic;">{cleaned_excerpt}</div>
                    <a href="{url}" target="_blank" style="color: #6B46C1; text-decoration: none; font-size: 0.9em;">Open Page</a>
                </div>
                """, unsafe_allow_html=True)

    # Added render for new sources
    with st.expander(f"Knowledge Base Docs ({len(docs_results)} found)", expanded=False):
        if not docs_results:
            st.info("No docs results found.")
        else:
            for idx, d in enumerate(docs_results, 1):
                title = d.get('title', 'Untitled')
                url = d.get('url', '')
                text = (d.get('text') or '').strip()
                score = d.get('score', 0.0)
                if len(text) > 300:
                    text = text[:300] + '...'
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28a745;">
                    <div style="font-weight: 600; color: #1f1f1f; font-size: 1.1em; margin-bottom: 8px;">{title}</div>
                    <div style="color: #666; font-size: 0.9em; margin-bottom: 10px;">Score: {score:.2f}</div>
                    <div style="color: #444; line-height: 1.6; margin-bottom: 10px;">{text}</div>
                    <a href="{url}" target="_blank" style="color: #28a745; text-decoration: none; font-size: 0.9em;">Open Document</a>
                </div>
                """, unsafe_allow_html=True)

    with st.expander(f"Zendesk Tickets (Results found)", expanded=False):
        if not zendesk_results or 'data' not in zendesk_results:
            st.info("No Zendesk results found.")
        else:
            data = zendesk_results['data']
            columns = data.get('columns', [])
            rows = data.get('rows', [])
            if columns and rows:
                st.table({"Columns": columns, "Rows": rows})

    with st.expander(f"Jira Issues (Results found)", expanded=False):
        if not jira_results or 'data' not in jira_results:
            st.info("No Jira results found.")
        else:
            data = jira_results['data']
            columns = data.get('columns', [])
            rows = data.get('rows', [])
            if columns and rows:
                st.table({"Columns": columns, "Rows": rows})

def generate_sql_for_schema(schema_name: str, query: str) -> str:
    schema_details = fetch_schema_details(schema_name)
    if 'error' in schema_details:
        return ""
    
    schema_json = json.dumps(schema_details, indent=2)
    
    sql_prompt = f"""
    You are a SQL query generator for Incorta schemas.
    Schema details: {schema_json}
    
    User query: {query}
    
    Generate a valid Spark SQL query to answer the query.
    Use the schema tables and columns appropriately.
    Return ONLY the SQL query, no explanations.
    """
    
    sql_response = ask_gemini(sql_prompt, "")
    return sql_response.strip()

def _is_slack_authenticated_cached() -> bool:
    """Cache Slack token validity to avoid repeated auth_test calls on every render."""
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

def main() -> None:
    st.title("Incorta AI Search Assistant Tool")
    st.caption("Searches Slack, Confluence, Docs, Zendesk, Jira, then asks Gemini to synthesize an answer with sources.")

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
                # Ensure chat state exists
                if "chat_messages" not in st.session_state:
                    st.session_state["chat_messages"] = [
                        {"role": "assistant", "content": "Hi! Ask me anything. I'll search Slack, Confluence, Docs, Zendesk, Jira, then summarize with sources."}
                    ]
                # Don't rerun - just continue with normal flow
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
                    # Ensure chat state exists
                    if "chat_messages" not in st.session_state:
                        st.session_state["chat_messages"] = [
                            {"role": "assistant", "content": "Hi! Ask me anything. I'll search Slack, Confluence, Docs, Zendesk, Jira, then summarize with sources."}
                        ]
                    # Continue without rerun
                else:
                    # Process the OAuth code
                    logger.info("Processing OAuth code...")
                    token = exchange_code_for_token(oauth_code)
                    user_info = get_user_info(token)
                    st.session_state["slack_token"] = token
                    st.session_state["slack_user_name"] = user_info.get("name", "User")
                    st.session_state["slack_user_display_name"] = user_info.get("display_name") or user_info.get("name", "User")
                    st.session_state["slack_user_id"] = user_info.get("id", "")
                    st.session_state["processed_oauth_code"] = oauth_code  # Mark code as processed
                    
                    # Ensure chat_messages exists after OAuth
                    if "chat_messages" not in st.session_state:
                        st.session_state["chat_messages"] = [
                            {"role": "assistant", "content": "Hi! Ask me anything. I'll search Slack, Confluence, Docs, Zendesk, Jira, then summarize with sources."}
                        ]
                        logger.info("Reinitialized chat_messages after OAuth")
                    
                    logger.info(f"Slack OAuth successful for user: {st.session_state['slack_user_display_name']}")
                    logger.info(f"Chat messages after OAuth: {len(st.session_state.get('chat_messages', []))} messages")
                    
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
                # Don't stop - allow user to continue
            else:
                st.error(f"❌ Authentication failed: {msg}")
                logger.error(f"OAuth authentication failed: {msg}", exc_info=True)
                try:
                    st.query_params.clear()
                except Exception:
                    pass

    # Auth gate: require Slack token to search Slack; allow rest to run but show connect if missing
    if "slack_token" not in st.session_state or not _is_slack_authenticated_cached():
        with st.sidebar:
            st.subheader("Slack Authentication")
            try:
                oauth_url = get_oauth_url()
                st.link_button("🔗 Connect to Slack", oauth_url, use_container_width=True)
                st.caption("You'll be redirected to Slack to authorize, then back here.")
            except Exception:
                st.info("To enable Slack login, set SLACK_CLIENT_ID, SLACK_CLIENT_SECRET, and SLACK_REDIRECT_URI in st.secrets or environment.")

    # Add logout in sidebar if logged in
    with st.sidebar:
        if st.session_state.get("slack_token"):
            display_name = st.session_state.get("slack_user_display_name") or st.session_state.get("slack_user_name") or "Signed in"
            st.markdown(f"**Slack:** {display_name}")
            if st.button("Logout"):
                for k in ["slack_token", "slack_user_name", "slack_user_display_name", "slack_user_id"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

    missing = _validate_env()
    if missing:
        # Don't require SLACK_USER_TOKEN anymore for OAuth flow
        if "SLACK_USER_TOKEN" in missing:
            missing.remove("SLACK_USER_TOKEN")
        if missing:
            st.warning("Missing environment variables: " + ", ".join(missing) + ". Set them to enable full functionality.")

    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []

    with st.sidebar:
        st.markdown("<h1 style=\"margin-top:0; margin-bottom:0.25rem; font-size:1.6rem;\">Search Options</h1>", unsafe_allow_html=True)
        
        st.divider()
        
        st.subheader("Filters")
        channel_hint = st.text_input("Slack channel", value=st.session_state.get("channel_hint", ""), 
                                     key="channel_hint", placeholder="e.g., general, engineering")
        space_hint = st.text_input("Confluence space", value=st.session_state.get("space_hint", ""), 
                                   key="space_hint", placeholder="e.g., PROJ, DOCS")

        st.divider()
        auto_refresh = True
        
    # Initialize chat state - must be done early, before any OAuth redirects
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Hi! Ask me anything. I'll search Slack, Confluence, Docs, Zendesk, Jira, then summarize with sources."}
        ]
        logger.info("Initialized chat_messages in session state")
    
    if "context" not in st.session_state:
        st.session_state["context"] = None
    if "slack_results" not in st.session_state:
        st.session_state["slack_results"] = []
    if "conf_results" not in st.session_state:
        st.session_state["conf_results"] = []
    if "filters" not in st.session_state:
        st.session_state["filters"] = {}
    
    logger.debug(f"Chat state initialized. Messages count: {len(st.session_state.get('chat_messages', []))}")

    # Display chat messages - handle errors gracefully
    try:
        chat_messages = st.session_state.get("chat_messages", [])
        if not chat_messages:
            # Reinitialize if empty
            st.session_state["chat_messages"] = [
                {"role": "assistant", "content": "Hi! Ask me anything. I'll search Slack, Confluence, Docs, Zendesk, Jira, then summarize with sources."}
            ]
            chat_messages = st.session_state["chat_messages"]
        
        for msg in chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg.get("content", ""))
    except Exception as e:
        logger.error(f"Error displaying chat messages: {e}", exc_info=True)
        # Reinitialize chat if corrupted
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Hi! Ask me anything. I'll search Slack, Confluence, Docs, Zendesk, Jira, then summarize with sources."}
        ]
        with st.chat_message("assistant"):
            st.markdown("Hi! Ask me anything. I'll search Slack, Confluence, Docs, Zendesk, Jira, then summarize with sources.")

    user_input = st.chat_input("Type your question")

    if user_input:
        logger.info(f"User input received: {user_input[:100]}...")
        try:
            # Ensure chat_messages exists (might have been lost during OAuth redirect)
            if "chat_messages" not in st.session_state:
                st.session_state["chat_messages"] = [
                    {"role": "assistant", "content": "Hi! Ask me anything. I'll search Slack, Confluence, Docs, Zendesk, Jira, then summarize with sources."}
                ]
                logger.info("Reinitialized chat_messages after OAuth redirect")
            
            # Add user message to chat immediately
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})
            logger.info(f"Added user message. Total messages: {len(st.session_state['chat_messages'])}")
            
            with st.chat_message("user"):
                st.write(user_input)
        except Exception as e:
            logger.error(f"Error adding user message to chat: {e}", exc_info=True)
            st.error(f"Error processing your message: {e}")
            # Instead of stopping, show error and continue
            if "chat_messages" not in st.session_state:
                st.session_state["chat_messages"] = []
            st.session_state["chat_messages"].append({"role": "assistant", "content": f"Error: {str(e)}"})

        try:
            with st.chat_message("assistant"):
                current_filters = {
                    "channel_hint": (channel_hint or "").strip(),
                    "space_hint": (space_hint or "").strip(),
                }
                stored_filters = st.session_state.get("filters", {})
                filters_changed = current_filters != stored_filters

                intent_type = ""
                if "intent_data" in st.session_state:
                    intent_type = st.session_state["intent_data"].get("intent", "")
                
                reuse_context = (not auto_refresh) and (not filters_changed) and bool(st.session_state.get("context")) and (intent_type != "latest_message")

                if reuse_context:
                    context = st.session_state.get("context") or ""
                    slack_results = st.session_state.get("slack_results") or []
                    conf_results = st.session_state.get("conf_results") or []
                    docs_results = st.session_state.get("docs_results") or []
                    zendesk_results = st.session_state.get("zendesk_results") or {}
                    jira_results = st.session_state.get("jira_results") or {}
                else:
                    with st.spinner("Analyzing query and retrieving sources..."):
                        cache_filters = {
                            "channel_hint": (channel_hint or "").strip(),
                            "space_hint": (space_hint or "").strip(),
                        }
                        
                        intent_data = get_cached_intent_analysis(user_input, cache_filters)
                        
                        if not intent_data:
                            intent_data = analyze_user_intent(user_input)
                            intent_data = validate_intent(intent_data)
                            cache_intent_analysis(user_input, cache_filters, intent_data)
                        
                        if channel_hint and channel_hint.strip():
                            intent_data["slack_params"]["channels"] = channel_hint.strip()
                        if space_hint and space_hint.strip():
                            intent_data["confluence_params"]["spaces"] = space_hint.strip()

                        logger.info(f"Intent analysis result: {intent_data}")
                        
                        slack_results: List[dict] = []
                        conf_results: List[dict] = []
                        docs_results: List[dict] = []
                        zendesk_results: dict = {}
                        jira_results: dict = {}

                        data_sources = intent_data.get("data_sources", ["slack", "confluence"])

                        # Enforce Slack login: if no valid token, do not search Slack
                        if ("slack_token" not in st.session_state) or (not _is_slack_authenticated_cached()):
                            if "slack" in data_sources:
                                data_sources = [s for s in data_sources if s != "slack"]
                                st.info("Sign in with Slack to search Slack (private and public channels).")

                        with ThreadPoolExecutor(max_workers=5) as pool:
                            futures = {}

                            if "slack" in data_sources:
                                futures["slack"] = pool.submit(
                                    search_slack,
                                    user_input,
                                    intent_data,
                                    15,
                                    st.session_state.get("slack_token")
                                )
                            
                            if "confluence" in data_sources:
                                futures["confluence"] = pool.submit(
                                    search_confluence_optimized,
                                    intent_data,
                                    user_input
                                )
                            
                            # Knowledge base (Docs) search: enable for doc-style queries even if not explicitly requested
                            def _should_search_docs(query: str) -> bool:
                                q = (query or "").lower()
                                doc_terms = [
                                    "how to", "install", "installation", "setup", "configure",
                                    "documentation", "doc", "guide", "on prem", "on-prem"
                                ]
                                return any(t in q for t in doc_terms)

                            if any(s in data_sources for s in ["docs", "knowledge_base"]) or _should_search_docs(user_input):
                                futures["docs"] = pool.submit(search_docs_plain, user_input, 5)
                            
                            if "zendesk" in data_sources:
                                zendesk_schema = fetch_schema_details("ZendeskTickets")
                                zendesk_sql = generate_sql_for_schema("ZendeskTickets", user_input)
                                if zendesk_sql:
                                    futures["zendesk"] = pool.submit(
                                        fetch_table_data,
                                        zendesk_sql
                                    )
                            
                            if "jira" in data_sources:
                                jira_schema = fetch_schema_details("Jira_F")
                                jira_sql = generate_sql_for_schema("Jira_F", user_input)
                                if jira_sql:
                                    futures["jira"] = pool.submit(
                                        fetch_table_data,
                                        jira_sql
                                    )
                            
                            for source, future in futures.items():
                                try:
                                    if source == "slack":
                                        slack_results = future.result(timeout=75)
                                    elif source == "confluence":
                                        conf_results = future.result(timeout=60)
                                    elif source == "docs":
                                        docs_results = future.result(timeout=30)
                                    elif source == "zendesk":
                                        zendesk_results = future.result(timeout=60)
                                    elif source == "jira":
                                        jira_results = future.result(timeout=60)
                                except Exception as e:
                                    logger.error(f"{source.title()} search failed: {e}")
                                    if "timeout" in str(e).lower():
                                        st.warning(f"⏱️ {source.title()} search timed out.")
                                    else:
                                        st.warning(f"⚠️ {source.title()} search encountered an issue: {str(e)}")

                        if not slack_results and not conf_results and not docs_results and not zendesk_results and not jira_results:
                            nores = "No results found in any source. Try adjusting your query or filters."
                            st.write(nores)
                            st.session_state["chat_messages"].append({"role": "assistant", "content": nores})
                            return
                        
                        context = _format_context(slack_results, conf_results, docs_results, zendesk_results, jira_results)
                        st.session_state["context"] = context
                        st.session_state["slack_results"] = slack_results
                        st.session_state["conf_results"] = conf_results
                        st.session_state["docs_results"] = docs_results
                        st.session_state["zendesk_results"] = zendesk_results
                        st.session_state["jira_results"] = jira_results
                        st.session_state["filters"] = current_filters
                        st.session_state["intent_data"] = intent_data

                preface = ("Previous conversation context (use for continuity):\n" + "\n".join([f"{prefix} {m['content']}" for m in st.session_state["chat_messages"][-6:] if (prefix := "User:" if m["role"] == "user" else "Assistant:")]) + "\n\n") if st.session_state["chat_messages"][-6:] else ""

                with st.spinner("Thinking..."):
                    # Generate structured answer with citations
                    try:
                        # Prepare passages for citation-based answer
                        passages = []
                        
                        # Add Slack passages
                        for msg in (slack_results or [])[:10]:
                            passages.append({
                                "source": "slack",
                                "text": _clean_slack_text(msg.get('text', '')),
                                "title": f"#{msg.get('channel', 'unknown')} - @{msg.get('username', 'unknown')}",
                                "url": msg.get('permalink', ''),
                                "timestamp": msg.get('date', '')
                            })
                        
                        # Add Confluence passages
                        for page in (conf_results or [])[:10]:
                            passages.append({
                                "source": "confluence",
                                "text": page.get('excerpt', ''),
                                "title": page.get('title', 'Untitled'),
                                "url": page.get('url', ''),
                                "space": page.get('space', '')
                            })
                        
                        # Add Knowledge Base passages
                        for doc in (docs_results or [])[:10]:
                            # search_docs_plain returns formatted dicts with title, url, text, score, source
                            # Handle both Qdrant ScoredPoint format and pre-formatted dict format
                            if hasattr(doc, 'payload'):
                                # Qdrant ScoredPoint object
                                payload = doc.payload
                                text = str(payload.get('text', '') or payload.get('content', '') or '').strip()
                                title = str(payload.get('title', 'Untitled') or 'Untitled')
                                url = str(payload.get('url', '') or '')
                                score = doc.score if hasattr(doc, 'score') else 0.0
                            elif isinstance(doc, dict):
                                # Pre-formatted dict from search_docs_plain
                                text = str(doc.get('text', '') or doc.get('content', '') or '').strip()
                                title = str(doc.get('title', 'Untitled') or 'Untitled')
                                url = str(doc.get('url', '') or '')
                                score = doc.get('score', 0.0)
                            else:
                                text = ''
                                title = 'Untitled'
                                url = ''
                                score = 0.0
                            
                            # Only add passage if it has text content (minimum 10 chars to avoid empty snippets)
                            if text and len(text.strip()) >= 10:
                                passages.append({
                                    "source": "knowledge_base",
                                    "text": text,
                                    "title": title,
                                    "url": url,
                                    "score": score
                                })
                            else:
                                logger.debug(f"Skipping Knowledge Base doc '{title}' - no text content (length: {len(text) if text else 0})")
                        
                        # Log passage count for debugging
                        logger.info(f"Prepared {len(passages)} passages for answer generation")
                        logger.info(f"Passage sources: {[p.get('source', 'unknown') for p in passages[:5]]}")
                        
                        # Use answer_with_citations for structured response with source mentions
                        if passages:
                            answer_result = answer_with_citations(user_input, passages)
                            
                            if answer_result and answer_result.get("exists"):
                                summary_text = answer_result.get("answer", "")
                                logger.info("Generated answer with citations successfully")
                            else:
                                # If answer_with_citations failed, try fallback with better prompt
                                logger.warning(f"answer_with_citations returned exists=false. Attempting fallback...")
                                context_for_summary = _format_context(slack_results, conf_results, docs_results, zendesk_results, jira_results)
                                summary_prompt = (
                                    "Based on the provided context, write a comprehensive answer to the user's question. "
                                    "Begin by mentioning the source (e.g., 'According to the documentation...' or 'Based on Slack discussions...'). "
                                    "For installation or step-by-step queries, provide clear numbered steps with headings.\n\n"
                                    f"User question: {user_input}\n\nContext:\n{context_for_summary}"
                                )
                                summary_text = ask_gemini(summary_prompt, "") or "I couldn't find relevant information to answer your question. Please try rephrasing or adjusting your search terms."
                        else:
                            logger.warning("No passages prepared - cannot generate answer")
                            summary_text = "I couldn't find relevant information to answer your question. Please try rephrasing or adjusting your search terms."
                    except Exception as e:
                        logger.error(f"Error generating structured answer: {e}", exc_info=True)
                        # Fallback to simple summary
                        try:
                            context_for_summary = context or _format_context(slack_results, conf_results, docs_results, zendesk_results, jira_results)
                            summary_prompt = (
                                "Write a concise answer answering the user's question directly. "
                                "Begin by mentioning the source (e.g., 'According to the documentation...' or 'Based on Slack discussions...'). "
                                "For installation or step-by-step queries, provide clear numbered steps with headings.\n\n"
                                f"User question: {user_input}\n\nContext:\n{context_for_summary}"
                            )
                            summary_text = ask_gemini(summary_prompt, "") or ""
                        except Exception:
                            summary_text = "An error occurred while generating the answer. Please try again."

                grouped_output = _format_grouped_response(
                    summary_text,
                    slack_results,
                    conf_results,
                )

                st.markdown(grouped_output)

                # Agent Trace: log to terminal instead of UI
                try:
                    strategies = [r.get('strategy') for r in (slack_results or []) if r.get('strategy')]
                    strat_counts = {}
                    for s in strategies:
                        strat_counts[s] = strat_counts.get(s, 0) + 1
                    channels = sorted({r.get('channel') for r in (slack_results or []) if r.get('channel')})
                    conf_top = ", ".join([c.get('title','') for c in (conf_results or [])[:5]])
                    docs_top = ", ".join([d.get('title','') for d in (docs_results or [])[:5]])
                    if strat_counts:
                        logger.info("Agent Trace: Slack strategies used: " + ", ".join([f"{k} x{v}" for k,v in strat_counts.items()]))
                    if channels:
                        logger.info("Agent Trace: Slack channels searched: " + ", ".join([f"#{c}" for c in channels]))
                    logger.info(f"Agent Trace: Confluence results: {len(conf_results or [])}; Top: {conf_top}")
                    logger.info(f"Agent Trace: Docs results: {len(docs_results or [])}; Top: {docs_top}")
                except Exception:
                    pass

                if slack_results or conf_results or docs_results or zendesk_results or jira_results:
                    _render_sources(slack_results, conf_results, docs_results, zendesk_results, jira_results)
                
                st.session_state["chat_messages"].append({"role": "assistant", "content": grouped_output})

                new_entry = {
                    "question": user_input.strip(),
                    "channel_hint": (channel_hint or "").strip() or None,
                    "space_hint": (space_hint or "").strip() or None,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                if not st.session_state["search_history"] or st.session_state["search_history"][0].get("question") != new_entry["question"]:
                    st.session_state["search_history"].insert(0, new_entry)
                    if len(st.session_state["search_history"]) > 50:
                        st.session_state["search_history"] = st.session_state["search_history"][:50]
        except Exception as e:
            error_msg = f"An error occurred while processing your query: {str(e)}"
            logger.error(f"Error in chat processing: {e}", exc_info=True)
            st.error(error_msg)
            st.session_state["chat_messages"].append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()