"""
Product Manager Agent for LangChain ReAct Agent Executor.

Provides agent tools and setup for searching across multiple knowledge sources
including Confluence, Slack, Docs, Zendesk, and Jira.
"""

from typing import List, Optional, Set
import logging
import os
import re
from html import unescape
import requests
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from ..handler.confluence_handler import search_confluence, search_confluence_optimized as confluence_optimized_handler
from ..handler.slack_handler import search_slack
from ..handler.intent_analyzer import analyze_user_intent
from ..storage.cache_manager import get_cached_search_results, cache_search_results

logger = logging.getLogger(__name__)


def _get_secret_or_env(name: str, default: str = "") -> str:
    """
    Get value from Streamlit secrets first, then fall back to environment variables.
    
    Args:
        name: Name of the secret/environment variable
        default: Default value if not found
    
    Returns:
        Value from Streamlit secrets or environment variable
    """
    # Try Streamlit secrets first (if available and in Streamlit context)
    try:
        import streamlit as st
        # Check if we're in a Streamlit context and if the secret exists
        if hasattr(st, 'secrets') and name in st.secrets:
            return st.secrets.get(name, default)
    except Exception:
        # Streamlit not available, not in Streamlit context, or any error
        # Fall through to environment variables
        pass
    
    # Fall back to environment variables (loaded from .env by load_dotenv())
    return os.getenv(name, default)


# =========================
# Agent Tools
# =========================

@tool(
    "search_confluence_pages",
    description="""Search for pages in Confluence with optimized query processing.

    Args:
        query: Search query
        max_results: Maximum number of results to return
        space_filter: Specific space to search (None for all spaces)

    Returns:
        List of page dictionaries with metadata
        
    Note: Use this tool when the query mentions "confluence", "wiki", "page", or "internal doc"."""
)
def search_confluence_tool(
    query: str,
    max_results: int = 10,
    space_filter: Optional[str] = None
) -> List[dict]:
    """Optimized Confluence search with query preprocessing."""
    if not query or not query.strip():
        return []

    # Use stopwords filtering for better relevance
    query_words = set(query.lower().split())
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "is", "are", "was", "were", "be", "been", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "what", "when", "where", "why", "how", "who", "which",
        "updates", "about", "on"
    }
    distinct_words = [word for word in query_words if word not in stop_words and len(word) > 2]

    # Build search query prioritizing distinct words
    if distinct_words:
        search_query = " ".join(distinct_words)
    else:
        search_query = query

    logger.info(f"Optimized Confluence search query: {search_query}, space_filter: {space_filter}")

    return search_confluence(search_query, max_results, space_filter)


@tool(
    "search_slack_messages",
    description="""Search for messages in Slack across all channels (including private channels the user has access to).

    Args:
        query: Search query
        max_results: Maximum number of results to return
        channel_filter: Specific channel to search (None for all channels)
        max_age_hours: Maximum age of messages in hours (default 0 = all history)

    Returns:
        List of message dictionaries with metadata
        
    Note: This tool searches both public and private channels. Private channels are only included if the user is a member."""
)
def search_slack_tool(
    query: str,
    max_results: int = 10,
    channel_filter: Optional[str] = None,
    max_age_hours: int = 0  # 0 = all history
) -> List[dict]:
    """Search Slack messages with intent-aware processing."""
    if not query or not query.strip():
        return []

    # Use stopwords filtering for better relevance
    query_words = set(query.lower().split())
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "is", "are", "was", "were", "be", "been", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "can", "what", "when", "where", "why", "how", "who", "which",
        "updates", "about", "on"
    }
    distinct_words = [word for word in query_words if word not in stop_words and len(word) > 2]

    # Create intent data for the search system with filtered keywords
    if distinct_words:
        keywords = distinct_words
    else:
        keywords = [term for term in query.lower().split() if len(term) > 2]

    priority_terms = keywords[:3]  # Top 3 terms as priority

    intent_data = {
        "slack_params": {
            "keywords": keywords,
            "priority_terms": priority_terms,
            "channels": channel_filter if channel_filter else "all",
            "time_range": "all",
            "limit": max_results
        },
        "search_strategy": "fuzzy_match"
    }

    logger.info(f"Optimized Slack search with keywords: {keywords}, channel_filter: {channel_filter}")

    # Use the search_slack function with user_token from session state
    import streamlit as st
    user_token = st.session_state.get("slack_token") if hasattr(st, 'session_state') else None

    # DEBUG: Log token status
    token_status = "present" if user_token else "MISSING"
    token_prefix = user_token[:10] if user_token else "None"
    logger.warning(f"🔐 Slack token status: {token_status} (prefix: {token_prefix}...)")

    results = search_slack(query, intent_data, max_results, user_token)

    # Convert to legacy format for compatibility
    legacy_results = []
    for result in results:
        legacy_results.append({
            "text": result.get("text", ""),
            "username": result.get("username", "Unknown"),
            "channel": result.get("channel", "unknown"),
            "channel_id": result.get("channel_id"),
            "ts": result.get("ts", ""),
            "date": result.get("date", ""),
            "permalink": result.get("permalink", ""),
            "relevance_score": result.get("relevance_score"),
            "score": result.get("relevance_score", result.get("score", 0.0)),
            "is_private": result.get("is_private", False),
            "strategy": result.get("strategy"),
            "source": "slack"
        })

    return legacy_results


@tool(
    "search_docs",
    description="""Search documents in Qdrant collection (Incorta Community, Docs & Support).

    Args:
        query: Search query
        limit: Number of results to return

    Returns:
        List of search results with metadata
        
    Note: Use this tool when the query mentions "docs", "documentation", "knowledge base", "kb", "guide", or "tutorial"."""
)
def search_docs(query: str, limit: int = 5) -> List[dict]:
    """Search documents in Qdrant collection with optimized relevance scoring."""
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    if not query or not query.strip():
        return []

    try:
        # Support env vars and Streamlit secrets
        qdrant_url = os.getenv("QDRANT_URL") or os.getenv("QDRANT_HOST")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        if not qdrant_url:
            try:
                import streamlit as st
                qdrant_url = st.secrets.get("QDRANT_URL", "") or st.secrets.get("QDRANT_HOST", "")
                qdrant_api_key = st.secrets.get("QDRANT_API_KEY", qdrant_api_key)
            except Exception:
                pass
        if not qdrant_url:
            logger.warning("QDRANT_URL not set; knowledge base search disabled")
            return []

        # Use stopwords filtering for better relevance
        query_words = set(query.lower().split())
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
            "with", "by", "is", "are", "was", "were", "be", "been", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may",
            "might", "can", "what", "when", "where", "why", "how", "who", "which",
            "updates", "about", "on"
        }
        distinct_words = [word for word in query_words if word not in stop_words and len(word) > 2]

        # Build optimized search query
        if distinct_words:
            search_query = " ".join(distinct_words)
        else:
            search_query = query

        logger.info(f"Optimized docs search query: {search_query}")

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
        query_vector = embedding_model.encode([search_query])[0]

        def _discover_doc_collections(qdrant_client: QdrantClient) -> List[str]:
            """Discover collections that likely hold documentation-like content."""
            try:
                collections_response = qdrant_client.get_collections()
                candidate_names = []
                for coll in collections_response.collections:
                    name = getattr(coll, "name", "")
                    lowered = (name or "").lower()
                    if any(keyword in lowered for keyword in ("doc", "kb", "knowledge", "guide", "community", "support")):
                        candidate_names.append(name)
                if candidate_names:
                    return candidate_names
            except Exception as discovery_error:
                logger.warning(f"Failed to auto-discover doc collections: {discovery_error}")
            # Fallback to default collection
            return ["docs"]

        def _strip_markup(raw_text: str) -> str:
            if not raw_text:
                return ""
            text = re.sub(r'@@@hl@@@(.*?)@@@endhl@@@', r'\1', raw_text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def _extract_procedural_steps(text: str, max_steps: int = 6) -> List[str]:
            if not text:
                return []
            steps: List[str] = []
            for line in text.splitlines():
                clean_line = line.strip()
                if not clean_line:
                    continue
                match = re.match(r'^(?:step\s*\d+|(?:\d+[\).\]])|[-*•])\s*(.+)', clean_line, flags=re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if candidate:
                        steps.append(candidate)
                elif re.match(r'^\d+\s+-\s+(.+)', clean_line):
                    candidate = re.sub(r'^\d+\s+-\s+', '', clean_line).strip()
                    if candidate:
                        steps.append(candidate)
                if len(steps) >= max_steps:
                    break
            if steps:
                return steps[:max_steps]

            # Fallback: take meaningful sentences as pseudo-steps
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                candidate = sentence.strip()
                if len(candidate) < 25:
                    continue
                steps.append(candidate)
                if len(steps) >= max_steps:
                    break
            return steps[:max_steps]

        target_collections = _discover_doc_collections(client)
        logger.info(f"Searching collections for docs: {target_collections}")

        combined_results: List[dict] = []
        seen_urls: Set[str] = set()
        per_collection_limit = max(limit, 5)

        for collection_name in target_collections:
            try:
                search_hits = client.search(
                    collection_name=collection_name,
                    query_vector=("content_vector", query_vector),
                    limit=per_collection_limit,
                    with_payload=True
                )
            except Exception as search_error:
                logger.warning(f"Docs search failed for collection '{collection_name}': {search_error}")
                continue

            for hit in search_hits:
                score = getattr(hit, "score", 0.0)
                if score < 0.2:
                    continue
                payload = getattr(hit, "payload", {}) or {}
                url = payload.get("url", "")
                if url and url in seen_urls:
                    continue

                raw_text = payload.get("text", "") or ""
                cleaned_text = _strip_markup(raw_text)
                steps = _extract_procedural_steps(cleaned_text)

                formatted = {
                    "title": payload.get("title", "") or "",
                    "url": url,
                    "text": cleaned_text[:1000],
                    "steps": steps,
                    "score": score,
                    "collection": collection_name,
                    "source": payload.get("source", "knowledge_base"),
                }
                combined_results.append(formatted)
                if url:
                    seen_urls.add(url)

        if combined_results:
            max_score = max(item.get("score", 0.0) for item in combined_results) or 1.0
            for item in combined_results:
                raw_score = item.get("score", 0.0)
                normalized = max(min(raw_score / max_score, 1.0), 0.0)
                item["score_raw"] = raw_score
                item["relevance_score"] = round(normalized, 4)
                item["score"] = normalized

        # Sort combined results by normalized score descending and truncate to requested limit
        combined_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        formatted_results = combined_results[:limit]

        logger.info(f"Returning {len(formatted_results)} knowledge base results from {len(target_collections)} collection(s)")

        return formatted_results
    except Exception as e:
        logger.error(f"Failed to search docs: {e}")
        return []


@tool(
    "fetch_schema_details",
    description="""Fetch the details of a schema from Incorta.
    
    Args:
        schema_name: Name of the schema to fetch details for. (Only ZendeskTickets, Jira_F are supported)
    
    Returns:
        Details of the schema including tables and columns."""
)
def fetch_schema_details(schema_name: str) -> dict:
    """Fetch schema details from the Incorta environment."""
    env_url = os.getenv("INCORTA_ENV_URL")
    tenant = os.getenv("INCORTA_TENANT")
    user = os.getenv("INCORTA_USERNAME")  # Fix: use os.getenv instead of undefined variable
    password = os.getenv("INCORTA_PASSWORD")
    
    if not all([env_url, tenant, user, password]):
        return {"error": "Missing Incorta credentials. Set INCORTA_ENV_URL, INCORTA_TENANT, INCORTA_USERNAME, INCORTA_PASSWORD"}
    
    try:
        # Login to get session
        response = requests.post(
            f"{env_url}/authservice/login",
            data={"tenant": tenant, "user": user, "pass": password},
            verify=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"Login failed: {response.status_code}"}
        
        # Extract session cookies
        id_cookie, login_id = None, None
        for item in response.cookies.items():
            if item[0].startswith("JSESSIONID"):
                id_cookie, login_id = item
                break
        
        if not id_cookie or not login_id:
            return {"error": "Failed to retrieve session cookies"}
        
        # Verify login and get CSRF token
        response = requests.get(
            f"{env_url}/service/user/isLoggedIn",
            cookies={id_cookie: login_id},
            verify=True,
            timeout=60
        )
        
        if response.status_code != 200 or "XSRF-TOKEN" not in response.cookies:
            return {"error": "Failed to verify login"}
        
        csrf_token = response.cookies["XSRF-TOKEN"]
        authorization = response.json().get("accessToken")
        
        if not authorization:
            return {"error": "Failed to retrieve access token"}
        
        # Fetch schema details
        url = f"{env_url}/bff/v1/schemas/name/{schema_name}"
        cookie = f"{id_cookie}={login_id};XSRF-TOKEN={csrf_token}"
        
        headers = {
            "Authorization": f"Bearer {authorization}",
            "Content-Type": "application/json",
            "X-XSRF-TOKEN": csrf_token,
            "Cookie": cookie
        }
        
        response = requests.get(url, headers=headers, verify=True, timeout=60)
        
        if response.status_code == 200:
            return {"schema_details": response.json()}
        else:
            return {"error": f"Failed to fetch schema details: {response.status_code} - {response.text}"}
    
    except Exception as e:
        logger.error(f"Failed to fetch schema details: {e}")
        return {"error": f"Exception: {str(e)}"}


@tool(
    "fetch_table_data",
    description="""Fetch data from a specified table in the schema using SQL query.
    
    Args:
        spark_sql: SQL query to fetch data from the table.
    
    Returns:
        Data from the table including columns and rows."""
)
def fetch_table_data(spark_sql: str) -> dict:
    """Fetch table data from the Incorta environment."""
    env_url = os.getenv("INCORTA_ENV_URL")
    tenant = os.getenv("INCORTA_TENANT")
    user = os.getenv("INCORTA_USERNAME")  # Fix: use os.getenv instead of undefined variable
    password = os.getenv("INCORTA_PASSWORD")
    
    if not all([env_url, tenant, user, password]):
        return {"error": "Missing Incorta credentials"}
    
    try:
        # Login to get session
        response = requests.post(
            f"{env_url}/authservice/login",
            data={"tenant": tenant, "user": user, "pass": password},
            verify=True,
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"Login failed: {response.status_code}"}
        
        # Extract session cookies
        id_cookie, login_id = None, None
        for item in response.cookies.items():
            if item[0].startswith("JSESSIONID"):
                id_cookie, login_id = item
                break
        
        if not id_cookie or not login_id:
            return {"error": "Failed to retrieve session cookies"}
        
        # Verify login and get CSRF token
        response = requests.get(
            f"{env_url}/service/user/isLoggedIn",
            cookies={id_cookie: login_id},
            verify=True,
            timeout=60
        )
        
        if response.status_code != 200 or "XSRF-TOKEN" not in response.cookies:
            return {"error": "Failed to verify login"}
        
        csrf_token = response.cookies["XSRF-TOKEN"]
        authorization = response.json().get("accessToken")
        
        if not authorization:
            return {"error": "Failed to retrieve access token"}
        
        # Execute SQL query
        url = f"{env_url}/bff/v1/sqlxquery"
        cookie = f"{id_cookie}={login_id};XSRF-TOKEN={csrf_token}"
        
        headers = {
            "Authorization": f"Bearer {authorization}",
            "Content-Type": "application/json",
            "X-XSRF-TOKEN": csrf_token,
            "Cookie": cookie
        }
        
        params = {"sql": spark_sql}
        response = requests.post(url, headers=headers, json=params, verify=True, timeout=60)
        
        if response.status_code == 200:
            return {"data": response.json()}
        else:
            return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}
    
    except Exception as e:
        logger.error(f"Failed to fetch table data: {e}")
        return {"error": f"Exception: {str(e)}"}


# =========================
# Agent Setup
# =========================

class RetryAgentExecutor:
    """
    Wrapper for AgentExecutor that automatically retries with different API keys on quota errors.
    """

    def __init__(self, api_manager, tools, prompt, max_retries: int = 4):
        """
        Initialize retry agent executor.

        Args:
            api_manager: GeminiAPIManager instance
            tools: List of agent tools
            prompt: Agent prompt template
            max_retries: Maximum number of retries (defaults to number of API keys)
        """
        self.api_manager = api_manager
        self.tools = tools
        self.prompt = prompt
        self.max_retries = min(max_retries, len(api_manager.api_keys))
        self.current_executor = None
        self._create_executor()

    def _create_executor(self):
        """Create a new agent executor with current API key."""
        llm = self.api_manager.get_llm()
        agent = create_react_agent(llm, self.tools, self.prompt)
        self.current_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=10
        )

    def stream(self, inputs: dict):
        """
        Stream agent execution with automatic retry on quota errors.

        Args:
            inputs: Agent inputs (e.g., {"input": "query"})

        Yields:
            Agent execution chunks
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                # Stream from current executor
                for chunk in self.current_executor.stream(inputs):
                    yield chunk

                # If we get here, execution succeeded
                self.api_manager.mark_success()
                return

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if it's a quota/rate limit error
                if any(keyword in error_str for keyword in ["quota", "429", "rate limit", "resourceexhausted"]):
                    attempt += 1
                    logger.warning(f"Quota error on attempt {attempt}/{self.max_retries}. Rotating API key...")

                    if attempt < self.max_retries:
                        # Mark failure and rotate key
                        self.api_manager.mark_failure(error_type="quota")

                        # Recreate executor with new API key
                        self._create_executor()

                        # Small delay before retry
                        import time
                        time.sleep(1)
                        continue
                    else:
                        # All keys exhausted
                        raise Exception(f"All {self.max_retries} API keys exhausted quota limits. Please wait or upgrade your plan.") from e
                else:
                    # Non-quota error, re-raise immediately
                    logger.error(f"Non-quota error: {e}")
                    raise

        # If we exit the loop, all retries failed
        raise Exception(f"Failed after {self.max_retries} attempts") from last_error


def create_pm_agent(api_key: Optional[str] = None):
    """
    Create and configure the Product Manager agent with API key rotation support.

    Args:
        api_key: Optional Gemini API key (uses API manager with multiple keys if not provided)

    Returns:
        Configured RetryAgentExecutor or AgentExecutor instance
    """
    # Try to use API manager for multiple keys
    api_manager = None
    if not api_key:
        try:
            from ..api_manager import create_api_manager_from_env
            api_manager = create_api_manager_from_env()
            logger.info(f"Agent: Using API manager with {len(api_manager.api_keys)} key(s)")
        except Exception as e:
            logger.warning(f"Agent: Failed to create API manager: {e}, using single key")
            api_key = _get_secret_or_env("GEMINI_API_KEY")

    if not api_manager and not api_key:
        raise ValueError("GEMINI_API_KEY must be set in environment, Streamlit secrets, or provided as argument")

    # Define tools (using refactored versions with intent analysis)
    tools = [
        search_confluence_tool,  # Updated: now uses intent analysis
        search_slack_tool,        # Updated: now uses intent analysis + channel intelligence
        search_docs,
        fetch_schema_details,
        fetch_table_data
    ]

    # Agent prompt template
    template = """You are an AI Assistant that helps Product Managers search across multiple internal knowledge bases including Confluence, Slack, Docs, Zendesk, and Jira.

Your available tools are:
{tools}

IMPORTANT SEARCH GUIDELINES:

1. Use `search_confluence_pages` to find Confluence documentation:
   - The tool uses optimized query processing with stopwords filtering for better relevance
   - Just pass the user's question directly - no need to extract keywords

2. Use `search_slack_messages` to find Slack conversations:
   - The tool searches across all channels (including private channels the user has access to) with intelligent keyword matching
   - Returns results with enriched metadata
   - Just pass the user's question directly

3. Use `search_docs` to search the knowledge base (Incorta Community, Docs & Support):
   - The tool uses optimized relevance scoring for better results

4. Use `fetch_schema_details` for Zendesk and Jira database schemas:
   - Input must be EXACTLY "ZendeskTickets" for Zendesk questions
   - Input must be EXACTLY "Jira_F" for Jira questions

5. Use `fetch_table_data` to query data from Zendesk/Jira tables

CRITICAL SEARCH STRATEGY WITH SOURCE PRIORITIZATION:

**Source Priority Detection:**
- If the user's query mentions "slack", "channel", "message", "conversation", or "#channelname":
  → PRIORITIZE searching Slack FIRST, then other sources
- If the user's query mentions "docs", "documentation", "knowledge base", "kb", "guide", "tutorial":
  → PRIORITIZE searching Docs FIRST, then other sources
- If the user's query mentions "confluence", "wiki", "page", "internal doc":
  → PRIORITIZE searching Confluence FIRST, then other sources
- If the user's query mentions "zendesk", "ticket", "customer issue":
  → PRIORITIZE Zendesk queries FIRST, then other sources
- If the user's query mentions "jira", "issue", "bug", "feature request", "roadmap":
  → PRIORITIZE Jira queries FIRST, then other sources

**Search Execution:**
- When a source is prioritized, search that source FIRST and wait for results
- After getting prioritized source results, search other relevant sources
- If no source is explicitly mentioned, search all sources in parallel
- Each tool automatically handles relevance scoring - trust the results
- If one tool returns 0 results, that's fine - combine results from other tools
- After searching all sources, synthesize a comprehensive answer from ALL results

CRITICAL RESPONSE RULES:
1. NEVER HALLUCINATE OR INVENT SOURCES
   - If a tool returns empty results [], DO NOT make up page names or references
   - If Confluence returns 0 results, say "No Confluence pages found"
   - If Slack returns 0 results, say "No Slack messages found"

2. ONLY CITE ACTUAL SEARCH RESULTS
   - Use ONLY the exact titles, channel names, and URLs from the Observation
   - If you cannot see a source in the Observation, it does not exist
   - DO NOT reference sources that are not in the actual tool output

3. BE HONEST ABOUT EMPTY RESULTS
   - If all tools return empty results, say "I couldn't find any information"
   - If results exist but don't answer the question, say "The results don't address your specific question"
   - Never pretend to have information you don't actually have

4. RESPONSE FORMAT
   - Base your answer ONLY on what you see in the Observations
   - Include recommendations from Zendesk/Jira tickets when they exist in results
   - Cite specific sources with exact names from the tool output

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # If we have API manager, use retry wrapper
    if api_manager:
        logger.info("Creating RetryAgentExecutor with automatic API key rotation")
        return RetryAgentExecutor(api_manager, tools, prompt)

    # Otherwise, create standard executor with single key
    logger.info("Creating standard AgentExecutor with single API key")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        api_key=api_key
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=10
    )

    return agent_executor


__all__ = [
    "create_pm_agent",
    "search_confluence_optimized",
    "search_slack_messages",
    "search_docs",
    "fetch_schema_details",
    "fetch_table_data"
]

