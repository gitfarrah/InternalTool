"""
Product Manager Agent for LangChain ReAct Agent Executor.

Provides agent tools and setup for searching across multiple knowledge sources
including Confluence, Slack, Docs, Zendesk, and Jira.
"""

from typing import List, Optional
import logging
import os
import requests
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from ..handler.confluence_handler import search_confluence
from ..handler.slack_handler import search_slack_simplified
from ..storage.cache_manager import get_cached_search_results, cache_search_results

logger = logging.getLogger(__name__)


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
        List of page dictionaries with metadata"""
)
def search_confluence_optimized(
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
    description="""Search for messages in Slack across all channels.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        channel_filter: Specific channel to search (None for all channels)
        max_age_hours: Maximum age of messages in hours (default 0 = all history)
    
    Returns:
        List of message dictionaries with metadata"""
)
def search_slack_messages(
    query: str,
    max_results: int = 10,
    channel_filter: Optional[str] = None,
    max_age_hours: int = 0  # 0 = all history
) -> List[dict]:
    """Search Slack messages with intent-aware processing."""
    if not query or not query.strip():
        return []

    # Create intent data for the search system
    query_terms = query.lower().split()
    keywords = [term for term in query_terms if len(term) > 2]
    priority_terms = keywords[:3]  # Top 3 terms as priority

    intent_data = {
        "slack_params": {
            "keywords": keywords,
            "priority_terms": priority_terms,
            "channels": channel_filter if channel_filter else "all",
            "time_range": "all",  # Repository system searches all history
            "limit": max_results
        },
        "search_strategy": "fuzzy_match"
    }

    # Use the simplified search
    results = search_slack_simplified(query, intent_data, max_results)
    
    # Convert to legacy format for compatibility
    legacy_results = []
    for result in results:
        legacy_results.append({
            "text": result.get("text", ""),
            "username": result.get("username", "Unknown"),
            "channel": result.get("channel", "unknown"),
            "ts": result.get("ts", ""),
            "permalink": result.get("permalink", ""),
            "source": "slack"
        })
    
    return legacy_results


def _search_docs_impl(query: str, limit: int = 5) -> List[dict]:
    """Implementation for searching documents in Qdrant collection."""
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
                import streamlit as st  # optional
                qdrant_url = st.secrets.get("QDRANT_URL", "") or st.secrets.get("QDRANT_HOST", "")
                qdrant_api_key = st.secrets.get("QDRANT_API_KEY", qdrant_api_key)
            except Exception:
                pass
        if not qdrant_url:
            logger.warning("QDRANT_URL not set; knowledge base search disabled")
            return []

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
        query_vector = embedding_model.encode([query])[0]

        search_result = client.search(
            collection_name="docs",
            query_vector=("content_vector", query_vector),
            limit=limit,
            with_payload=True
        )
        
        # Format results
        formatted_results = []
        for r in search_result:
            if r.score >= 0.2:  # Lower threshold to avoid over-filtering
                formatted_results.append({
                    "title": r.payload.get("title", "") or "",
                    "url": r.payload.get("url", "") or "",
                    "text": r.payload.get("text", "") or "",
                    "score": r.score,
                    "source": "knowledge_base"
                })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Failed to search docs: {e}")
        return []


@tool(
    "search_docs",
    description="""Search documents in Qdrant collection (Incorta Community, Docs & Support).
    
    Args:
        query: Search query
        limit: Number of results to return
    
    Returns:
        List of search results with metadata"""
)
def search_docs(query: str, limit: int = 5) -> List[dict]:
    """Tool wrapper for agent usage."""
    return _search_docs_impl(query, limit)


def search_docs_plain(query: str, limit: int = 5) -> List[dict]:
    """Plain callable for programmatic use (no LangChain tool wrapping)."""
    return _search_docs_impl(query, limit)


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
            api_key = os.getenv("GEMINI_API_KEY")

    if not api_manager and not api_key:
        raise ValueError("GEMINI_API_KEY must be set in environment or provided as argument")

    # Define tools
    tools = [
        search_confluence_optimized,
        search_slack_messages,
        search_docs,
        fetch_schema_details,
        fetch_table_data
    ]

    # Agent prompt template
    template = """You are an AI Assistant that helps the Product Managers to Search across multiple internal knowledge bases including Confluence, Slack, Docs, Zendesk, and Jira.

Your available tools are:
{tools}

Use the `search_confluence_pages` tool to search for relevant Confluence pages.

Use the `search_slack_messages` tool to search for relevant Slack messages.

Use the `search_docs` tool to search for relevant Docs for the PM.

Use the `fetch_schema_details` tool to get the details of Zendesk and Jira
the input of the `fetch_schema_details` tool should be ONLY ZendeskTickets if the question is about Zendesk
and Jira_F if the question is about Jira.

Use the `fetch_table_data` tool to get the data from the tables in ZendeskTickets and Jira_F schemas.


Your Default is to search in all the resources using the relevant tools above.
but if a PM specifically asks to search in a specific resource, use the relevant tool only.

if the PM keyword doesn't return any relevant results try to use similar keywords that are related to the PM keyword.

Provide Recommendations based on relevant tickets from Zendesk and Jira for the PMs to be able to take action on them.

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
        model="gemini-2.0-flash-exp",
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

