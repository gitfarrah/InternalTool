from typing import List, Optional, Set
import logging
import os
import re
from html import unescape
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.handler.confluence_handler import search_confluence_pages
from src.handler.slack_handler import search_slack_simplified
from dotenv import load_dotenv
import requests

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor

import jaydebeapi
from pathlib import Path

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp",
                             temperature=0.1, api_key=os.getenv("GEMINI_API_KEY"))


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
    """
    Optimized Confluence search with query preprocessing.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        space_filter: Specific space to search (None for all spaces)
    
    Returns:
        List of page dictionaries with metadata
    """
    # Extract distinct words from user query for better relevance
    query_words = set(query.lower().split())
    # Remove common stop words that don't add meaning
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
    
    print(f"Optimized Confluence search query: {search_query}")
    print(f"Space filter: {space_filter}")
    
    return search_confluence_pages(search_query, max_results, space_filter)


@tool(
    "search_slack_messages",
    description="""Legacy compatibility function for searching Slack messages.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        channel_filter: Specific channel to search (None for all channels)
        max_age_hours: Maximum age of messages in hours (default 168 hours = 1 week)
    
    Returns:
        List of message dictionaries with metadata"""
)
def search_slack_messages(
    query: str,
    max_results: int = 10,
    channel_filter: Optional[str] = None,
    max_age_hours: int = 168  # 1 week default
) -> List[dict]:
    """
    Legacy compatibility function for your existing app.py.
    This function adapts the new search system to your existing interface.
    """
    if not query or not query.strip():
        return []

    # Create intent data for the new system
    # Extract keywords from query for better relevance
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

    # Use the new simplified search
    results = search_slack_simplified(query, intent_data, max_results)
    
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

def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def get_qdrant_client():
    """Initialize Qdrant client with error handling."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        client.get_collections()  # smoke test
        return client
    except Exception as e:
        logger.error("Failed to connect to Qdrant.")
        logger.debug(f"QDRANT_URL={QDRANT_URL}\nError={repr(e)}")
        raise


@tool(    "search_docs",
    description="""Search documents in Qdrant collection.
    
    Args:
        query: Search query
        limit: Number of results to return
    
    Returns:
        List of search results with payloads"""
)
def search_docs(query: str, limit: int = 5):
    """Search documents in Qdrant collections with step extraction."""
    client = get_qdrant_client()
    embedding_model = load_embedding_model()
    query_vector = embedding_model.encode([query])[0]

    def _discover_doc_collections(qdrant_client: QdrantClient) -> List[str]:
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
        return ["docs"]

    def _strip_markup(raw_text: str) -> str:
        if not raw_text:
            return ""
        text = re.sub(r'@@@hl@@@(.*?)@@@endhl@@@', r'\1', raw_text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_steps(text: str, max_steps: int = 6) -> List[str]:
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

        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            candidate = sentence.strip()
            if len(candidate) < 25:
                continue
            steps.append(candidate)
            if len(steps) >= max_steps:
                break
        return steps[:max_steps]

    collections_to_search = _discover_doc_collections(client)
    per_collection_limit = max(limit, 5)
    combined_results: List[dict] = []
    seen_urls: Set[str] = set()

    for collection_name in collections_to_search:
        try:
            hits = client.search(
                collection_name=collection_name,
                query_vector=("content_vector", query_vector),
                limit=per_collection_limit,
                with_payload=True
            )
        except Exception as search_error:
            logger.warning(f"Docs search failed for collection '{collection_name}': {search_error}")
            continue

        for hit in hits:
            score = getattr(hit, "score", 0.0)
            if score < 0.2:
                continue
            payload = getattr(hit, "payload", {}) or {}
            url = payload.get("url", "")
            if url and url in seen_urls:
                continue

            raw_text = payload.get("text", "") or ""
            cleaned_text = _strip_markup(raw_text)
            steps = _extract_steps(cleaned_text)

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

    combined_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return combined_results[:limit]

env_url = os.getenv("INCORTA_ENV_URL")
tenant = os.getenv("INCORTA_TENANT")
PAT = os.getenv("PAT")
password = os.getenv("INCORTA_PASSWORD")


def login_with_credentials(env_url, tenant, user, password):
    """
    Logs into Incorta using username, password, and tenant credentials.
    Returns a session with relevant authentication details.
    """
    response = requests.post(
        f"{env_url}/authservice/login",
        data={"tenant": tenant, "user": user, "pass": password},
        verify=True,
        timeout=60
    )

    if response.status_code != 200:
        response.raise_for_status()

        # Extract session cookies
    id_cookie, login_id = None, None
    for item in response.cookies.items():
        if item[0].startswith("JSESSIONID"):
            id_cookie, login_id = item
            break

    if not id_cookie or not login_id:
        raise Exception("Failed to retrieve session cookies during login.")

        # Verify login and retrieve CSRF token
    response = requests.get(
        f"{env_url}/service/user/isLoggedIn",
        cookies={id_cookie: login_id},
        verify=True,
        timeout=60
    )

    if response.status_code != 200 or "XSRF-TOKEN" not in response.cookies:
        raise Exception(f"Failed to log in to {env_url} for tenant {tenant} using user {user}. Please verify credentials.")

        # Retrieve CSRF token and access token
    csrf_token = response.cookies["XSRF-TOKEN"]
    authorization = response.json().get("accessToken")

    if not authorization:
        raise Exception("Failed to retrieve access token during login.")

    return {
        "env_url": env_url,
        "id_cookie": id_cookie,
        "id": login_id,
        "csrf": csrf_token,
        "authorization": authorization,
        "verify": True,
        "session_cookie": {id_cookie: login_id, "XSRF-TOKEN": csrf_token}
    }

@tool(
    "fetch_schema_details",
    description="""Fetch the details of a schema from Incorta.
    
    Args:
        schema_name: Name of the schema to fetch details for. (Only ZendeskTickets, Jira_F are supported)
    
    Returns:
        Details of the schema including tables and columns."""
)
def fetch_schema_details(schema_name: str):
    """Fetch schema details from the Incorta environment."""

    login_creds = login_with_credentials(env_url, tenant, user, password)

    url = f"{env_url}/bff/v1/schemas/name/{schema_name}"

    cookie = ""
    for key, value in login_creds['session_cookie'].items():
        cookie += f"{key}={value};"

    headers = {
        "Authorization": f"Bearer {login_creds['authorization']}",
        "Content-Type": "application/json",
        "X-XSRF-TOKEN": login_creds["csrf"],
        "Cookie": cookie
    }

    response = requests.get(url, headers=headers, verify=False)

    if response.status_code == 200:
        response_data = response.json()

        return {"schema_details": response_data}
    else:
        return {"error": f"Failed to fetch schema details: {response.status_code} - {response.text}"}
    

sqlx_host = os.getenv("INCORTA_SQLX_HOST")
user = os.getenv("INCORTA_USERNAME")
driver = "org.apache.hive.jdbc.HiveDriver"


@tool(
    "fetch_table_data",
    description="""Fetch data from a specified table in the schema.
    
    Args:
        spark_sql: SQL query to fetch data from the table.
    
    Returns:
        Data from the table including columns and rows."""
)
def fetch_table_data(spark_sql: str):
    """Fetch table data from the Incorta environment."""

    login_creds = login_with_credentials(env_url, tenant, user, password)

    url = f"{env_url}/bff/v1/sqlxquery"

    cookie = ""
    for key, value in login_creds['session_cookie'].items():
        cookie += f"{key}={value};"

    headers = {
        "Authorization": f"Bearer {login_creds['authorization']}",
        "Content-Type": "application/json",
        "X-XSRF-TOKEN": login_creds["csrf"],
        "Cookie": cookie
    }

    params = {
        "sql": spark_sql
    }

    response = requests.post(url, headers=headers, json=params, verify=False)

    if response.status_code == 200:
        return {"data": response.json()}
    else:
        return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}


tools = [search_confluence_optimized, search_slack_messages, search_docs, fetch_schema_details, fetch_table_data]

from langchain_core.prompts import PromptTemplate

template = '''You are an AI Assistant that helps the Product Managers to Search across multiple internal knowledge bases including Confluence, Slack, Docs, Zendesk, and Jira.

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

if the PM keyword doesn't return any relevant results try to use simmilar keywords that are related to the PM keyword.

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
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)


agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)


if __name__ == "__main__":
    print("Starting interactive agent session...")
    print("You can ask questions about Confluence, Slack, Docs, Zendesk, and Jira.")
    print("Type 'exit' to quit.")
    print("-----------------------------------------------------")
    while True:
        question = input("Enter your question: ")
        if question.lower() == 'exit':
            break
        for chunk in agent_executor.stream({"input": question}):
            # Print agent actions (tool calls)
            if "actions" in chunk:
                for action in chunk["actions"]:
                    print(f"\n🔧 Tool Call: {action.tool}")
                    print(f"📥 Input: {action.tool_input}")
            
            # Print tool observations (tool outputs)
            if "steps" in chunk:
                for step in chunk["steps"]:
                    print(f"\n📤 Tool Output ({step.action.tool}):")
                    print(f"{step.observation}")
            
            # Print final output
            if "output" in chunk:
                print(f"\n✅ Final Answer:")
                print(f"{chunk['output']}")
                print("\n" + "="*60 + "\n")