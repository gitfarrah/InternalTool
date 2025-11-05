from typing import List, Optional
import logging
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.handler.confluence_handler import search_confluence
from src.handler.slack_handler import search_slack
from dotenv import load_dotenv
import os
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
    
    return search_confluence(search_query, max_results, space_filter)


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
    results = search_slack(query, intent_data, max_results)
    
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


@tool("search_docs",
    description="""Search documents in Qdrant collection.
    
    Args:
        query: Search query
        limit: Number of results to return
    
    Returns:
        List of search results with payloads"""
)
def search_docs(query: str, limit: int = 5):
    """Search documents in Qdrant collection."""
    client = get_qdrant_client()
    embedding_model = load_embedding_model()
    query_vector = embedding_model.encode([query])[0]

    search_result = client.search(
        collection_name="docs",
        query_vector=("content_vector", query_vector),
        limit=limit,
        with_payload=True,
        score_threshold=0.0  # Get all results, let relevance scoring handle ranking
    )
    
    # Log relevance scores for knowledge base results
    logger.info("=" * 80)
    logger.info("KNOWLEDGE BASE RELEVANCE SCORES (Qdrant Search Results):")
    logger.info("=" * 80)
    for idx, result in enumerate(search_result[:10], 1):
        # Handle both ScoredPoint and dict formats
        if hasattr(result, 'score'):
            score = result.score
            payload = result.payload if hasattr(result, 'payload') else {}
        elif isinstance(result, dict):
            score = result.get('score', 0.0)
            payload = result.get('payload', {})
        else:
            score = 0.0
            payload = {}
        
        title = payload.get('title', 'Unknown')
        url = payload.get('url', '')
        text_preview = (payload.get('text', '') or payload.get('content', ''))[:60]
        logger.info(f"{idx}. Score: {score:.4f} | Title: {title}")
        if url:
            logger.info(f"   URL: {url}")
        if text_preview:
            logger.info(f"   Preview: {text_preview}...")
    logger.info("=" * 80)
    
    return search_result

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
        print("\n" + "="*80)
        print("AGENT THINKING CHAIN:")
        print("="*80)
        
        for chunk in agent_executor.stream({"input": question}):
            # Print agent thinking/thoughts
            if "intermediate_steps" in chunk:
                for step in chunk["intermediate_steps"]:
                    if hasattr(step, 'action') and hasattr(step, 'observation'):
                        print(f"\n💭 Thought: {getattr(step.action, 'tool_input', {}).get('input', 'Processing...')}")
                        print(f"🔧 Action: {step.action.tool}")
                        print(f"📥 Input: {step.action.tool_input}")
                        print(f"📤 Observation: {str(step.observation)[:200]}...")
            
            # Print agent actions (tool calls)
            if "actions" in chunk:
                for action in chunk["actions"]:
                    print(f"\n💭 Thought: {getattr(action, 'log', '')[:200] if hasattr(action, 'log') else 'Taking action...'}")
                    print(f"🔧 Tool Call: {action.tool}")
                    print(f"📥 Input: {action.tool_input}")
            
            # Print tool observations (tool outputs)
            if "steps" in chunk:
                for step in chunk["steps"]:
                    print(f"\n💭 Thought: {getattr(step.action, 'log', '')[:200] if hasattr(step.action, 'log') else 'Processing result...'}")
                    print(f"📤 Tool Output ({step.action.tool}):")
                    print(f"{str(step.observation)[:300]}...")
            
            # Print agent scratchpad (thinking process)
            if "agent" in chunk and hasattr(chunk["agent"], "scratchpad"):
                scratchpad = str(chunk["agent"].scratchpad)
                if scratchpad:
                    print(f"\n💭 Agent Scratchpad: {scratchpad[:300]}...")
            
            # Print final output
            if "output" in chunk:
                print("\n" + "="*80)
                print("✅ FINAL ANSWER:")
                print("="*80)
                print(f"{chunk['output']}")
                print("\n" + "="*80 + "\n")