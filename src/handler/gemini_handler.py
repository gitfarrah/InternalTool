"""
Gemini handler for AI-powered search result analysis and answer generation.

Combines structured citation extraction with conversational capabilities,
model fallback, and enhanced error handling.
"""

from __future__ import annotations

import os
import json
import logging
import time
from typing import Optional, Sequence, List, Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv
import toon

from ..storage.cache_manager import get_cached_llm_response, cache_llm_response

logger = logging.getLogger(__name__)

# Load API key from environment variables
load_dotenv()

# Initialize API manager with all available keys
_api_manager = None
_single_api_key = None

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
    # This will work because load_dotenv() is called at module level
    return os.getenv(name, default)

def _get_api_manager():
    """Get or create API manager instance (supports multiple keys with rotation)."""
    global _api_manager, _single_api_key
    
    if _api_manager is None:
        try:
            from ..api_manager import create_api_manager_from_env
            _api_manager = create_api_manager_from_env()
            logger.info(f"API Manager initialized with {len(_api_manager.api_keys)} key(s)")
        except Exception as e:
            logger.warning(f"Failed to create API manager: {e}, falling back to single key")
            _single_api_key = _get_secret_or_env("GEMINI_API_KEY")
            if not _single_api_key:
                raise RuntimeError("Missing GEMINI_API_KEY in environment variables or Streamlit secrets")
            logger.info("Using single API key (no rotation)")
    
    return _api_manager

def _get_current_api_key():
    """Get current API key (rotates if using API manager)."""
    api_mgr = _get_api_manager()
    if api_mgr:
        # Get current key based on rotation strategy
        if api_mgr.rotation_strategy == "round_robin":
            current_key = api_mgr.api_keys[api_mgr.current_index]
        else:
            import random
            current_key = random.choice(api_mgr.api_keys)
        return current_key
    else:
        return _single_api_key

# Initialize and configure genai with first available key
try:
    initial_key = _get_current_api_key()
    genai.configure(api_key=initial_key)
    logger.info("Genai configured with initial API key")
except Exception as e:
    logger.error(f"Failed to configure genai: {e}")
    raise

# Default model: Gemini 2.5 Flash
DEFAULT_MODEL = "gemini-2.5-flash"

# Supported models with fallback order
SUPPORTED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
]


def _supported_models() -> List[str]:
    """
    Get list of supported models with fallback order.
    Can be extended to check available models from API.
    """
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        return [env_model] + [m for m in SUPPORTED_MODELS if m != env_model]
    
    return SUPPORTED_MODELS


SYSTEM_MSG = """Ibn Battouta: AI search assistant for Incorta PM intelligence. Analyze passages from multiple sources, synthesize accurate answers for Product Managers.

DATA SOURCES:
knowledge_base: Community, docs, support (official)
slack: Discussions, announcements (real-time)
confluence: Internal docs, projects (detailed)
zendesk: Support tickets, issues (customer perspective)
jira: Features, bugs, tracking (dev perspective)

SOURCE PRIORITY (query type → priority order):
features_docs: knowledge_base>confluence>slack>jira
releases_dates: slack>knowledge_base>jira>confluence
customer_issues: zendesk>jira>slack>confluence
dev_roadmap: jira>slack>confluence>knowledge_base
processes: confluence>slack>knowledge_base
troubleshooting: all_equal (favor recent)

CITATION RULES:
1. ALWAYS include "source" field (knowledge_base|slack|confluence|zendesk|jira)
2. Use ONLY supplied passages, no inference
3. Quote 1-2 key sentences supporting answer
4. Preserve exact: version numbers, IDs, dates, terms
5. Slack: add username/channel for credibility
6. Zendesk: note patterns if multiple tickets
7. Jira: include status/priority if relevant

SYNTHESIS:
- Sources agree → merge into unified answer
- Sources conflict → note discrepancy with dates
- Complementary → comprehensive synthesis
- Cross-reference when relevant
- No repetition

TEMPORAL:
- "latest/current" → prioritize Slack/Jira over old docs
- Include dates from passages
- Flag potentially outdated info

ANSWER STRUCTURE:
1. Direct answer first
2. Supporting context
3. Actionable next steps (PM queries)

LENGTH: Simple=2-4 sent, Complex=4-8 sent, PM=+recommendations

TONE: Concise, factual, professional. No filler. State what's missing if uncertain.

QUERY-SPECIFIC:
when/date → explicit dates/timeframes
how/why → actionable steps
status → current state+next steps
customer_impact → Zendesk patterns
roadmap → Jira+Confluence cross-ref

PM VALUE:
- Pattern ID across Zendesk+Jira
- Connect pain points to features/roadmap
- Data-driven recommendations
- Flag doc vs implementation gaps

OUTPUT (JSON only):
{
  "exists": bool,
  "answer": "Direct response, context, PM insights",
  "citations": [{"url": str, "title": str, "evidence": "1-2 sentences", "source": "knowledge_base|slack|confluence|zendesk|jira"}]
}

SPECIAL:
- Incomplete passages → state available+missing
- No info → exists=false, brief explanation
- PM recommendations → multi-source synthesis
- Cross-source patterns → explicit connections

Return valid JSON only. No markdown, no commentary.
"""


def build_user_payload(query: str, passages: List[dict], max_chars_per_passage: int = 900, use_toon: bool = True) -> str:
    """
    Build payload from query and passages using TOON format for token efficiency.

    Args:
        query: User query string
        passages: List of passage dicts with title, url, text/excerpt, source
        max_chars_per_passage: Maximum characters per passage snippet
        use_toon: If True, use TOON format (30-60% fewer tokens); if False, use JSON

    Returns:
        TOON-encoded or JSON string of query and passages
    """
    blocks = []
    for p in passages:
        snippet = (p.get("text") or p.get("excerpt") or "")[:max_chars_per_passage]
        blocks.append({
            "title": p.get("title", ""),
            "url": p.get("url", ""),
            "snippet": snippet,
            "source": p.get("source", "unknown")
        })
    payload = {"query": query, "passages": blocks}
    if use_toon:
        try:
            # Use TOON format for 30-60% token reduction
            return toon.encode(payload)
        except Exception as e:
            # Fallback to JSON if TOON encoding fails
            logger.warning(f"TOON encoding failed, falling back to JSON: {e}")
            return json.dumps(payload, ensure_ascii=False)
    else:
        return json.dumps(payload, ensure_ascii=False)


def build_enhanced_prompt(
    query: str,
    passages: List[dict],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    query_type: str = "new_search"
) -> str:
    """
    Build enhanced prompt with conversation context and query type awareness.

    Args:
        query: User query
        passages: List of passage dictionaries
        conversation_history: Previous conversation messages
        query_type: Type of query (new_search, follow_up, clarification)

    Returns:
        Complete prompt string
    """
    user_payload = build_user_payload(query, passages)

    # Build conversation context if available
    conv_context = ""
    if conversation_history and len(conversation_history) > 1:
        recent_conv = conversation_history[-4:]  # Last 2 exchanges
        conv_lines = []
        for msg in recent_conv:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")[:150]  # Truncate long messages
            conv_lines.append(f"{role}: {content}")
        conv_context = "\nPrevious conversation:\n" + "\n".join(conv_lines) + "\n"

    # Adapt instructions based on query type
    additional_instructions = ""
    if query_type in ["follow_up", "clarification"]:
        additional_instructions += (
            "\nNote: This is a follow-up question. Reference previous conversation "
            "if relevant, but still cite sources for new information."
        )

    # Analyze query to provide targeted instructions
    query_lower = query.lower()

    # Query-specific instructions
    if any(term in query_lower for term in ["release", "version", "date", "when", "ga", "general availability", "announcement"]):
        additional_instructions += (
            "\n⚠️ RELEASE/DATE QUERY: Prioritize recent Slack discussions in release/announce channels. "
            "Check Jira for planned release tickets. Include specific dates, version numbers, and timelines. "
            "If date is uncertain, check multiple sources and note any discrepancies."
        )

    if any(term in query_lower for term in ["how to", "configure", "setup", "implement", "install"]):
        additional_instructions += (
            "\n⚠️ HOW-TO QUERY: Prioritize knowledge_base and Confluence documentation. "
            "Provide step-by-step guidance if available. Include links to detailed guides."
        )

    if any(term in query_lower for term in ["customer", "issue", "problem", "bug", "ticket"]):
        additional_instructions += (
            "\n⚠️ CUSTOMER ISSUE QUERY: Check Zendesk for customer-reported patterns. "
            "Cross-reference with Jira for development status. Note frequency/severity if multiple tickets exist."
        )

    if any(term in query_lower for term in ["roadmap", "plan", "future", "upcoming", "next"]):
        additional_instructions += (
            "\n⚠️ ROADMAP QUERY: Check Jira for planned work and priorities. "
            "Reference Confluence for strategic roadmap docs. Note development status and timelines."
        )

    if any(term in query_lower for term in ["status", "progress", "current"]):
        additional_instructions += (
            "\n⚠️ STATUS QUERY: Check Jira for development status, Slack for recent updates. "
            "Provide current state and expected next steps."
        )

    if any(term in query_lower for term in ["recommend", "should", "best practice", "advice"]):
        additional_instructions += (
            "\n⚠️ RECOMMENDATION REQUEST: Synthesize insights from multiple sources. "
            "Provide data-driven recommendations based on patterns from Zendesk, Jira, and internal discussions."
        )

    # Detect if query mentions multiple sources
    source_count = sum([
        1 for source in ["zendesk", "jira", "slack", "confluence", "docs"]
        if source in query_lower
    ])
    if source_count >= 2:
        additional_instructions += (
            "\n⚠️ MULTI-SOURCE QUERY: User is asking to compare/synthesize across multiple sources. "
            "Explicitly show information from each requested source and connect the insights."
        )

    prompt = (
        f"{SYSTEM_MSG}\n\n"
        "User Query and Passages (TOON format - compact, human-readable):\n"
        f"{user_payload}\n{conv_context}\n"
        "Rules:\n"
        "- Cite only passages that directly support the answer.\n"
        "- Keep 'evidence' to 1–2 sentences copied from the snippet (no ellipses at both ends).\n"
        "- If unsure, set exists=false and explain briefly.\n"
        "- ALWAYS include 'source' field in citations.\n"
        f"{additional_instructions}"
    )

    return prompt


def _parse_json_response(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON response from LLM, handling markdown code fences.
    
    Args:
        text: Raw response text from LLM
    
    Returns:
        Parsed dictionary with defaults
    """
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: wrap the raw text
        logger.warning("Failed to parse JSON, using fallback format")
        data = {"exists": False, "answer": text[:600], "citations": []}
    
    # Ensure required fields exist
    data.setdefault("exists", False)
    data.setdefault("answer", "")
    data.setdefault("citations", [])
    
    return data


def answer_with_citations(
    query: str,
    passages: List[dict],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    query_type: str = "new_search",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate answer with citations from passages.
    
    Args:
        query: User query string
        passages: List of passage dictionaries with title, url, text/excerpt, source
        conversation_history: Optional conversation history for context
        query_type: Type of query (new_search, follow_up, clarification)
        model_name: Optional model name override
    
    Returns:
        Dictionary with exists, answer, and citations
    """
    user_payload = build_user_payload(query, passages)
    
    # Check cache for LLM response
    cached_response = get_cached_llm_response(query, user_payload)
    if cached_response:
        logger.info(f"Cache hit for LLM response: {query[:50]}...")
        return cached_response
    
    # Build enhanced prompt
    prompt = build_enhanced_prompt(query, passages, conversation_history, query_type)
    
    # Get model list
    if model_name:
        candidate_models = [model_name] + [m for m in _supported_models() if m != model_name]
    else:
        candidate_models = _supported_models()
    
    last_error: Optional[Exception] = None
    
    # Get API manager for key rotation
    api_mgr = _get_api_manager()
    
    # Try models with fallback and key rotation
    max_model_attempts = len(candidate_models)
    max_key_attempts = len(api_mgr.api_keys) if api_mgr else 1
    
    for model_name_attempt in candidate_models:
        # Try with different API keys if we have multiple
        for key_attempt in range(max_key_attempts):
            try:
                # Get current API key and reconfigure if needed
                current_key = _get_current_api_key()
                genai.configure(api_key=current_key)
                
                logger.info(f"Trying Gemini model: {model_name_attempt} with API key {key_attempt + 1}/{max_key_attempts}")
                model = genai.GenerativeModel(model_name_attempt)
                
                # Configure generation for structured responses (same temperature as main.py)
                generation_config = {
                    "temperature": 0.1,  # Same as main.py for consistency
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 1000,
                }
                
                # Adapt for follow-ups
                if query_type in ["follow_up", "clarification"]:
                    generation_config["temperature"] = 0.2
                    generation_config["max_output_tokens"] = 800
                
                # Retry logic for transient errors (per key)
                for attempt in range(2):
                    try:
                        resp = model.generate_content(
                            prompt,
                            generation_config=generation_config
                        )
                        
                        text = resp.text.strip() if hasattr(resp, "text") else ""
                        if text:
                            data = _parse_json_response(text)
                            
                            # Mark success in API manager
                            if api_mgr:
                                api_mgr.mark_success()
                            
                            # Cache the response
                            try:
                                cache_llm_response(query, user_payload, data)
                            except Exception as e:
                                logger.debug(f"Failed to cache LLM response: {e}")
                            
                            return data
                        
                    except Exception as e:
                        error_str = str(e).lower()
                        last_error = e
                        
                        # Check if it's a quota/rate limit error - rotate key immediately
                        if any(keyword in error_str for keyword in ["quota", "429", "rate limit", "resource_exhausted"]):
                            if api_mgr:
                                logger.warning(f"Rate limit/quota error detected, rotating API key")
                                api_mgr.mark_failure(error_type="quota")
                                # Get next key for next iteration
                                break  # Break from retry loop, try next key
                            else:
                                # Single key, wait and retry
                                if attempt < 1:
                                    logger.warning(f"Rate limit error with single key, waiting...")
                                    time.sleep(1)  # Wait before retry
                                    continue
                        else:
                            # Non-quota error, retry with same key
                            if attempt < 1:
                                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                                time.sleep(0.2)
                                continue
                            else:
                                # Failed after retries, try next key
                                break
                
                # If we get here, both retries failed with this key
                if api_mgr and key_attempt < max_key_attempts - 1:
                    # Rotate to next key and try again
                    api_mgr.rotate_key()
                    continue
                else:
                    # No more keys to try with this model
                    break
                
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name_attempt} failed with key {key_attempt + 1}: {e}")
                # Rotate key if we have API manager and more keys to try
                if api_mgr and key_attempt < max_key_attempts - 1:
                    api_mgr.rotate_key()
                    continue
                else:
                    # Try next model
                    break
        
        # If all keys failed for this model, try next model
        continue
    
    # If all models fail, return fallback response
    logger.error(f"All Gemini model attempts failed. Last error: {last_error}")
    return {
        "exists": False,
        "answer": "Sorry, I couldn't generate a response right now. Please try again in a moment.",
        "citations": []
    }


def answer_with_multiple_sources(
    query: str,
    qdrant_results: List[dict],
    slack_results: List[dict],
    confluence_results: List[dict],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    query_type: str = "new_search",
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate answer using multiple data sources (Qdrant, Slack, Confluence).
    
    Args:
        query: User query
        qdrant_results: Results from Qdrant vector search
        slack_results: Results from Slack search
        confluence_results: Results from Confluence search
        conversation_history: Optional conversation history for context
        query_type: Type of query (new_search, follow_up, clarification)
        model_name: Optional model name override
    
    Returns:
        Dictionary with answer and citations from all sources
    """
    # Combine all results into a single passages list
    all_passages = []
    
    # Add Qdrant results
    for result in qdrant_results:
        all_passages.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "text": result.get("text", ""),
            "source": "knowledge_base"
        })
    
    # Add Slack results
    for result in slack_results:
        all_passages.append({
            "title": f"Slack: #{result.get('channel', 'unknown')} - @{result.get('username', 'unknown')}",
            "url": result.get("permalink", ""),
            "text": result.get("text", ""),
            "source": "slack"
        })
    
    # Add Confluence results
    for result in confluence_results:
        all_passages.append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "text": result.get("excerpt", ""),
            "source": "confluence"
        })
    
    # Build context string for caching
    context_str = json.dumps({"passages": all_passages}, ensure_ascii=False)
    
    # Check cache for LLM response
    cached_response = get_cached_llm_response(query, context_str)
    if cached_response:
        logger.info(f"Cache hit for multi-source LLM response: {query[:50]}...")
        return cached_response
    
    # Use the enhanced answer_with_citations function
    response = answer_with_citations(
        query,
        all_passages,
        conversation_history=conversation_history,
        query_type=query_type,
        model_name=model_name
    )
    
    # Cache the response
    try:
        cache_llm_response(query, context_str, response)
    except Exception as e:
        logger.debug(f"Failed to cache multi-source LLM response: {e}")
    
    return response


# Backward compatibility: Simple function without conversation history
def answer_with_citations_simple(query: str, passages: List[dict]) -> Dict[str, Any]:
    """
    Simple version without conversation history (backward compatibility).
    
    Args:
        query: User query string
        passages: List of passage dictionaries
    
    Returns:
        Dictionary with exists, answer, and citations
    """
    return answer_with_citations(query, passages)


__all__ = [
    "answer_with_citations",
    "answer_with_citations_simple",  # Backward compatibility
    "answer_with_multiple_sources",
    "build_user_payload",
    "build_enhanced_prompt",
    "_supported_models",
    "_parse_json_response"
]
