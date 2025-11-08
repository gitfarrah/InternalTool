"""
Intent analyzer for query understanding and search parameter extraction.

This module provides intelligent intent analysis using AI-powered analysis
with Gemini API, along with rule-based preprocessing for better performance.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import Counter

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

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


# Enhanced stopwords including technical and common words
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been", 
    "being", "have", "has", "had", "do", "does", "did", "will", "would", 
    "could", "should", "may", "might", "can", "about", "what", "when", 
    "where", "why", "how", "who", "which", "this", "that", "these", "those",
    "am", "your", "you", "me", "my", "i", "we", "us", "our", "they", "them",
    "their", "it", "its", "get", "got", "getting", "give", "gave", "given",
    "updates", "update", "tell", "show", "find", "search", "look", "see",
    "there", "here", "any", "some", "all", "more", "most", "such", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "under", "again", "further", "then", "once"
}


# Global API manager instance
_api_manager = None

def configure_genai() -> None:
    """Configure the Google Generative AI client with API manager support."""
    global _api_manager
    
    # Try to use API manager first (supports multiple keys)
    if _api_manager is None:
        try:
            from ..api_manager import create_api_manager_from_env
            _api_manager = create_api_manager_from_env()
            logger.info(f"Intent analyzer: Using API manager with {len(_api_manager.api_keys)} key(s)")
        except Exception as e:
            logger.warning(f"Intent analyzer: Failed to create API manager: {e}, using single key")
            _api_manager = None
    
    # Get current API key
    if _api_manager:
        # Use current key based on rotation
        if _api_manager.rotation_strategy == "round_robin":
            api_key = _api_manager.api_keys[_api_manager.current_index]
        else:
            import random
            api_key = random.choice(_api_manager.api_keys)
    else:
        api_key = _get_secret_or_env("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment variables or Streamlit secrets.")
    
    genai.configure(api_key=api_key)


def extract_keywords_smart(text: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
    """
    Intelligently extract keywords from text using semantic understanding.
    
    Strategies:
    1. Extract technical identifiers (version numbers, IDs, codes)
    2. Extract capitalized terms (proper nouns, acronyms)
    3. Extract domain-specific terms
    4. Extract semantically important words based on context
    """
    if not text or not text.strip():
        return []
    
    keywords: List[str] = []
    text_lower = text.lower()
    
    # Strategy 1: Extract version numbers, IDs, and technical codes
    technical_pattern = r'\b(?:v?\d+\.\d+(?:\.\d+)?(?:\.\d+)?|[A-Z]+-\d+|[A-Z]{2,}\d+)\b'
    technical_matches = re.findall(technical_pattern, text, re.IGNORECASE)
    keywords.extend([m.lower() for m in technical_matches])
    
    # Strategy 2: Extract capitalized terms (proper nouns, acronyms)
    capitalized_pattern = r'\b[A-Z][A-Za-z]*(?:[A-Z][a-z]*)*\b'
    capitalized_matches = re.findall(capitalized_pattern, text)
    capitalized_filtered = [
        m.lower() for m in capitalized_matches 
        if len(m) > 1 and m.lower() not in STOP_WORDS
    ]
    keywords.extend(capitalized_filtered)
    
    # Strategy 3: Extract quoted terms (user-emphasized content)
    quoted_pattern = r'["\']([^"\']+)["\']'
    quoted_matches = re.findall(quoted_pattern, text)
    keywords.extend([m.lower().strip() for m in quoted_matches if m.strip()])
    
    # Strategy 4: Extract compound terms
    compound_pattern = r'\b[a-z]+(?:-[a-z]+)+\b'
    compound_matches = re.findall(compound_pattern, text_lower)
    keywords.extend(compound_matches)
    
    # Strategy 5: Semantic keyword extraction - prioritize contextually important words
    words = re.findall(r'\b[a-z]+\b', text_lower)
    meaningful_words = [
        w for w in words 
        if len(w) >= min_length and w not in STOP_WORDS
    ]
    
    # Weight words by semantic importance and frequency
    if meaningful_words:
        word_freq = Counter(meaningful_words)
        
        # Define semantic importance weights
        semantic_weights = {
            # Technical terms (high importance)
            'architecture': 3, 'server': 3, 'api': 3, 'database': 3, 'system': 3,
            'integration': 3, 'authentication': 3, 'security': 3, 'protocol': 3,
            'framework': 3, 'component': 3, 'service': 3, 'endpoint': 3,
            
            # Business terms (medium-high importance)
            'project': 2, 'initiative': 2, 'strategy': 2, 'solution': 2,
            'platform': 2, 'product': 2, 'feature': 2, 'capability': 2,
            
            # Action terms (medium importance)
            'implement': 2, 'develop': 2, 'deploy': 2, 'build': 2,
            'create': 2, 'design': 2, 'configure': 2, 'setup': 2,
            
            # General important words (low-medium importance)
            'latest': 1.5, 'new': 1.5, 'current': 1.5, 'recent': 1.5,
            'demo': 1.5, 'example': 1.5, 'documentation': 1.5, 'guide': 1.5,
        }
        
        # Score words by frequency and semantic importance
        scored_words = []
        for word, count in word_freq.items():
            base_score = count
            semantic_multiplier = semantic_weights.get(word, 1.0)
            final_score = base_score * semantic_multiplier
            scored_words.append((word, final_score))
        
        # Sort by score and take top words
        scored_words.sort(key=lambda x: x[1], reverse=True)
        important_words = [word for word, score in scored_words[:max_keywords]]
        keywords.extend(important_words)
    
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_keywords: List[str] = []
    for kw in keywords:
        if kw and kw not in seen and len(kw) >= min_length:
            seen.add(kw)
            unique_keywords.append(kw)
    
    logger.info(f"Extracted {len(unique_keywords)} keywords: {unique_keywords[:10]}")
    return unique_keywords[:max_keywords]


def detect_source_preference(user_query: str) -> Dict[str, int]:
    """
    Detect which data sources the user wants to prioritize based on query content.
    
    Returns a dictionary mapping source names to priority scores (higher = more priority).
    Sources: slack, docs, confluence, zendesk, jira
    """
    query_lower = user_query.lower()
    source_scores = {
        "slack": 0,
        "docs": 0,
        "confluence": 0,
        "zendesk": 0,
        "jira": 0
    }
    
    # Slack indicators
    slack_patterns = [
        r'\bslack\b',
        r'\bchannel\b',
        r'\bmessage\b',
        r'\bconversation\b',
        r'\bdiscussion\b',
        r'\bannouncement\b',
        r'\bteam chat\b',
        r'#\w+',  # Channel mentions like #engineering
    ]
    for pattern in slack_patterns:
        if re.search(pattern, query_lower):
            source_scores["slack"] += 2
    
    # Docs indicators
    docs_patterns = [
        r'\bdocs?\b',
        r'\bdocumentation\b',
        r'\bknowledge base\b',
        r'\bkb\b',
        r'\bcommunity\b',
        r'\bsupport article\b',
        r'\bhelp article\b',
        r'\bguide\b',
        r'\btutorial\b',
    ]
    for pattern in docs_patterns:
        if re.search(pattern, query_lower):
            source_scores["docs"] += 2
    
    # Confluence indicators
    confluence_patterns = [
        r'\bconfluence\b',
        r'\bwiki\b',
        r'\bpage\b',
        r'\binternal doc\b',
        r'\bprocess doc\b',
        r'\bproject doc\b',
    ]
    for pattern in confluence_patterns:
        if re.search(pattern, query_lower):
            source_scores["confluence"] += 2
    
    # Zendesk indicators
    zendesk_patterns = [
        r'\bzendesk\b',
        r'\bticket\b',
        r'\bcustomer issue\b',
        r'\bsupport ticket\b',
        r'\bcustomer problem\b',
        r'\bcustomer report\b',
    ]
    for pattern in zendesk_patterns:
        if re.search(pattern, query_lower):
            source_scores["zendesk"] += 2
    
    # Jira indicators
    jira_patterns = [
        r'\bjira\b',
        r'\bissue\b',
        r'\bbug\b',
        r'\bfeature request\b',
        r'\bepic\b',
        r'\bstory\b',
        r'\btask\b',
        r'\broadmap\b',
        r'\bbacklog\b',
    ]
    for pattern in jira_patterns:
        if re.search(pattern, query_lower):
            source_scores["jira"] += 2
    
    # If no explicit source mentioned, return all zeros (no prioritization)
    max_score = max(source_scores.values())
    if max_score == 0:
        return source_scores
    
    logger.info(f"Source preference detected: {source_scores}")
    return source_scores


def detect_query_type(
    user_query: str, 
    conversation_history: List[Dict[str, str]],
    last_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect whether the query is:
    - new_search: Requires fresh data retrieval
    - follow_up: Can be answered from existing context
    - clarification: Asking for more details on previous answer
    """
    query_lower = user_query.lower().strip()
    
    # Follow-up indicators
    follow_up_patterns = [
        r'\b(what about|how about|tell me more|explain|elaborate|clarify|details?)\b',
        r'\b(in that|from (?:that|those|the above)|based on)\b',
        r'^(and|but|also|additionally|furthermore)\b',
        r'\b(you (?:mentioned|said|told|showed)|you just)\b',
        r'\b(the (?:previous|last|above) (?:message|response|answer|result))\b',
        r'^(why|how|when|where|who)\b(?!.*\b(?:latest|recent|new|update)\b)',
    ]
    
    # New search indicators
    new_search_patterns = [
        r'\b(latest|recent|new|current|today|yesterday|now)\b',
        r'\b(find|search|look|show|get|fetch)\b',
        r'\b(what is|what are|tell me about|information on)\b.*\b(?!that|those|it|them)\b',
        r'\b(in #?\w+|from #?\w+)\b',  # Mentions specific channel
        r'\b(update[sd]?|announcement[s]?|change[s]?|release[s]?)\b',
        r'\b(announcement|announce|release|version|update)\b',  # Release-related terms
    ]
    
    is_follow_up = any(re.search(pattern, query_lower) for pattern in follow_up_patterns)
    is_new_search = any(re.search(pattern, query_lower) for pattern in new_search_patterns)
    
    # Check if query references previous context
    has_pronoun_reference = bool(re.search(r'\b(that|those|it|this|these|them)\b', query_lower))
    
    # Short queries after a conversation are likely follow-ups
    is_short_query = len(query_lower.split()) <= 5
    has_recent_context = len(conversation_history) > 1
    
    # Decision logic
    if is_new_search and not is_follow_up:
        query_type = "new_search"
        confidence = 0.9
    elif is_follow_up or (has_pronoun_reference and has_recent_context):
        query_type = "follow_up"
        confidence = 0.8
    elif is_short_query and has_recent_context and last_context:
        query_type = "clarification"
        confidence = 0.7
    else:
        # Default to new search if uncertain
        query_type = "new_search"
        confidence = 0.5
    
    logger.info(f"Query type: {query_type} (confidence: {confidence})")
    
    return {
        "query_type": query_type,
        "confidence": confidence,
        "needs_fresh_data": query_type == "new_search",
        "can_use_cache": query_type in ["follow_up", "clarification"]
    }


def analyze_user_intent(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    last_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enhanced intent analysis with query type detection and smart keyword extraction.
    """
    if not user_query or not user_query.strip():
        return _get_default_intent()
    
    conversation_history = conversation_history or []
    
    # Step 1: Detect query type
    query_type_info = detect_query_type(user_query, conversation_history, last_context)
    
    # Step 1.5: Detect source preference
    source_preference = detect_source_preference(user_query)
    
    # Step 2: Extract smart keywords
    smart_keywords = extract_keywords_smart(user_query)
    
    # If it's a follow-up, we might not need full intent analysis
    if query_type_info["query_type"] == "follow_up" and last_context:
        return {
            "intent": "follow_up",
            "query_type": query_type_info["query_type"],
            "needs_fresh_data": False,
            "data_sources": [],
            "slack_params": {
                "channels": "all",
                "time_range": "all",
                "keywords": smart_keywords,
                "sort": "relevance",
                "limit": 5
            },
            "confluence_params": {
                "keywords": smart_keywords,
                "spaces": None,
                "limit": 3
            },
            "reasoning": "Follow-up question detected - can use existing context"
        }
    
    # Step 3: Build enhanced prompt with conversation context
    context_summary = ""
    if conversation_history and len(conversation_history) > 1:
        recent_messages = conversation_history[-4:]
        context_summary = "Recent conversation:\n" + "\n".join([
            f"{msg['role']}: {msg['content'][:100]}..." 
            for msg in recent_messages
        ]) + "\n\n"
    
    prompt = f"""
You are an AI assistant that analyzes user queries to extract search parameters for Slack and Confluence.

{context_summary}Current Query: "{user_query}"

Smart Keywords Already Extracted: {smart_keywords}

Analyze this query and return a JSON response with the following structure:

{{
    "intent": "latest_message|search_messages|mixed_search|confluence_only|slack_only|follow_up",
    "query_type": "{query_type_info['query_type']}",
    "needs_fresh_data": {str(query_type_info['needs_fresh_data']).lower()},
    "data_sources": ["slack", "confluence"],
    "slack_params": {{
        "channels": "all|specific_channel_name|list_of_channels",
        "time_range": "all",
        "keywords": ["use smart keywords + add important domain terms"],
        "sort": "relevance|timestamp",
        "limit": number,
        "priority_terms": ["critical search terms that MUST match"]
    }},
    "confluence_params": {{
        "keywords": ["use smart keywords + add documentation-relevant terms"],
        "spaces": "all|specific_space|list_of_spaces",
        "content_types": ["page", "blogpost"],
        "limit": number,
        "priority_terms": ["critical search terms that MUST match"]
    }},
    "search_strategy": "exact_match|fuzzy_match|semantic_search",
    "reasoning": "Brief explanation of the analysis"
}}

IMPORTANT INSTRUCTIONS:
1. Use the smart keywords already extracted, but ADD domain-specific terms if needed
2. For technical queries, prioritize exact matches for version numbers, IDs, and technical terms
3. For conceptual queries, allow fuzzy matching
4. Set priority_terms for MUST-MATCH keywords (version numbers, product names, specific topics)
5. If query mentions specific channels/spaces, extract them
6. Search ALL available history - never miss the right answer because it's old.
7. For follow-up questions, keep limit low (5-10) since context already exists

Intent Types:
- "latest_message": User wants the most recent message(s)
- "search_messages": User wants to search for specific content
- "mixed_search": Information could be in both Slack and Confluence
- "confluence_only": Documentation/knowledge base content
- "slack_only": Discussions/announcements
- "follow_up": Clarification on previous response

Search Strategies:
- "exact_match": For technical terms, IDs, versions (use priority_terms)
- "fuzzy_match": For general topics, concepts
- "semantic_search": For complex questions requiring understanding

Examples:
- "What is the latest in engineering?" → intent: "latest_message", channels: "engineering", time_range: "all"
- "Find info about version 2024.1.2" → priority_terms: ["2024.1.2"], search_strategy: "exact_match"
- "Incorta MCO architecture?" → keywords: ["incorta", "mco", "architecture"], mixed_search
- "latest release announcement" → intent: "latest_message", keywords: ["latest", "release", "announcement"], priority_terms: ["release", "announcement"]
- "Search in Slack for the delivery date of Incorta release 25.7.2" → intent: "specific_info", keywords: ["delivery", "date", "incorta", "release", "25.7.2"], priority_terms: ["25.7.2", "delivery", "date", "release"]
- "Tell me more about that" → intent: "follow_up", needs_fresh_data: false

Return ONLY the JSON response.
"""

    # Step 4: Initialize API manager if not already done
    global _api_manager
    if _api_manager is None:
        try:
            from ..api_manager import create_api_manager_from_env
            _api_manager = create_api_manager_from_env()
            logger.info(f"Intent analyzer: Using API manager with {len(_api_manager.api_keys)} key(s)")
        except Exception as e:
            logger.warning(f"Intent analyzer: Failed to create API manager: {e}, using single key")
            _api_manager = None
    
    # Step 5: Generate intent analysis with key rotation support
    # Use API manager's execute_with_retry for automatic key rotation on quota errors
    if _api_manager:
        def _generate_intent():
            configure_genai()  # Ensure genai is configured with current key
            model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-09-2025")  # Use same model as other modules
            response = model.generate_content(prompt)
            return response.text.strip()
        
        try:
            # Use API manager's retry logic with key rotation
            response_text = _api_manager.execute_with_retry(
                _generate_intent,
                max_retries=len(_api_manager.api_keys),
                retry_delay=1
            )
        except Exception as e:
            logger.error(f"Intent analysis failed after trying all keys: {e}")
            # Fall back to single key attempt
            configure_genai()
            model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-09-2025")
            response = model.generate_content(prompt)
            response_text = response.text.strip()
    else:
        # Single key mode
        configure_genai()
        model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-09-2025")  # Use same model as other modules
        response = model.generate_content(prompt)
        response_text = response.text.strip()
    
    # Clean JSON markers
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    if response_text.startswith('```'):
        response_text = response_text[3:]
    response_text = response_text.strip()
    
    try:
        intent_data = json.loads(response_text)
        
        # Merge smart keywords with AI-extracted keywords
        if smart_keywords:
            slack_kw = intent_data.get("slack_params", {}).get("keywords", [])
            conf_kw = intent_data.get("confluence_params", {}).get("keywords", [])
            
            # Deduplicate and combine
            intent_data["slack_params"]["keywords"] = list(set(slack_kw + smart_keywords))
            intent_data["confluence_params"]["keywords"] = list(set(conf_kw + smart_keywords))
        
        # Add query type info
        intent_data["query_type"] = query_type_info["query_type"]
        intent_data["needs_fresh_data"] = query_type_info["needs_fresh_data"]
        
        # Add source preference
        intent_data["source_preference"] = source_preference
        
        logger.info(f"Intent analysis result: {intent_data}")
        return intent_data
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from Gemini: {e}")
        logger.warning(f"Raw response: {response_text}")
        return _get_default_intent_with_keywords(smart_keywords)
    except Exception as e:
        logger.error(f"Intent analysis failed: {e}")
        return _get_default_intent_with_keywords(smart_keywords)


def _get_default_intent() -> Dict[str, Any]:
    """Return a default intent when analysis fails."""
    return {
        "intent": "mixed_search",
        "query_type": "new_search",
        "needs_fresh_data": True,
        "data_sources": ["slack", "confluence"],
        "source_preference": {"slack": 0, "docs": 0, "confluence": 0, "zendesk": 0, "jira": 0},
        "slack_params": {
            "channels": "all",
            "time_range": "all",  # No time restrictions - search all available history
            "keywords": [],
            "sort": "relevance",
            "limit": 25,  # Increased limit for better Slack coverage
            "priority_terms": []
        },
        "confluence_params": {
            "keywords": [],
            "spaces": "all",  # Search ALL spaces, not just None
            "limit": 10,  # Reduced limit to give more weight to Slack
            "priority_terms": []
        },
        "search_strategy": "fuzzy_match",
        "reasoning": "Default fallback intent"
    }


def _get_default_intent_with_keywords(keywords: List[str]) -> Dict[str, Any]:
    """Return a default intent with extracted keywords."""
    intent = _get_default_intent()
    intent["slack_params"]["keywords"] = keywords
    intent["confluence_params"]["keywords"] = keywords
    intent["reasoning"] = "Default intent with smart keyword extraction"
    return intent


def validate_intent(intent_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize intent data."""
    
    # Ensure required fields
    intent_data.setdefault("intent", "mixed_search")
    intent_data.setdefault("query_type", "new_search")
    intent_data.setdefault("needs_fresh_data", True)
    intent_data.setdefault("data_sources", ["slack", "confluence"])
    intent_data.setdefault("search_strategy", "fuzzy_match")
    
    # Validate slack_params
    slack_params = intent_data.setdefault("slack_params", {})
    slack_params.setdefault("channels", "all")
    slack_params.setdefault("time_range", "all")  # No time restrictions - search all history
    slack_params.setdefault("limit", 25)  # Increased default limit for comprehensive search
    slack_params.setdefault("sort", "relevance")
    slack_params.setdefault("keywords", [])
    slack_params.setdefault("priority_terms", [])
    
    # Validate confluence_params
    conf_params = intent_data.setdefault("confluence_params", {})
    conf_params.setdefault("spaces", "all")  # Search all spaces by default
    conf_params.setdefault("limit", 15)  # Increased default limit for comprehensive search
    conf_params.setdefault("keywords", [])
    conf_params.setdefault("priority_terms", [])
    
    return intent_data


# Backward compatibility - keep the old function name
def analyze_intent(query: str) -> Dict[str, object]:
    """
    Legacy function for backward compatibility.
    Calls analyze_user_intent with no conversation history.
    """
    result = analyze_user_intent(query)
    # Convert to old format if needed
    return result


__all__ = [
    "analyze_user_intent", 
    "analyze_intent",  # Backward compatibility
    "extract_keywords_smart",
    "detect_query_type",
    "detect_source_preference",
    "validate_intent",
    "configure_genai"
]
