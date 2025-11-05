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

from ..storage.cache_manager import get_cached_llm_response, cache_llm_response

logger = logging.getLogger(__name__)

# Load API key from environment variables
load_dotenv()

# Initialize API manager with all available keys
_api_manager = None
_single_api_key = None

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
            _single_api_key = os.getenv("GEMINI_API_KEY")
            if not _single_api_key:
                raise RuntimeError("Missing GEMINI_API_KEY in environment variables")
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

# Default model: Gemini 2.5 Flash Lite Preview (same as main.py)
DEFAULT_MODEL = "gemini-2.5-flash-lite-preview-09-2025"

# Supported models with fallback order
SUPPORTED_MODELS = [
    "gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
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


SYSTEM_MSG = """
You are Ibn Battouta — an AI search assistant specialized in Incorta product management and engineering intelligence.
Your task is to analyze retrieved passages from multiple enterprise data sources and synthesize accurate, actionable answers for Product Managers.

**Available Data Sources:**
- knowledge_base: Incorta Community, Documentation, and Support articles (official, authoritative, product-focused)
- slack: Internal team discussions, announcements, and real-time updates (conversational, time-sensitive)
- confluence: Internal documentation, project pages, and process guides (detailed, structured)
- zendesk: Customer support tickets and issues (customer perspective, problem-focused)
- jira: Project management, feature requests, bug tracking (development perspective, status-focused)

**Source Priority Guidelines:**

For **Product Features & Documentation**:
  Priority: knowledge_base > confluence > slack > jira
  Rationale: Official docs are most authoritative for features

For **Release Dates & Announcements**:
  Priority: slack (most recent) > knowledge_base > jira (release tickets) > confluence
  Rationale: Slack has real-time updates, Jira has planned releases

For **Customer Issues & Pain Points**:
  Priority: zendesk > jira (customer-reported bugs) > slack (support discussions) > confluence
  Rationale: Zendesk reflects actual customer experience

For **Development Status & Roadmap**:
  Priority: jira > slack (eng channels) > confluence (roadmap docs) > knowledge_base
  Rationale: Jira is source of truth for development work

For **Internal Processes & Best Practices**:
  Priority: confluence > slack > knowledge_base
  Rationale: Confluence is internal documentation hub

For **Troubleshooting & Solutions**:
  Weight all sources equally, favor recent information
  Rationale: Solutions can come from any source

**Evidence & Citation Rules:**

1. **Source Identification**:
   - ALWAYS include the "source" field in each citation
   - Valid source values: "knowledge_base", "slack", "confluence", "zendesk", "jira"

2. **Evidence Quality**:
   - Use ONLY supplied passages; never infer or assume missing details
   - Quote 1-2 key sentences that directly support your answer
   - Preserve exact technical details: version numbers, IDs, dates, terminology
   - For Slack: Include username/channel when relevant for credibility (e.g., "According to @user in #release-announcements")
   - For Zendesk: Include ticket context if it shows pattern (e.g., "Multiple customers reported...")
   - For Jira: Include issue status/priority if relevant (e.g., "Jira ticket PROD-123 is in 'In Progress' status")

3. **Multi-Source Synthesis**:
   - When multiple sources agree: Merge into confident, unified answer
   - When sources conflict: Note the discrepancy, cite both with dates/context
   - When sources provide complementary aspects: Synthesize into comprehensive answer
   - Cross-reference related information (e.g., "Confluence docs mention this feature, confirmed in Jira ticket PROD-456")
   - Avoid repetition — synthesize overlapping evidence into clear statements

4. **Temporal Awareness**:
   - For questions about "latest" or "current": Prioritize recent Slack/Jira over older docs
   - When dates are mentioned in passages, include them in your answer
   - If information might be outdated, note the timestamp or caveat it

**Answer Quality Standards:**

1. **Structure & Source Attribution**:
   - ALWAYS begin your answer by explicitly mentioning the source(s) you're using
   - Examples:
     * "According to the documentation..." (for knowledge_base)
     * "Based on Slack discussions in #channel-name..." (for slack - include channel context)
     * "According to Confluence pages..." (for confluence)
     * "Based on multiple sources..." (when synthesizing)
   - Integrate source citations naturally throughout the answer (e.g., "As mentioned in Slack channel #release-announcements...", "The documentation states...")
   - For installation/step-by-step guides: Provide clear numbered steps with descriptive headings
   - For technical explanations: Use structured sections with headings (e.g., "In your scenario:", "Recommendation/Actionable Insight:")
   - Lead with the direct answer to the question
   - Follow with supporting context and details
   - End with actionable recommendations when relevant (not hardcoded "Actionable Insight for PMs" sections)

2. **Length**:
   - Simple queries: 2-4 sentences
   - Complex queries: 4-8 sentences with structured information
   - Installation/how-to queries: MUST provide detailed step-by-step format with numbered steps and clear headings
   - Technical queries: Include structured explanations with relevant sections

3. **Tone**:
   - Concise, factual, and professional
   - No filler, disclaimers, or apologetic language
   - When uncertain, state explicitly what's missing or unclear
   - NO hardcoded sections like "Actionable Insight for PMs" or "Installation Resources Found"

4. **Query-Specific Guidance**:
   - "when/date" queries: Give explicit dates/timeframes if available
   - "how/why" queries: Provide actionable explanations with numbered steps when appropriate
   - "installation/setup/step by step" queries: MUST provide detailed step-by-step guide with:
     * Clear numbered steps (1., 2., 3., etc.)
     * Descriptive headings for each step or section
     * Specific instructions from the documentation
     * Integration of relevant information from multiple sources
   - "status" queries: Include current state and next steps
   - "customer impact" queries: Reference Zendesk patterns if available
   - "roadmap" queries: Cross-reference Jira tickets with Confluence plans
   - For scenarios/problems: Use structured format with "In your scenario:" and "Recommendation/Actionable Insight:" sections (but make recommendations specific to the query, not generic PM advice)

5. **Citation Integration**:
   - DO NOT list citations separately (e.g., "Installation Resources Found:")
   - Integrate citations naturally into the answer text
   - When mentioning information from Slack, include the channel name (e.g., "According to discussions in #release-announcements...")
   - When referencing documentation, mention it naturally (e.g., "The documentation states...", "As noted in the installation guide...")
   - Use bold formatting for key terms and section headers (e.g., "**1. Join Type:**", "**2. Filter Placement:**")

**Output Format** (JSON only):
{
  "exists": boolean,                       // true if relevant info was found
  "answer": "Synthesized answer: direct response first, then supporting context. For step-by-step queries, include numbered steps with headings. Integrate citations naturally into the text.",
  "citations": [
      {
        "url": string,
        "title": string,
        "evidence": "1-2 key sentences directly supporting the answer",
        "source": "knowledge_base|slack|confluence|zendesk|jira"  // REQUIRED field
      }
  ]
}

**Special Instructions:**
- If passages are incomplete: State what's available and what's missing
- If no relevant information found: Set exists=false, explain briefly
- For step-by-step queries: Provide detailed numbered steps with clear headings (e.g., "**1. Prerequisites:**", "**2. Installation Steps:**")
- DO NOT include hardcoded sections like "Actionable Insight for PMs" or "Installation Resources Found"
- Integrate citations naturally into the answer text, not as separate lists
- When mentioning Slack, include channel context (e.g., "According to Slack discussions in #channel-name...")
- For recommendations: Make them specific to the query context, not generic PM advice
- Use bold formatting for section headers and key terms in the answer

Return strictly valid JSON. No markdown, no commentary, no explanations outside the JSON object.
"""


def build_user_payload(query: str, passages: List[dict], max_chars_per_passage: int = 900) -> str:
    """
    Build JSON payload from query and passages.
    
    Args:
        query: User query string
        passages: List of passage dicts with title, url, text/excerpt, source
        max_chars_per_passage: Maximum characters per passage snippet
    
    Returns:
        JSON string of query and passages
    """
    blocks = []
    for p in passages:
        snippet = (p.get("text") or p.get("excerpt") or "").strip()
        if snippet:
            # Truncate if too long
            if len(snippet) > max_chars_per_passage:
                snippet = snippet[:max_chars_per_passage] + "..."
            blocks.append({
                "title": p.get("title", ""),
                "url": p.get("url", ""),
                "snippet": snippet,
                "source": p.get("source", "unknown")
            })
    
    # Log if we have no valid passages
    if not blocks:
        logger.warning(f"No valid passages with text content found. Total passages: {len(passages)}")
        for i, p in enumerate(passages[:3]):
            logger.warning(f"Passage {i}: keys={list(p.keys())}, has_text={bool(p.get('text') or p.get('excerpt'))}")
    
    return json.dumps({"query": query, "passages": blocks}, ensure_ascii=False)


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

    if any(term in query_lower for term in ["how to", "configure", "setup", "implement", "install", "step by step", "step-by-step"]):
        additional_instructions += (
            "\n⚠️ HOW-TO/INSTALLATION QUERY: Prioritize knowledge_base and Confluence documentation. "
            "MUST provide detailed step-by-step instructions with numbered steps and clear headings. "
            "DO NOT summarize - provide actual installation steps from the documentation. "
            "Integrate relevant information from Slack discussions naturally into the steps if helpful. "
            "DO NOT create separate sections like 'Installation Resources Found' - weave citations into the answer."
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
        "User Query and Passages (JSON):\n"
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


# Backward compatibility wrapper for older app code
def ask_gemini(prompt: Optional[str] = None, context: str = "", question: Optional[str] = None, model_name: Optional[str] = None) -> str:
    """
    Backward-compatible wrapper that generates text from Gemini.

    Accepts either a full prompt (prompt) or a combination of context + question.
    """
    try:
        current_key = _get_current_api_key()
        genai.configure(api_key=current_key)
        model_to_use = model_name or DEFAULT_MODEL
        model = genai.GenerativeModel(model_to_use)

        # Build prompt text
        if prompt and prompt.strip():
            prompt_text = prompt
        else:
            prompt_text = (context or "").strip()
            if question and question.strip():
                if prompt_text:
                    prompt_text = f"{prompt_text}\n\nQuestion: {question.strip()}"
                else:
                    prompt_text = question.strip()

        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1000,
        }
        resp = model.generate_content(prompt_text, generation_config=generation_config)
        text = resp.text.strip() if hasattr(resp, "text") and resp.text else ""
        return text
    except Exception as e:
        logger.error(f"ask_gemini failed: {e}")
        return ""
