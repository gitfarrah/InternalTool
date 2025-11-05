from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import streamlit as st

from .channel_intelligence import get_channel_intelligence  # Use external module

logger = logging.getLogger(__name__)


@dataclass
class ChannelInfo:
    """Information about a Slack channel."""
    id: str
    name: str
    purpose: str
    topic: str
    is_private: bool
    is_member: bool
    member_count: int
    keywords: Set[str]
    patterns: Set[str]
    category: Optional[str] = None


@dataclass
class ChannelPattern:
    """Detected pattern in channel naming."""
    pattern: str  # e.g., "data-*", "team-*"
    regex: re.Pattern
    channels: Set[str]
    category: str
    confidence: float


class ChannelIntelligence:
    """
    Intelligent channel analysis and filtering for Slack workspaces.
    
    This class analyzes all accessible channels to:
    1. Discover channel patterns and categories
    2. Extract keywords from names, purposes, and topics
    3. Build an index for fast relevant channel lookup
    4. Cache results in session state for performance
    """
    
    def __init__(self, client: WebClient):
        self.client = client
        self._lock = threading.Lock()
        self._channels: Dict[str, ChannelInfo] = {}
        self._patterns: List[ChannelPattern] = []
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
        self._initialized = False
    
    def _get_slack_client(self) -> WebClient:
        """Get Slack client from session state or create new one."""
        if "slack_client" not in st.session_state:
            token = st.secrets.get("SLACK_USER_TOKEN")
            if not token:
                raise RuntimeError("Missing SLACK_USER_TOKEN environment variable.")
            st.session_state["slack_client"] = WebClient(token=token)
        return st.session_state["slack_client"]
    
    def initialize(self, force_refresh: bool = False) -> None:
        """
        Initialize channel intelligence by analyzing all accessible channels.
        
        Args:
            force_refresh: If True, re-analyze even if cached data exists
        """
        with self._lock:
            if self._initialized and not force_refresh:
                return
            
            # Check session state cache first
            cache_key = "channel_intelligence_cache"
            if not force_refresh and cache_key in st.session_state:
                cached_data = st.session_state[cache_key]
                self._channels = cached_data.get("channels", {})
                self._patterns = cached_data.get("patterns", [])
                self._keyword_index = defaultdict(set, cached_data.get("keyword_index", {}))
                self._category_index = defaultdict(set, cached_data.get("category_index", {}))
                self._initialized = True
                logger.info(f"Loaded cached channel intelligence for {len(self._channels)} channels")
                return
            
            logger.info("Initializing channel intelligence...")
            
            # Discover all channels
            self._discover_channels()
            
            # Analyze patterns
            self._detect_patterns()
            
            # Build keyword index
            self._build_keyword_index()
            
            # Cache in session state
            st.session_state[cache_key] = {
                "channels": self._channels,
                "patterns": self._patterns,
                "keyword_index": dict(self._keyword_index),
                "category_index": dict(self._category_index)
            }
            
            self._initialized = True
            logger.info(f"Channel intelligence initialized: {len(self._channels)} channels, {len(self._patterns)} patterns")
    
    def _discover_channels(self) -> None:
        """Discover all accessible channels and extract basic information."""
        logger.info("Discovering accessible channels...")
        
        try:
            next_cursor: Optional[str] = None
            total_channels = 0
            
            while True:
                response = self.client.conversations_list(
                    types="public_channel,private_channel",
                    exclude_archived=True,
                    limit=1000,
                    cursor=next_cursor or None,
                )
                
                channels = response.get("channels", [])
                if not channels:
                    break
                
                for channel in channels:
                    channel_id = channel.get("id")
                    if not channel_id:
                        continue
                    
                    # Skip DMs and group DMs
                    if channel.get("is_im") or channel.get("is_mpim"):
                        continue
                    
                    # Skip excluded channels
                    channel_name = channel.get("name", "").lower()
                    if channel_name == "inhousezendesk":
                        continue
                    
                    # Only include channels user is member of (for private) or all public
                    is_private = channel.get("is_private", False)
                    is_member = channel.get("is_member", False)
                    
                    if is_private and not is_member:
                        continue
                    
                    # Extract channel information
                    name = channel.get("name", "")
                    purpose = channel.get("purpose", {}).get("value", "")
                    topic = channel.get("topic", {}).get("value", "")
                    member_count = channel.get("num_members", 0)
                    
                    # Extract keywords from name, purpose, and topic
                    keywords = self._extract_keywords_from_text(f"{name} {purpose} {topic}")
                    
                    channel_info = ChannelInfo(
                        id=channel_id,
                        name=name,
                        purpose=purpose,
                        topic=topic,
                        is_private=is_private,
                        is_member=is_member,
                        member_count=member_count,
                        keywords=keywords,
                        patterns=set()
                    )
                    
                    self._channels[channel_id] = channel_info
                    total_channels += 1
                
                next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
                if not next_cursor:
                    break
            
            logger.info(f"Discovered {total_channels} accessible channels")
            
        except SlackApiError as e:
            logger.error(f"Failed to discover channels: {e}")
            raise
    
    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return set()
        
        # Common stopwords to filter out
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "about", "what", "when",
            "where", "why", "how", "who", "which", "this", "that", "these", "those",
            "channel", "chat", "discussion", "team", "group", "project"
        }
        
        # Extract words (alphanumeric + hyphens)
        words = re.findall(r'\b[a-zA-Z0-9-]+\b', text.lower())
        
        # Filter and clean
        keywords = set()
        for word in words:
            # Skip stopwords and very short words
            if word in stopwords or len(word) < 2:
                continue
            
            # Skip pure numbers
            if word.isdigit():
                continue
            
            # Clean up the word
            clean_word = word.strip('-')
            if len(clean_word) >= 2:
                keywords.add(clean_word)
        
        return keywords
    
    def _detect_patterns(self) -> None:
        """Detect naming patterns in channels and categorize them."""
        logger.info("Detecting channel patterns...")
        
        # Group channels by common prefixes
        prefix_groups = defaultdict(list)
        suffix_groups = defaultdict(list)
        
        for channel_id, channel_info in self._channels.items():
            name = channel_info.name.lower()
            
            # Extract prefixes (e.g., "data-", "team-", "eng-")
            for i in range(2, len(name)):
                if name[i] == '-':
                    prefix = name[:i+1]
                    prefix_groups[prefix].append(channel_id)
                    break
            
            # Extract suffixes (e.g., "-team", "-dev", "-prod")
            for i in range(len(name)-2, 0, -1):
                if name[i] == '-':
                    suffix = name[i:]
                    suffix_groups[suffix].append(channel_id)
                    break
        
        # Create patterns from significant groups
        self._patterns = []
        
        # Process prefixes
        for prefix, channel_ids in prefix_groups.items():
            if len(channel_ids) >= 2:  # At least 2 channels with this pattern
                category = self._infer_category_from_pattern(prefix, channel_ids)
                confidence = min(1.0, len(channel_ids) / 10.0)  # Higher confidence for more channels
                
                pattern = ChannelPattern(
                    pattern=prefix + "*",
                    regex=re.compile(f"^{re.escape(prefix[:-1])}-", re.IGNORECASE),
                    channels=set(channel_ids),
                    category=category,
                    confidence=confidence
                )
                self._patterns.append(pattern)
                
                # Update channel info
                for channel_id in channel_ids:
                    self._channels[channel_id].patterns.add(pattern.pattern)
                    self._channels[channel_id].category = category
        
        # Process suffixes
        for suffix, channel_ids in suffix_groups.items():
            if len(channel_ids) >= 2:
                category = self._infer_category_from_pattern(suffix, channel_ids)
                confidence = min(1.0, len(channel_ids) / 10.0)
                
                pattern = ChannelPattern(
                    pattern="*" + suffix,
                    regex=re.compile(f"{re.escape(suffix[1:])}$", re.IGNORECASE),
                    channels=set(channel_ids),
                    category=category,
                    confidence=confidence
                )
                self._patterns.append(pattern)
                
                # Update channel info
                for channel_id in channel_ids:
                    self._channels[channel_id].patterns.add(pattern.pattern)
                    if not self._channels[channel_id].category:
                        self._channels[channel_id].category = category
        
        # Sort patterns by confidence
        self._patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        logger.info(f"Detected {len(self._patterns)} channel patterns")
    
    def _infer_category_from_pattern(self, pattern: str, channel_ids: List[str]) -> str:
        """Infer category from pattern and channel names."""
        # Get sample channel names for this pattern
        sample_names = [self._channels[cid].name.lower() for cid in channel_ids[:5]]
        sample_text = " ".join(sample_names)
        
        # Category mapping based on common patterns
        category_keywords = {
            "engineering": ["eng", "dev", "development", "backend", "frontend", "api", "code", "tech"],
            "data": ["data", "analytics", "metrics", "dashboard", "viz", "visualization", "reporting"],
            "product": ["product", "feature", "roadmap", "planning", "strategy"],
            "design": ["design", "ui", "ux", "mockup", "prototype", "wireframe"],
            "marketing": ["marketing", "campaign", "content", "social", "brand"],
            "sales": ["sales", "revenue", "customer", "lead", "prospect"],
            "support": ["support", "help", "customer", "ticket", "issue"],
            "operations": ["ops", "operations", "infrastructure", "deployment", "monitoring"],
            "security": ["security", "auth", "compliance", "audit", "privacy"],
            "qa": ["qa", "test", "testing", "quality", "validation"],
            "project": ["project", "initiative", "milestone", "delivery"],
            "team": ["team", "squad", "group", "department"],
            "release": ["release", "announcement", "announce", "version", "update", "deploy", "deployment", "launch"]
        }
        
        # Find best matching category
        best_category = "general"
        best_score = 0
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in sample_text)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def _build_keyword_index(self) -> None:
        """Build keyword index for fast channel lookup."""
        logger.info("Building keyword index...")
        
        for channel_id, channel_info in self._channels.items():
            # Index all keywords from the channel
            for keyword in channel_info.keywords:
                self._keyword_index[keyword].add(channel_id)
            
            # Index category
            if channel_info.category:
                self._category_index[channel_info.category].add(channel_id)
            
            # Index channel name words
            name_words = self._extract_keywords_from_text(channel_info.name)
            for word in name_words:
                self._keyword_index[word].add(channel_id)
        
        logger.info(f"Built keyword index with {len(self._keyword_index)} keywords")
    
    def get_relevant_channels(
        self, 
        query: str, 
        top_k: int = 20,
        keywords: Optional[List[str]] = None,
        priority_terms: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get the most relevant channels for a query.
        
        Args:
            query: Search query
            top_k: Maximum number of channels to return
            keywords: Additional keywords to consider
            priority_terms: High-priority terms that must match
            
        Returns:
            List of channel IDs ordered by relevance
        """
        if not self._initialized:
            self.initialize()
        
        if not query and not keywords and not priority_terms:
            # Return most active channels if no specific query
            return self._get_most_active_channels(top_k)
        
        # Extract keywords from query if not provided
        if not keywords:
            keywords = list(self._extract_keywords_from_text(query))
        
        if priority_terms:
            keywords.extend(priority_terms)
        
        # Score channels based on keyword matches
        channel_scores = defaultdict(float)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Direct keyword matches (highest priority)
            if keyword_lower in self._keyword_index:
                for channel_id in self._keyword_index[keyword_lower]:
                    channel_scores[channel_id] += 2.0  # Increased weight
            
            # Pattern matches
            for pattern in self._patterns:
                if pattern.regex.search(keyword_lower):
                    for channel_id in pattern.channels:
                        channel_scores[channel_id] += pattern.confidence * 1.0  # Increased weight
            
            # Fuzzy matches (partial keyword matching)
            for indexed_keyword, channel_ids in self._keyword_index.items():
                if keyword_lower in indexed_keyword or indexed_keyword in keyword_lower:
                    for channel_id in channel_ids:
                        channel_scores[channel_id] += 0.5  # Increased weight
            
            # Channel name matching (very important for specific channels)
            for channel_id, channel_info in self._channels.items():
                channel_name = channel_info.name.lower()
                if keyword_lower in channel_name:
                    channel_scores[channel_id] += 3.0  # High weight for channel name matches
                elif any(word in channel_name for word in keyword_lower.split()):
                    channel_scores[channel_id] += 1.5  # Medium weight for partial matches
        
        # Boost scores for priority terms
        if priority_terms:
            for term in priority_terms:
                term_lower = term.lower()
                if term_lower in self._keyword_index:
                    for channel_id in self._keyword_index[term_lower]:
                        channel_scores[channel_id] += 2.0  # Higher boost for priority terms
        
        # Sort by score and return top channels
        sorted_channels = sorted(
            channel_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return channel IDs
        relevant_channels = [channel_id for channel_id, score in sorted_channels[:top_k]]
        
        logger.info(f"Found {len(relevant_channels)} relevant channels for query: {query}")
        return relevant_channels
    
    def _get_most_active_channels(self, top_k: int) -> List[str]:
        """Get the most active channels (by member count)."""
        sorted_channels = sorted(
            self._channels.items(),
            key=lambda x: x[1].member_count,
            reverse=True
        )
        return [channel_id for channel_id, _ in sorted_channels[:top_k]]
    
    def get_channel_info(self, channel_id: str) -> Optional[ChannelInfo]:
        """Get information about a specific channel."""
        if not self._initialized:
            self.initialize()
        return self._channels.get(channel_id)
    
    def get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        """
        Find a channel ID by channel name (case-insensitive).
        
        Args:
            channel_name: The name of the channel to find (with or without #)
        
        Returns:
            The channel ID if found, None otherwise
        """
        if not self._initialized:
            self.initialize()
        
        # Remove # prefix if present
        search_name = channel_name.lstrip('#').lower().strip()
        
        # Search for exact match first
        for channel_id, channel_info in self._channels.items():
            if channel_info.name.lower() == search_name:
                return channel_id
        
        # If no exact match, try partial match
        for channel_id, channel_info in self._channels.items():
            if search_name in channel_info.name.lower():
                return channel_id
        
        return None
    
    def find_similar_channels(self, channel_name: str, max_suggestions: int = 5) -> List[str]:
        """
        Find channels with similar names to the given channel name.
        
        Args:
            channel_name: The channel name to search for (with or without #)
            max_suggestions: Maximum number of suggestions to return
        
        Returns:
            List of similar channel names (without # prefix)
        """
        if not self._initialized:
            self.initialize()
        
        search_name = channel_name.lstrip('#').lower().strip()
        suggestions = []
        
        # Find channels containing any word from the search term
        search_words = search_name.split('_')
        
        for channel_id, channel_info in self._channels.items():
            channel_lower = channel_info.name.lower()
            
            # Calculate similarity score
            score = 0
            
            # Exact match (shouldn't happen if this is called, but just in case)
            if channel_lower == search_name:
                score = 100
            # Partial match
            elif search_name in channel_lower:
                score = 80
            # Check if any search words are in the channel name
            else:
                for word in search_words:
                    if word in channel_lower:
                        score += 20
            
            if score > 0:
                suggestions.append((channel_info.name, score))
        
        # Sort by score (highest first) and return top N
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in suggestions[:max_suggestions]]
    
    def get_patterns(self) -> List[ChannelPattern]:
        """Get all detected channel patterns."""
        if not self._initialized:
            self.initialize()
        return self._patterns
    
    def get_categories(self) -> Dict[str, int]:
        """Get channel categories and their counts."""
        if not self._initialized:
            self.initialize()
        
        category_counts = defaultdict(int)
        for channel_info in self._channels.values():
            if channel_info.category:
                category_counts[channel_info.category] += 1
        
        return dict(category_counts)
    
    def search_channels_by_name(self, name_pattern: str) -> List[str]:
        """Search channels by name pattern."""
        if not self._initialized:
            self.initialize()
        
        pattern = re.compile(name_pattern, re.IGNORECASE)
        matching_channels = []
        
        for channel_id, channel_info in self._channels.items():
            if pattern.search(channel_info.name):
                matching_channels.append(channel_id)
        
        return matching_channels


def get_channel_intelligence() -> ChannelIntelligence:
    """Get or create ChannelIntelligence instance with caching."""
    if "channel_intelligence" not in st.session_state:
        token = st.secrets.get("SLACK_USER_TOKEN")
        if not token:
            raise RuntimeError("Missing SLACK_USER_TOKEN environment variable.")
        client = WebClient(token=token)
        st.session_state["channel_intelligence"] = ChannelIntelligence(client)
    
    intelligence = st.session_state["channel_intelligence"]
    intelligence.initialize()
    return intelligence

# Removed duplicate future import and redundant re-imports
# ============================================================================
# VERSION/RELEASE PATTERN RECOGNITION
# ============================================================================

def _parse_version_number(query: str) -> Optional[Dict[str, Any]]:
    """
    Intelligently parse version numbers from query.
    
    Supports multiple formats:
    - "25.7.2" → {year: 2025, major: 7, minor: 2, original: "25.7.2"}
    - "2025.10.0" → {year: 2025, major: 10, minor: 0, original: "2025.10.0"}
    - "v25.7" → {year: 2025, major: 7, minor: None, original: "v25.7"}
    - "version 2024.5.3" → {year: 2024, major: 5, minor: 3, original: "2024.5.3"}
    
    Returns:
        Dict with parsed version info or None if no version found
    """
    version_patterns = [
        # Pattern: 2024.7.5 (full year format)
        r'(?:version\s+)?(\d{4})\.(\d+)\.(\d+)',
        # Pattern: v25.7.2 or 25.7.2 (short year format)
        r'(?:v|version\s+)?(\d{2})\.(\d+)\.(\d+)',
        # Pattern: 2024.10 or v25.7 (no patch version)
        r'(?:v|version\s+)?(\d{2,4})\.(\d+)(?:\.\d+)?',
    ]
    
    for pattern in version_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            groups = match.groups()
            year_part = int(groups[0])
            major = int(groups[1])
            minor = int(groups[2]) if len(groups) > 2 and groups[2] else None
            
            # Convert 2-digit year to 4-digit year (25 → 2025, 24 → 2024)
            if year_part < 100:
                year = 2000 + year_part
            else:
                year = year_part
            
            # Validate year is reasonable (between 2020-2030)
            if 2020 <= year <= 2030:
                return {
                    "year": year,
                    "major": major,
                    "minor": minor,
                    "original": match.group(0),
                    "pattern": pattern
                }
    
    return None


def _detect_release_query(query: str) -> bool:
    """
    Detect if query is about releases/versions/deliveries.
    
    Looks for keywords like: release, delivery, version, update, announcement, etc.
    """
    release_keywords = [
        r'\brelease\b', r'\bdelivery\b', r'\bversion\b', r'\bupdate\b',
        r'\bannouncement\b', r'\bannounce\b', r'\bdelivered?\b', r'\bshipped?\b',
        r'\bga\b', r'\bgeneral availability\b', r'\blaunch\b', r'\bdeploy\b'
    ]
    
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in release_keywords)


def _filter_messages_by_year(messages: List[Dict], year: int) -> List[Dict]:
    """
    Filter messages by year extracted from version number.
    
    Args:
        messages: List of Slack messages
        year: Year to filter by (e.g., 2025)
    
    Returns:
        Filtered list of messages from the specified year
    """
    filtered = []
    for msg in messages:
        ts = msg.get("ts")
        if ts:
            try:
                timestamp = float(ts)
                msg_datetime = datetime.fromtimestamp(timestamp)
                if msg_datetime.year == year:
                    filtered.append(msg)
            except (ValueError, TypeError):
                continue
    
    logger.info(f"Filtered {len(filtered)} messages from year {year} (out of {len(messages)} total)")
    return filtered


def _get_release_channels(intelligence) -> List[str]:
    """
    Get channels related to releases and announcements.
    
    Uses channel intelligence to find channels with keywords:
    - release, announcement, deploy, update, version, etc.
    """
    release_keywords = [
        "release", "announcement", "announce", "deploy", "deployment",
        "update", "version", "ga", "launch", "delivery"
    ]
    
    # Use channel intelligence to find relevant channels
    channel_ids = intelligence.get_relevant_channels(
        query="release announcement deployment",
        top_k=15,
        keywords=release_keywords,
        priority_terms=["release", "announcement"]
    )
    
    logger.info(f"Identified {len(channel_ids)} release-related channels")
    return channel_ids


# ============================================================================
# THREAD-FOCUSED SEARCH STRATEGY
# ============================================================================

def _search_message_thread(
    client: WebClient,
    channel_id: str,
    thread_ts: str,
    search_terms: List[str],
    version_info: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search all replies in a specific thread.
    
    This is critical for finding answers in conversation threads, especially
    when the parent message mentions a version and the replies contain details.
    
    Args:
        client: Slack WebClient
        channel_id: Channel ID containing the thread
        thread_ts: Thread timestamp (parent message timestamp)
        search_terms: Keywords to search for in replies
        version_info: Parsed version information if available
    
    Returns:
        List of relevant thread replies with metadata
    """
    results = []
    
    try:
        # Fetch all replies in the thread
        response = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            inclusive=True,  # Include the parent message
            limit=100  # Get up to 100 replies
        )
        
        messages = response.get("messages", [])
        if not messages:
            return []
        
        logger.info(f"Found {len(messages)} messages in thread {thread_ts}")
        
        # Get channel name for display
        try:
            channel_info = client.conversations_info(channel=channel_id)
            channel_name = channel_info.get("channel", {}).get("name", f"Channel {channel_id}")
        except Exception:
            channel_name = f"Channel {channel_id}"
        
        # Process each message in the thread
        for msg in messages:
            text = msg.get("text", "")
            if not text:
                continue
            
            # Calculate relevance for this thread message
            relevance_score = _calculate_thread_message_relevance(
                text, search_terms, version_info, is_parent=(msg.get("ts") == thread_ts)
            )
            
            if relevance_score > 0:
                user_id = msg.get("user")
                username = _resolve_username(client, user_id)
                ts = msg.get("ts")
                
                # Get permalink
                try:
                    permalink_resp = client.chat_getPermalink(channel=channel_id, message_ts=ts)
                    permalink = permalink_resp.get("permalink", "")
                except Exception:
                    permalink = ""
                
                results.append({
                    "text": text,  # Raw text, will resolve mentions later for top results only
                    "username": username,
                    "channel": channel_name,
                    "channel_id": channel_id,
                    "ts": ts,
                    "date": _format_slack_timestamp(ts),
                    "permalink": permalink,
                    "thread_relevance_score": relevance_score,
                    "is_thread_reply": True,
                    "thread_ts": thread_ts,
                    "strategy": "thread_focused"
                })
        
        logger.info(f"Thread search found {len(results)} relevant messages")
        
    except SlackApiError as e:
        logger.warning(f"Failed to search thread {thread_ts}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error searching thread {thread_ts}: {e}")
    
    return results


def _calculate_thread_message_relevance(
    text: str,
    search_terms: List[str],
    version_info: Optional[Dict[str, Any]],
    is_parent: bool
) -> float:
    """
    Calculate relevance score for a message within a thread.
    
    Higher scores for:
    - Messages containing answer keywords (delivery, date, timeline, schedule, etc.)
    - Messages mentioning the version number
    - Replies to version-mentioning parent messages
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    score = 0.0
    
    # 1. ANSWER KEYWORDS - High priority for thread replies
    answer_keywords = [
        "delivery", "date", "schedule", "timeline", "when", "available",
        "released", "shipped", "ga", "launch", "track", "on track", "ready",
        "november", "december", "january", "february", "march", "april",
        "may", "june", "july", "august", "september", "october",
        "week", "month", "quarter", "q1", "q2", "q3", "q4",
        "next week", "next month", "this week", "this month"
    ]
    
    for keyword in answer_keywords:
        if keyword in text_lower:
            score += 15.0  # High weight for answer keywords
    
    # 2. VERSION MATCH - If the message mentions the version we're looking for
    if version_info:
        version_str = version_info.get("original", "").lower()
        if version_str in text_lower:
            score += 20.0  # Very high weight for version matches
        
        # Also check for alternative version formats
        year = version_info.get("year")
        major = version_info.get("major")
        minor = version_info.get("minor")
        
        # Check various version format variations
        if year and major:
            alt_formats = [
                f"{year}.{major}",
                f"{str(year)[-2:]}.{major}",
            ]
            if minor is not None:
                alt_formats.extend([
                    f"{year}.{major}.{minor}",
                    f"{str(year)[-2:]}.{major}.{minor}"
                ])
            
            for alt_format in alt_formats:
                if alt_format in text_lower:
                    score += 18.0
                    break
    
    # 3. SEARCH TERMS - Match provided search terms
    for term in search_terms:
        if term.lower() in text_lower:
            score += 8.0
    
    # 4. PARENT MESSAGE BONUS - Parent messages that started the thread
    if is_parent and version_info:
        score += 5.0  # Modest bonus for parent messages
    
    # 5. CONFIRMATION PATTERNS - Phrases indicating answers
    confirmation_patterns = [
        r'\bon track\b', r'\bconfirm\b', r'\byes\b', r'\bready\b',
        r'\bwill be\b', r'\bplanned for\b', r'\bscheduled for\b',
        r'\bexpect\b', r'\btarget\b', r'\baiming for\b'
    ]
    
    for pattern in confirmation_patterns:
        if re.search(pattern, text_lower):
            score += 10.0
    
    return score


def _identify_thread_candidates(
    messages: List[Dict[str, Any]],
    version_info: Optional[Dict[str, Any]],
    search_terms: List[str]
) -> List[Tuple[str, str, float]]:
    """
    Identify messages that are likely parents of valuable threads.
    
    Returns:
        List of (channel_id, thread_ts, priority_score) tuples, sorted by priority
    """
    candidates = []
    
    for msg in messages:
        text = msg.get("text", "")
        if not text:
            continue
        
        # Check if message has a thread
        # Native Slack search may not include reply_count, so also check for thread_ts
        reply_count = msg.get("reply_count", 0)
        has_thread = reply_count > 0 or msg.get("thread_ts") or msg.get("reply_users_count", 0) > 0
        if not has_thread:
            # Even if no explicit thread indicator, prioritize messages with version numbers
            # They might be thread parents that we should check
            if version_info:
                version_str = version_info.get("original", "").lower()
                if version_str not in text.lower():
                    continue
            else:
                continue
        
        # Calculate priority for searching this thread
        priority = 0.0
        text_lower = text.lower()
        
        # High priority if message contains version number
        if version_info:
            version_str = version_info.get("original", "").lower()
            if version_str in text_lower:
                priority += 50.0
        
        # High priority if message is asking a question
        if "?" in text:
            priority += 20.0
        
        # Medium priority if message contains relevant keywords
        for term in search_terms:
            if term.lower() in text_lower:
                priority += 10.0
        
        # Bonus for messages with multiple replies (active discussions)
        if reply_count > 2:
            priority += min(reply_count * 2, 20.0)
        
        if priority > 0:
            channel_id = msg.get("channel_id", "")
            thread_ts = msg.get("ts", "")
            candidates.append((channel_id, thread_ts, priority))
    
    # Sort by priority (highest first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Identified {len(candidates)} thread candidates for deep search")
    return candidates


def _get_slack_client(user_token: Optional[str] = None) -> WebClient:
    """Get Slack client; requires authenticated user's token (no fallback)."""
    token = user_token or st.session_state.get("slack_token")
    if not token:
        raise RuntimeError("Missing Slack token. Authenticate via OAuth.")
    cache_key = f"slack_client_{hash(token)}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = WebClient(token=token)
    return st.session_state[cache_key]


def _resolve_username(client: WebClient, user_id: Optional[str]) -> str:
    """Resolve user ID to username with caching."""
    if not user_id:
        return "Unknown User"
    
    # Use session state for caching
    cache_key = f"user_cache_{user_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        resp = client.users_info(user=user_id)
        if resp.get("ok"):
            profile = resp.get("user", {})
            username = (
                profile.get("real_name") or 
                profile.get("display_name") or 
                profile.get("name") or 
                profile.get("profile", {}).get("display_name") or
                profile.get("profile", {}).get("real_name") or
                f"User {user_id}"
            )
        else:
            username = f"User {user_id}"
        
        st.session_state[cache_key] = username
        return username
        
    except Exception as e:
        logger.warning(f"Failed to resolve username for {user_id}: {e}")
        fallback = f"User {user_id}"
        st.session_state[cache_key] = fallback
        return fallback


def _resolve_user_mentions_in_text(client: WebClient, text: str) -> str:
    """
    Resolve all user mentions in Slack message text to actual usernames.
    
    Converts:
    - <@U066Z0F7GKD|Scott Blauer> -> @Scott Blauer
    - <@U066Z0F7GKD> -> @Scott Blauer (looks up the name via API)
    """
    import re
    
    # First, handle mentions WITH names: <@USER_ID|Name> -> @Name
    text = re.sub(r'<@[A-Z0-9]+\|([^>]+)>', r'@\1', text)
    
    # Find all mentions WITHOUT names: <@USER_ID>
    mention_pattern = r'<@([A-Z0-9]+)>'
    
    def replace_mention(match):
        user_id = match.group(1)
        username = _resolve_username(client, user_id)
        # Return @username without the "User " prefix if it's a fallback
        if username.startswith("User "):
            return f"@{username}"
        return f"@{username}"
    
    text = re.sub(mention_pattern, replace_mention, text)
    
    return text


def _resolve_mentions_in_results(client: WebClient, results: List[dict]) -> List[dict]:
    """
    Resolve user mentions in message text for a list of results.
    This is done AFTER ranking to minimize API calls.
    
    Args:
        client: Slack WebClient
        results: List of message dictionaries
    
    Returns:
        Same list with resolved user mentions in text fields
    """
    logger.info(f"Resolving user mentions for {len(results)} final results")
    
    for result in results:
        if "text" in result:
            result["text"] = _resolve_user_mentions_in_text(client, result["text"])
    
    return results


def _format_slack_timestamp(ts: str) -> str:
    """Convert Slack timestamp to readable date format."""
    try:
        if not ts:
            return "Unknown date"
        timestamp = float(ts)
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Unknown date"


def strategy_1_native_slack_search(user_query: str, user_token: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Strategy 1: Native Slack Search (Unfiltered)
    
    Goal: Mimic typing in the Slack search bar to get natural results.
    Action: Call search.messages with the user's query directly.
    Constraints: NO artificial limits, channel filters, or time restrictions.
    Return ALL results exactly as Slack ranks them.
    """
    logger.info(f"Strategy 1: Native Slack search for: {user_query}")
    
    client = _get_slack_client(user_token)
    results = []
    
    try:
        # Pass query directly to Slack - no modifications, no filters, no time limits
        response = client.search_messages(
            query=user_query,
            count=1000,  # Maximum allowed by Slack API
            sort="score"  # Let Slack rank by relevance
        )
        
        messages = response.get("messages", {})
        matches = messages.get("matches", [])
        
        logger.info(f"Native Slack search returned {len(matches)} matches")
        
        # Process results - filter out private DMs but keep all other results
        for match in matches:
            # Extract message data
            text = match.get("text", "")
            if not text:
                continue
            
            # Get channel info
            channel_info = match.get("channel", {})
            channel_name = channel_info.get("name", "unknown")
            channel_id = channel_info.get("id", "")
            
            # BLOCK PRIVATE DMs - Skip direct messages and group DMs
            if channel_info.get("is_im") or channel_info.get("is_mpim"):
                logger.debug(f"Skipping private DM: {channel_name}")
                continue
            
            # BLOCK SPECIFIC CHANNELS - Skip inhousezendesk channel
            if channel_name.lower() == "inhousezendesk":
                logger.debug(f"Skipping excluded channel: {channel_name}")
                continue
            
            # Get user info
            user_id = match.get("user")
            username = _resolve_username(client, user_id)
            
            # Get timestamp and permalink
            ts = match.get("ts")
            permalink = match.get("permalink", "")
            
            results.append({
                "text": text,  # Raw text, will resolve mentions later for top results only
                "username": username,
                "channel": channel_name,
                "channel_id": channel_id,
                "ts": ts,
                "date": _format_slack_timestamp(ts),
                "permalink": permalink,
                "strategy": "native_slack",
                "slack_score": match.get("score", 0.0)  # Store Slack's relevance score
            })
        
        logger.info(f"Strategy 1: Returning {len(results)} results (no filtering applied)")
        return results
        
    except SlackApiError as e:
        logger.error(f"Strategy 1: Slack API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Strategy 1: Unexpected error: {e}")
        return []


def strategy_2_smart_targeted_search(
    user_query: str,
    intent_data: Dict[str, Any],
    version_info: Optional[Dict[str, Any]] = None,
    is_release_query: bool = False
) -> List[Dict[str, Any]]:
    """
    Strategy 2: Smart Targeted Search (Channel/Time Intelligence)
    
    Enhanced with release intelligence:
    - Prioritizes release channels when version numbers or release keywords detected
    - Uses channel intelligence to find most relevant channels
    - Searches ALL history in selected channels
    - Applies semantic relevance scoring
    """
    logger.info(f"Strategy 2: Smart targeted search for: {user_query}")
    
    client = _get_slack_client()
    intelligence = get_channel_intelligence()
    
    # Extract parameters
    slack_params = intent_data.get("slack_params", {})
    keywords = slack_params.get("keywords", [])
    priority_terms = slack_params.get("priority_terms", [])
    specific_channel = slack_params.get("channels", "").strip()
    
    # CHECK: If user specified a specific channel via filter, search ONLY that channel
    if specific_channel and specific_channel.lower() not in ["all", "any", "", "none"]:
        logger.info(f"Using user-specified channel filter: {specific_channel}")
        # Try to find the channel by name
        try:
            channel_id = intelligence.get_channel_id_by_name(specific_channel)
            if channel_id:
                relevant_channel_ids = [channel_id]
                logger.info(f"Found channel '{specific_channel}' with ID: {channel_id}")
            else:
                # Channel not found, log warning and fall back to normal search
                logger.warning(f"Channel '{specific_channel}' not found, falling back to intelligent channel selection")
                relevant_channel_ids = []
        except Exception as e:
            logger.warning(f"Error looking up channel '{specific_channel}': {e}, falling back to intelligent selection")
            relevant_channel_ids = []
    else:
        relevant_channel_ids = []
    
    # If no specific channel provided or lookup failed, use intelligent channel selection
    if not relevant_channel_ids:
        # ENHANCED: Use release channels if version/release query detected
        if version_info or is_release_query:
            logger.info("Using release-focused channel selection")
            release_channel_ids = _get_release_channels(intelligence)
            
            # Get general relevant channels too
            general_channel_ids = intelligence.get_relevant_channels(
                query=user_query,
                top_k=5,
                keywords=keywords,
                priority_terms=priority_terms
            )
            
            # Combine and deduplicate (release channels first)
            seen = set()
            relevant_channel_ids = []
            for cid in release_channel_ids + general_channel_ids:
                if cid not in seen:
                    seen.add(cid)
                    relevant_channel_ids.append(cid)
            
            # Limit to top 10 total channels for faster search
            relevant_channel_ids = relevant_channel_ids[:10]
            
            logger.info(f"Release-focused search: {len(relevant_channel_ids)} channels "
                       f"({len(release_channel_ids)} release + {len(general_channel_ids)} general)")
        else:
            # Standard channel selection
            relevant_channel_ids = intelligence.get_relevant_channels(
                query=user_query,
                top_k=10,
                keywords=keywords,
                priority_terms=priority_terms
            )
    
    if not relevant_channel_ids:
        logger.info("Strategy 2: No relevant channels found")
        return []
    
    logger.info(f"Strategy 2: Searching {len(relevant_channel_ids)} relevant channels")
    
    # Search ALL history in relevant channels (no time restrictions)
    all_results = []
    
    # Use thread pool for parallel channel searching (increased workers for better speed)
    max_workers = min(8, len(relevant_channel_ids))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_channel = {
            executor.submit(
                _search_channel_all_history,
                client,
                channel_id,
                user_query,
                keywords,
                priority_terms
            ): channel_id
            for channel_id in relevant_channel_ids
        }
        
        # Collect results with timeout and early-exit strategy
        try:
            completed = 0
            for future in as_completed(future_to_channel, timeout=50):
                channel_id = future_to_channel[future]
                try:
                    channel_results = future.result(timeout=10)
                    all_results.extend(channel_results)
                    completed += 1
                    
                    # Early-exit if we have enough high-quality results
                    if len(all_results) >= 50 and completed >= 5:
                        logger.info(f"Strategy 2: Early exit - found {len(all_results)} results from {completed} channels")
                        # Cancel remaining futures
                        for remaining_future in future_to_channel:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                        
                except Exception as e:
                    logger.warning(f"Strategy 2: Failed to search channel {channel_id}: {e}")
        except TimeoutError:
            logger.warning("Strategy 2: Timeout waiting for channel search results")
            # Cancel remaining futures
            for future in future_to_channel:
                future.cancel()
    
    logger.info(f"Strategy 2: Found {len(all_results)} results from targeted channels")
    return all_results


def _search_channel_all_history(
    client: WebClient,
    channel_id: str,
    user_query: str,
    keywords: List[str],
    priority_terms: List[str]
) -> List[Dict[str, Any]]:
    """Search a single channel for ALL history (no time restrictions)."""
    results = []
    
    try:
        # Get channel info
        channel_info = client.conversations_info(channel=channel_id)
        channel_data = channel_info.get("channel", {})
        channel_name = channel_data.get("name", f"Channel {channel_id}")
        
        # BLOCK PRIVATE DMs - Skip direct messages and group DMs
        if channel_data.get("is_im") or channel_data.get("is_mpim"):
            logger.debug(f"Skipping private DM channel: {channel_name}")
            return []
        
        # BLOCK SPECIFIC CHANNELS - Skip inhousezendesk channel
        if channel_name.lower() == "inhousezendesk":
            logger.debug(f"Skipping excluded channel: {channel_name}")
            return []
        
        # Search ALL history - no time restrictions
        next_cursor: Optional[str] = None
        total_fetched = 0
        max_messages = 100  # Increased limit for better coverage
        
        while total_fetched < max_messages:
            response = client.conversations_history(
                channel=channel_id,
                limit=min(100, max_messages - total_fetched),
                cursor=next_cursor or None,
                inclusive=True
                # NO oldest parameter - search ALL history
            )
            
            messages = response.get("messages", [])
            if not messages:
                break
            
            total_fetched += len(messages)
            
            for msg in messages:
                text = msg.get("text", "")
                if not text:
                    continue
                
                # Skip system messages
                if msg.get("subtype") in ["channel_join", "channel_leave", "channel_topic", "channel_purpose"]:
                    continue
                
                # Calculate semantic relevance score
                relevance_score = _calculate_semantic_relevance(text, user_query, keywords, priority_terms)
                
                # Include ALL messages with any relevance (let ranking handle it)
                if relevance_score > 0:
                    # Get user info
                    user_id = msg.get("user")
                    username = _resolve_username(client, user_id)
                    
                    # Get timestamp and permalink
                    ts = msg.get("ts")
                    try:
                        permalink_resp = client.chat_getPermalink(channel=channel_id, message_ts=ts)
                        permalink = permalink_resp.get("permalink", "")
                    except Exception:
                        permalink = ""
                    
                    results.append({
                        "text": text,  # Raw text, will resolve mentions later for top results only
                        "username": username,
                        "channel": channel_name,
                        "channel_id": channel_id,
                        "ts": ts,
                        "date": _format_slack_timestamp(ts),
                        "permalink": permalink,
                        "semantic_score": relevance_score,
                        "strategy": "smart_targeted"
                    })
            
            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break
        
        logger.info(f"Channel {channel_name}: Found {len(results)} relevant messages from {total_fetched} total")
        
    except SlackApiError as e:
        logger.warning(f"Failed to search channel {channel_id}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error searching channel {channel_id}: {e}")
    
    return results


def _calculate_semantic_relevance(
    text: str,
    user_query: str,
    keywords: List[str],
    priority_terms: List[str]
) -> float:
    """Calculate intelligent semantic relevance score for a message."""
    import re
    
    if not text:
        return 0.0
    
    text_lower = text.lower()
    query_lower = user_query.lower()
    score = 0.0
    
    # 1. EXACT MATCHES - Highest priority
    if query_lower in text_lower:
        score += 20.0
    
    # 2. PRIORITY TERMS - High weight for important terms
    for term in priority_terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            # Bonus for exact word boundaries
            if re.search(r'\b' + re.escape(term_lower) + r'\b', text_lower):
                score += 15.0
            else:
                score += 10.0
    
    # 3. KEYWORD MATCHING - Medium weight
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            # Bonus for exact word boundaries
            if re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                score += 8.0
            else:
                score += 5.0
    
    # 4. INTELLIGENT QUERY TERM MATCHING
    query_terms = [term for term in query_lower.split() if len(term) > 2]
    matched_terms = 0
    for term in query_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            score += 2.0
            matched_terms += 1
    
    # Bonus for matching multiple query terms
    if matched_terms > 0:
        coverage_bonus = (matched_terms / len(query_terms)) * 10.0
        score += coverage_bonus
    
    # 5. SEMANTIC PATTERN RECOGNITION
    semantic_patterns = _generate_semantic_patterns(query_lower)
    for pattern in semantic_patterns:
        if re.search(pattern, text_lower):
            score += 5.0
    
    # 6. TECHNICAL ENTITY MATCHING
    technical_entities = _extract_technical_entities(query_lower)
    text_entities = _extract_technical_entities(text_lower)
    
    # Exact technical entity matches
    for entity in technical_entities:
        if entity in text_entities:
            score += 12.0
        else:
            # Fuzzy matching for related entities
            similarity = _calculate_entity_similarity(entity, text_entities)
            if similarity > 0.7:
                score += similarity * 8.0
    
    # 7. CONTEXTUAL RELEVANCE
    context_score = _calculate_contextual_relevance(text_lower, query_lower)
    score += context_score
    
    return score


def _generate_semantic_patterns(query: str) -> List[str]:
    """Generate semantic patterns based on query content."""
    patterns = []
    
    # Extract key concepts from query
    if 'release' in query:
        patterns.extend([
            r'\brelease\b.*\b(?:date|schedule|announcement|available|ga)\b',
            r'\b(?:date|schedule|announcement|available|ga)\b.*\brelease\b',
            r'\bgo\s+ga\b',
            r'\bgeneral\s+availability\b'
        ])
    
    if 'delivery' in query:
        patterns.extend([
            r'\bdelivery\b.*\b(?:date|schedule|timeline)\b',
            r'\b(?:date|schedule|timeline)\b.*\bdelivery\b'
        ])
    
    if any(word in query for word in ['version', 'v', 'update']):
        patterns.extend([
            r'\bversion\b.*\d+\.\d+',
            r'\bv\d+\.\d+',
            r'\bupdate\b.*\b(?:available|released)\b'
        ])
    
    if 'date' in query:
        patterns.extend([
            r'\b(?:date|schedule|timeline)\b.*\b(?:release|delivery|available)\b',
            r'\b(?:release|delivery|available)\b.*\b(?:date|schedule|timeline)\b'
        ])
    
    return patterns


def _extract_technical_entities(text: str) -> List[str]:
    """Extract technical entities like version numbers, codes, etc."""
    import re
    entities = []
    
    # Version numbers (various formats)
    version_patterns = [
        r'\b\d{4}\.\d+\.\d+\b',  # 2024.7.5
        r'\b\d+\.\d+\.\d+\b',    # 25.7.2
        r'\bv\d+\.\d+(?:\.\d+)?\b',  # v25.7.2
        r'\bversion\s+\d+\.\d+(?:\.\d+)?\b'  # version 25.7.2
    ]
    
    for pattern in version_patterns:
        entities.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Ticket numbers, codes
    code_patterns = [
        r'\b[A-Z]+-\d+\b',  # INC-123
        r'\b[A-Z]{2,}\d+\b'  # ABC123
    ]
    
    for pattern in code_patterns:
        entities.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return entities


def _calculate_entity_similarity(entity1: str, entities2: List[str]) -> float:
    """Calculate similarity between technical entities."""
    if not entities2:
        return 0.0
    
    # Extract version components
    def extract_version_parts(version_str):
        import re
        parts = re.findall(r'\d+', version_str)
        return [int(p) for p in parts] if parts else []
    
    parts1 = extract_version_parts(entity1)
    if not parts1:
        return 0.0
    
    max_similarity = 0.0
    for entity2 in entities2:
        parts2 = extract_version_parts(entity2)
        if not parts2:
            continue
        
        # Calculate similarity based on version structure
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Same major.minor version
            if parts1[0] == parts2[0] and parts1[1] == parts2[1]:
                similarity = 0.9
            # Related versions (e.g., 25.7.2 vs 2024.7.5 - same minor.patch)
            elif len(parts1) >= 3 and len(parts2) >= 3 and parts1[1] == parts2[1] and parts1[2] == parts2[2]:
                similarity = 0.8
            # Same minor version
            elif parts1[1] == parts2[1]:
                similarity = 0.6
            # Same patch version
            elif len(parts1) >= 3 and len(parts2) >= 3 and parts1[2] == parts2[2]:
                similarity = 0.4
            else:
                similarity = 0.0
        else:
            similarity = 0.0
        
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def _calculate_contextual_relevance(text: str, query: str) -> float:
    """Calculate contextual relevance based on surrounding context."""
    score = 0.0
    
    # Question-answer context
    if '?' in text and any(word in query for word in ['what', 'when', 'where', 'how', 'why']):
        score += 3.0
    
    # Time-related context
    if any(word in query for word in ['date', 'when', 'schedule', 'timeline']):
        if any(word in text for word in ['date', 'schedule', 'timeline', 'when', 'available']):
            score += 4.0
    
    # Status-related context
    if any(word in query for word in ['status', 'available', 'released', 'ga']):
        if any(word in text for word in ['available', 'released', 'ga', 'status', 'ready']):
            score += 4.0
    
    return score


def _analyze_time_intent(user_query: str) -> Dict[str, Any]:
    """
    Analyze query for temporal specificity.
    Returns intent data for use in adaptive recency boost calculation.
    """
    query_lower = user_query.lower()
    
    # High temporal specificity patterns
    high_temporal_patterns = [
        r'\b(?:version|v)\s*\d{4}\.\d+\.\d+\b',  # version 2025.7.2
        r'\b(?:latest|current|newest|most recent)\b',
        r'\b(?:today|yesterday|this week|this month)\b',
        r'\b(?:just|recently|recent)\b'
    ]
    
    # Check for high temporal specificity
    import re
    for pattern in high_temporal_patterns:
        if re.search(pattern, query_lower):
            return {
                "temporal_specificity": "high",
                "boost_factor": 5.0
            }
    
    # Medium temporal specificity
    medium_temporal_patterns = [
        r'\b(?:when|date|schedule|timeline)\b',
        r'\b(?:release|announcement|update)\b'
    ]
    
    for pattern in medium_temporal_patterns:
        if re.search(pattern, query_lower):
            return {
                "temporal_specificity": "medium",
                "boost_factor": 2.0
            }
    
    # Low temporal specificity (conceptual queries)
    return {
        "temporal_specificity": "low",
        "boost_factor": 0.5
    }


def _calculate_adaptive_recency_boost(
    ts: str,
    user_query: str,
    intent_data: Dict[str, Any]
) -> float:
    """
    Calculate adaptive recency boost based on query intent.
    
    Formula: Adaptive Boost = boost_factor * recency_decay
    
    Where:
    - boost_factor depends on temporal specificity of the query
    - recency_decay decreases with age of the message
    
    This is a MULTIPLIER, not a filter - old valuable content can still rank #1
    """
    try:
        timestamp = float(ts)
        days_ago = (time.time() - timestamp) / (24 * 3600)
    except (ValueError, TypeError):
        return 1.0  # No boost if timestamp is invalid
    
    # Analyze time intent
    time_intent = _analyze_time_intent(user_query)
    boost_factor = time_intent["boost_factor"]
    
    # Calculate recency decay
    if time_intent["temporal_specificity"] == "high":
        # High temporal specificity: strong recency preference
        if days_ago < 1:  # Last 24 hours
            recency_decay = 1.0
        elif days_ago < 7:  # Last week
            recency_decay = 0.8
        elif days_ago < 30:  # Last month
            recency_decay = 0.6
        elif days_ago < 90:  # Last 3 months
            recency_decay = 0.4
        else:
            recency_decay = 0.2
    
    elif time_intent["temporal_specificity"] == "medium":
        # Medium temporal specificity: moderate recency preference
        if days_ago < 7:  # Last week
            recency_decay = 1.0
        elif days_ago < 30:  # Last month
            recency_decay = 0.8
        elif days_ago < 90:  # Last 3 months
            recency_decay = 0.6
        else:
            recency_decay = 0.4
    
    else:  # low temporal specificity
        # Low temporal specificity: minimal recency preference (concepts are timeless)
        if days_ago < 30:  # Last month
            recency_decay = 1.0
        elif days_ago < 90:  # Last 3 months
            recency_decay = 0.9
        else:
            recency_decay = 0.8  # Even old content gets high boost
    
    # Calculate final adaptive boost
    adaptive_boost = boost_factor * recency_decay
    
    return adaptive_boost


def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate results based on message_timestamp + channel_id."""
    seen_hashes: Set[int] = set()
    unique_results = []
    
    for result in results:
        # Create a hash based on timestamp and channel_id (as specified)
        ts = result.get("ts", "")
        channel_id = result.get("channel_id", "")
        result_hash = hash(f"{ts}_{channel_id}")
        
        if result_hash not in seen_hashes:
            seen_hashes.add(result_hash)
            unique_results.append(result)
    
    return unique_results


def _calculate_final_score(
    result: Dict[str, Any],
    user_query: str,
    intent_data: Dict[str, Any],
    version_info: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate final score using the enhanced adaptive ranking formula:
    
    Final Score(M) = (α · Semantic Score(M) + β · Slack Score(M) + γ · Thread Score(M)) 
                     × Adaptive Recency Boost(Intent) × Version Boost(M)
    
    Where:
    - α = 0.6 (semantic relevance)
    - β = 0.2 (Slack's native ranking)
    - γ = 0.2 (thread relevance - NEW)
    """
    # Extract scores
    semantic_score = result.get("semantic_score", 0.0)
    slack_score = result.get("slack_score", 0.0)
    thread_score = result.get("thread_relevance_score", 0.0)
    
    # If no semantic score (from strategy 1), calculate one
    if semantic_score == 0.0:
        slack_params = intent_data.get("slack_params", {})
        keywords = slack_params.get("keywords", [])
        priority_terms = slack_params.get("priority_terms", [])
        semantic_score = _calculate_semantic_relevance(
            result.get("text", ""),
            user_query,
            keywords,
            priority_terms
        )
    
    # Calculate adaptive recency boost
    adaptive_boost = _calculate_adaptive_recency_boost(
        result.get("ts", "0"),
        user_query,
        intent_data
    )
    
    # Calculate version boost (if version info available)
    version_boost = 1.0
    if version_info:
        text = result.get("text", "").lower()
        version_str = version_info.get("original", "").lower()
        
        # Significant boost if message contains the exact version
        if version_str in text:
            version_boost = 2.0
        
        # Also check alternative formats
        year = version_info.get("year")
        major = version_info.get("major")
        if year and major:
            alt_version = f"{year}.{major}"
            if alt_version in text:
                version_boost = max(version_boost, 1.8)
    
    # Apply the enhanced formula with thread awareness
    alpha = 0.6  # Semantic relevance (reduced to make room for thread score)
    beta = 0.2   # Slack score (reduced)
    gamma = 0.2  # Thread relevance (NEW)
    
    base_score = (alpha * semantic_score) + (beta * slack_score) + (gamma * thread_score)
    final_score = base_score * adaptive_boost * version_boost
    
    return final_score


def _rank_results(
    results: List[Dict[str, Any]],
    user_query: str,
    intent_data: Dict[str, Any],
    version_info: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Rank results using the enhanced adaptive ranking algorithm.
    
    Now includes:
    - Thread relevance scoring
    - Version-aware boosting
    - Semantic + Slack + Thread scores combined
    """
    logger.info(f"Ranking {len(results)} results using enhanced adaptive ranking")
    
    # Calculate final scores for all results
    scored_results = []
    for result in results:
        final_score = _calculate_final_score(result, user_query, intent_data, version_info)
        scored_results.append((final_score, result))
    
    # Sort by final score (descending)
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    # Remove internal scoring fields from final results
    final_results = []
    for score, result in scored_results:
        # Clean up internal fields but keep useful metadata
        result.pop("semantic_score", None)
        result.pop("slack_score", None)
        result.pop("thread_relevance_score", None)
        # Keep: is_thread_reply, thread_ts for UI display if needed
        final_results.append(result)
    
    logger.info(f"Ranked {len(final_results)} results")
    if final_results:
        logger.info(f"Top result score: {scored_results[0][0]:.2f}")
    
    return final_results


def search_slack_simplified(
    user_query: str,
    intent_data: Dict[str, Any],
    max_total_results: int = 15,  # TOP 15 limit for UI
    user_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    INTELLIGENT SEARCH WITH VERSION/RELEASE AWARENESS AND THREAD FOCUS
    
    Enhanced features:
    1. Parse version numbers (e.g., "25.7.2" → year 2025)
    2. Detect release-related queries
    3. Prioritize release channels when appropriate
    4. Search threads deeply when version found in parent message
    5. Apply intelligent ranking with thread awareness
    
    Execute search strategies:
    1. Strategy 1: Native Slack Search (Unfiltered)
    2. Strategy 2: Smart Targeted Search (Channel/Time Intelligence)
    3. Strategy 3: Thread-Focused Search (when applicable)
    
    Then apply Adaptive Smart Ranking and return TOP 15 messages.
    """
    logger.info(f"Starting intelligent search for: {user_query}")
    
    # FAST PATH: "Latest message in channel" queries
    # If user asks for latest/recent message AND specifies a channel, skip complex search
    query_lower = user_query.lower()
    is_latest_query = any(term in query_lower for term in [
        "latest message", "last message", "most recent message", 
        "recent message", "newest message", "latest post"
    ])
    
    slack_params = intent_data.get("slack_params", {})
    specific_channel = slack_params.get("channels", "").strip()
    
    if is_latest_query and specific_channel and specific_channel.lower() not in ["all", "any", "none", ""]:
        logger.info(f"⚡ FAST PATH: Latest message query for channel '{specific_channel}'")
        
        # Get channel ID using local helper
        intelligence = get_channel_intelligence()
        channel_id = intelligence.get_channel_id_by_name(specific_channel)
        
        if channel_id:
            logger.info(f"Found channel ID: {channel_id}, fetching latest messages")
            
            try:
                client = _get_slack_client(user_token)
                
                # Fetch latest messages from this channel only
                # Note: Slack returns messages in reverse chronological order (newest first)
                response = client.conversations_history(
                    channel=channel_id,
                    limit=5  # Get latest 5 messages (in case some are system messages)
                )
                
                messages = response.get("messages", [])
                
                if messages:
                    logger.info(f"✓ Fast path found {len(messages)} latest messages")
                    
                    # Format results - filter out system messages and return only the LATEST real message
                    results = []
                    for msg in messages:
                        # Skip system messages (they're not real user messages)
                        if msg.get("subtype") in ["channel_join", "channel_leave", "channel_topic", "channel_purpose", "bot_message"]:
                            continue
                        
                        # Skip messages without text
                        if not msg.get("text"):
                            continue
                        
                        # Resolve user ID to username
                        user_id = msg.get("user", "")
                        username = _resolve_username(client, user_id) if user_id else "Unknown"
                        
                        # Convert timestamp to readable format first
                        ts = msg.get("ts", "")
                        date_str = "Unknown"
                        if ts:
                            try:
                                ts_float = float(ts)
                                from datetime import datetime
                                dt = datetime.fromtimestamp(ts_float)
                                date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                            except:
                                date_str = "Unknown"
                        
                        # Generate permalink (conversations_history doesn't include it)
                        permalink = ""
                        if ts and channel_id:
                            # Remove the decimal point from timestamp for permalink
                            ts_for_link = ts.replace(".", "")
                            permalink = f"https://incorta-group.slack.com/archives/{channel_id}/p{ts_for_link}"
                        
                        result = {
                            "text": msg.get("text", ""),
                            "channel": specific_channel.lstrip('#'),  # UI expects 'channel' field
                            "channel_id": channel_id,
                            "timestamp": ts,
                            "ts": ts,  # Keep for compatibility
                            "date": date_str,  # UI expects 'date' field
                            "user": user_id,
                            "username": username,  # Resolved username
                            "permalink": permalink,
                            "score": 100.0,  # Max relevance for exact channel match
                            "is_fast_path": True
                        }
                        
                        results.append(result)
                        
                        # For "latest message" queries, only return the FIRST real message (newest)
                        break  # Exit after finding the first real user message
                    
                    # Resolve user mentions in message text
                    results = _resolve_mentions_in_results(client, results)
                    
                    logger.info(f"⚡ Fast path complete: Returning {len(results)} results immediately")
                    return results[:max_total_results]
                else:
                    logger.warning(f"No messages found in channel '{specific_channel}'")
            
            except Exception as e:
                logger.error(f"Fast path failed: {e}, falling back to normal search", exc_info=True)
        else:
            logger.warning(f"Channel '{specific_channel}' not found, falling back to normal search")
    
    # STEP 0: Analyze query for version/release intelligence
    version_info = _parse_version_number(user_query)
    is_release_query = _detect_release_query(user_query)
    
    if version_info:
        logger.info(f"Detected version: {version_info['original']} (year: {version_info['year']})")
    if is_release_query:
        logger.info("Detected release-related query")
    
    # Extract search terms for thread search
    slack_params = intent_data.get("slack_params", {})
    keywords = slack_params.get("keywords", [])
    priority_terms = slack_params.get("priority_terms", [])
    search_terms = list(set(keywords + priority_terms + [user_query]))
    
    # STEP 1: Run parallel search strategies
    strategy_1_results = []
    strategy_2_results = []
    
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both strategies
            future_strategy_1 = executor.submit(strategy_1_native_slack_search, user_query, user_token)
            future_strategy_2 = executor.submit(
                strategy_2_smart_targeted_search, 
                user_query, 
                intent_data, 
                version_info, 
                is_release_query
            )
            
            # Collect results with error handling
            try:
                strategy_1_results = future_strategy_1.result(timeout=35)
                logger.info(f"Strategy 1: Retrieved {len(strategy_1_results)} results")
            except TimeoutError:
                logger.warning("Strategy 1 timed out after 35 seconds")
            except Exception as e:
                logger.error(f"Strategy 1 failed: {e}", exc_info=True)
            
            try:
                strategy_2_results = future_strategy_2.result(timeout=60)  # Increased to exceed Strategy 2's internal 50s timeout
                logger.info(f"Strategy 2: Retrieved {len(strategy_2_results)} results")
            except TimeoutError:
                logger.warning("Strategy 2 timed out after 60 seconds")
            except Exception as e:
                logger.error(f"Strategy 2 failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Thread pool execution failed: {e}", exc_info=True)
    
    # Combine initial results
    all_results = strategy_1_results + strategy_2_results
    logger.info(f"Combined: {len(all_results)} total results from strategies 1 & 2")
    
    # STEP 2: Thread-Focused Search (Strategy 3)
    # If version found or release query, search threads of relevant messages
    thread_results = []
    if (version_info or is_release_query) and all_results:
        logger.info("Initiating thread-focused search for version/release query")
        
        # Identify thread candidates
        thread_candidates = _identify_thread_candidates(all_results, version_info, search_terms)
        
        # Search top 5 most promising threads
        client = _get_slack_client()
        for channel_id, thread_ts, priority in thread_candidates[:5]:
            logger.info(f"Searching thread {thread_ts} (priority: {priority:.1f})")
            thread_msgs = _search_message_thread(
                client, channel_id, thread_ts, search_terms, version_info
            )
            thread_results.extend(thread_msgs)
        
        logger.info(f"Thread search found {len(thread_results)} additional messages")
    
    # Combine all results
    all_results.extend(thread_results)
    logger.info(f"Total results after thread search: {len(all_results)}")
    
    if not all_results:
        logger.warning("No results from any strategy")
        return []
    
    # STEP 3: Apply year filter if version detected (with smart fallback)
    if version_info and version_info.get("year") and all_results:
        year = version_info["year"]
        before_filter = len(all_results)
        filtered_results = _filter_messages_by_year(all_results, year)
        
        # If year filter removes ALL results, be more lenient (±1 year)
        if not filtered_results:
            logger.warning(f"Year filter ({year}) removed all results, trying ±1 year")
            filtered_results = []
            for year_offset in [-1, 0, 1]:
                filtered_results.extend(_filter_messages_by_year(all_results, year + year_offset))
            
            # Deduplicate after multi-year filter
            filtered_results = _deduplicate_results(filtered_results)
        
        if filtered_results:
            all_results = filtered_results
            logger.info(f"Year filter ({year}): {before_filter} → {len(all_results)} messages")
        else:
            logger.warning(f"Year filter would remove all results, skipping filter")
            # Keep all results if even lenient filtering yields nothing
    
    # STEP 4: Deduplicate by message_timestamp + channel_id
    unique_results = _deduplicate_results(all_results)
    logger.info(f"After deduplication: {len(unique_results)} unique results")
    
    # STEP 5: Apply Adaptive Smart Ranking
    ranked_results = _rank_results(unique_results, user_query, intent_data, version_info)
    
    # STEP 6: Final Output - TOP 15 messages for UI
    if len(ranked_results) > max_total_results:
        logger.info(f"Limiting to TOP {max_total_results} messages from {len(ranked_results)} total")
        ranked_results = ranked_results[:max_total_results]
    
    # STEP 7: Resolve user mentions ONLY for final top results (performance optimization)
    client = _get_slack_client()
    ranked_results = _resolve_mentions_in_results(client, ranked_results)
    
    logger.info(f"Final results: {len(ranked_results)} messages sent to UI")
    return ranked_results


__all__ = ["search_slack_simplified"]