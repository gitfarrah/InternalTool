from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import threading

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class ChannelInfo:
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
    pattern: str
    regex: re.Pattern
    channels: Set[str]
    category: str
    confidence: float


class ChannelIntelligence:
    def __init__(self, client: WebClient):
        self.client = client
        self._lock = threading.Lock()
        self._channels: Dict[str, ChannelInfo] = {}
        self._patterns: List[ChannelPattern] = []
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
        self._initialized = False

    def initialize(self, force_refresh: bool = False) -> None:
        with self._lock:
            if self._initialized and not force_refresh:
                return

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
            self._discover_channels()
            self._detect_patterns()
            self._build_keyword_index()

            st.session_state[cache_key] = {
                "channels": self._channels,
                "patterns": self._patterns,
                "keyword_index": dict(self._keyword_index),
                "category_index": dict(self._category_index)
            }
            self._initialized = True
            logger.info(f"Channel intelligence initialized: {len(self._channels)} channels, {len(self._patterns)} patterns")

    def _discover_channels(self) -> None:
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
                    if channel.get("is_im") or channel.get("is_mpim"):
                        continue
                    channel_name = channel.get("name", "").lower()
                    if channel_name == "inhousezendesk":
                        continue
                    is_private = channel.get("is_private", False)
                    is_member = channel.get("is_member", False)
                    if is_private and not is_member:
                        continue
                    name = channel.get("name", "")
                    purpose = channel.get("purpose", {}).get("value", "")
                    topic = channel.get("topic", {}).get("value", "")
                    member_count = channel.get("num_members", 0)
                    keywords = self._extract_keywords_from_text(f"{name} {purpose} {topic}")
                    self._channels[channel_id] = ChannelInfo(
                        id=channel_id,
                        name=name,
                        purpose=purpose,
                        topic=topic,
                        is_private=is_private,
                        is_member=is_member,
                        member_count=member_count,
                        keywords=keywords,
                        patterns=set(),
                    )
                    total_channels += 1
                next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
                if not next_cursor:
                    break
            logger.info(f"Discovered {total_channels} accessible channels")
        except SlackApiError as e:
            logger.error(f"Failed to discover channels: {e}")
            raise

    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        if not text:
            return set()
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "about", "what", "when",
            "where", "why", "how", "who", "which", "this", "that", "these", "those",
            "channel", "chat", "discussion", "team", "group", "project"
        }
        words = re.findall(r'\b[a-zA-Z0-9-]+\b', text.lower())
        keywords = set()
        for word in words:
            if word in stopwords or len(word) < 2:
                continue
            if word.isdigit():
                continue
            clean_word = word.strip('-')
            if len(clean_word) >= 2:
                keywords.add(clean_word)
        return keywords

    def _detect_patterns(self) -> None:
        logger.info("Detecting channel patterns...")
        prefix_groups = defaultdict(list)
        suffix_groups = defaultdict(list)
        for channel_id, channel_info in self._channels.items():
            name = channel_info.name.lower()
            for i in range(2, len(name)):
                if name[i] == '-':
                    prefix = name[:i+1]
                    prefix_groups[prefix].append(channel_id)
                    break
            for i in range(len(name)-2, 0, -1):
                if name[i] == '-':
                    suffix = name[i:]
                    suffix_groups[suffix].append(channel_id)
                    break
        self._patterns = []
        for prefix, channel_ids in prefix_groups.items():
            if len(channel_ids) >= 2:
                category = self._infer_category_from_pattern(prefix, channel_ids)
                confidence = min(1.0, len(channel_ids) / 10.0)
                pattern = ChannelPattern(
                    pattern=prefix + "*",
                    regex=re.compile(f"^{re.escape(prefix[:-1])}-", re.IGNORECASE),
                    channels=set(channel_ids),
                    category=category,
                    confidence=confidence,
                )
                self._patterns.append(pattern)
                for channel_id in channel_ids:
                    self._channels[channel_id].patterns.add(pattern.pattern)
                    self._channels[channel_id].category = category
        for suffix, channel_ids in suffix_groups.items():
            if len(channel_ids) >= 2:
                category = self._infer_category_from_pattern(suffix, channel_ids)
                confidence = min(1.0, len(channel_ids) / 10.0)
                pattern = ChannelPattern(
                    pattern="*" + suffix,
                    regex=re.compile(f"{re.escape(suffix[1:])}$", re.IGNORECASE),
                    channels=set(channel_ids),
                    category=category,
                    confidence=confidence,
                )
                self._patterns.append(pattern)
                for channel_id in channel_ids:
                    self._channels[channel_id].patterns.add(pattern.pattern)
                    if not self._channels[channel_id].category:
                        self._channels[channel_id].category = category
        self._patterns.sort(key=lambda p: p.confidence, reverse=True)
        logger.info(f"Detected {len(self._patterns)} channel patterns")

    def _infer_category_from_pattern(self, pattern: str, channel_ids: List[str]) -> str:
        sample_names = [self._channels[cid].name.lower() for cid in channel_ids[:5]]
        sample_text = " ".join(sample_names)
        category_keywords = {
            "engineering": ["eng", "dev", "development", "backend", "frontend", "api", "tech"],
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
            "release": ["release", "announcement", "announce", "version", "update", "deploy", "deployment", "launch"],
        }
        best_category = "general"
        best_score = 0
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in sample_text)
            if score > best_score:
                best_score = score
                best_category = category
        return best_category

    def _build_keyword_index(self) -> None:
        logger.info("Building keyword index...")
        for channel_id, channel_info in self._channels.items():
            for keyword in channel_info.keywords:
                self._keyword_index[keyword].add(channel_id)
            if channel_info.category:
                self._category_index[channel_info.category].add(channel_id)
            name_words = self._extract_keywords_from_text(channel_info.name)
            for word in name_words:
                self._keyword_index[word].add(channel_id)
        logger.info(f"Built keyword index with {len(self._keyword_index)} keywords")

    def get_relevant_channels(
        self,
        query: str,
        top_k: int = 20,
        keywords: Optional[List[str]] = None,
        priority_terms: Optional[List[str]] = None,
    ) -> List[str]:
        if not self._initialized:
            self.initialize()
        if not query and not keywords and not priority_terms:
            return self._get_most_active_channels(top_k)
        if not keywords:
            keywords = list(self._extract_keywords_from_text(query))
        if priority_terms:
            keywords.extend(priority_terms)
        channel_scores = defaultdict(float)
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self._keyword_index:
                for channel_id in self._keyword_index[keyword_lower]:
                    channel_scores[channel_id] += 2.0
            for pattern in self._patterns:
                if pattern.regex.search(keyword_lower):
                    for channel_id in pattern.channels:
                        channel_scores[channel_id] += pattern.confidence * 1.0
            for indexed_keyword, channel_ids in self._keyword_index.items():
                if keyword_lower in indexed_keyword or indexed_keyword in keyword_lower:
                    for channel_id in channel_ids:
                        channel_scores[channel_id] += 0.5
            for channel_id, channel_info in self._channels.items():
                channel_name = channel_info.name.lower()
                if keyword_lower in channel_name:
                    channel_scores[channel_id] += 3.0
                elif any(word in channel_name for word in keyword_lower.split()):
                    channel_scores[channel_id] += 1.5
        if priority_terms:
            for term in priority_terms:
                term_lower = term.lower()
                if term_lower in self._keyword_index:
                    for channel_id in self._keyword_index[term_lower]:
                        channel_scores[channel_id] += 2.0
        sorted_channels = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)
        return [channel_id for channel_id, _ in sorted_channels[:top_k]]

    def _get_most_active_channels(self, top_k: int) -> List[str]:
        sorted_channels = sorted(self._channels.items(), key=lambda x: x[1].member_count, reverse=True)
        return [channel_id for channel_id, _ in sorted_channels[:top_k]]

    def get_channel_info(self, channel_id: str) -> Optional[ChannelInfo]:
        if not self._initialized:
            self.initialize()
        return self._channels.get(channel_id)

    def get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        if not self._initialized:
            self.initialize()
        search_name = channel_name.lstrip('#').lower().strip()
        for channel_id, channel_info in self._channels.items():
            if channel_info.name.lower() == search_name:
                return channel_id
        for channel_id, channel_info in self._channels.items():
            if search_name in channel_info.name.lower():
                return channel_id
        return None

    def find_similar_channels(self, channel_name: str, max_suggestions: int = 5) -> List[str]:
        if not self._initialized:
            self.initialize()
        search_name = channel_name.lstrip('#').lower().strip()
        suggestions = []
        search_words = search_name.split('_')
        for channel_id, channel_info in self._channels.items():
            channel_lower = channel_info.name.lower()
            score = 0
            if channel_lower == search_name:
                score = 100
            elif search_name in channel_lower:
                score = 80
            else:
                for word in search_words:
                    if word in channel_lower:
                        score += 20
            if score > 0:
                suggestions.append((channel_info.name, score))
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in suggestions[:max_suggestions]]

    def get_patterns(self) -> List[ChannelPattern]:
        if not self._initialized:
            self.initialize()
        return self._patterns

    def get_categories(self) -> Dict[str, int]:
        if not self._initialized:
            self.initialize()
        category_counts = defaultdict(int)
        for channel_info in self._channels.values():
            if channel_info.category:
                category_counts[channel_info.category] += 1
        return dict(category_counts)

    def search_channels_by_name(self, name_pattern: str) -> List[str]:
        if not self._initialized:
            self.initialize()
        pattern = re.compile(name_pattern, re.IGNORECASE)
        matching_channels = []
        for channel_id, channel_info in self._channels.items():
            if pattern.search(channel_info.name):
                matching_channels.append(channel_id)
        return matching_channels


def get_channel_intelligence() -> ChannelIntelligence:
    """Get or create ChannelIntelligence instance with caching; requires user OAuth token."""
    if "channel_intelligence" not in st.session_state:
        token = st.session_state.get("slack_token")
        if not token:
            raise RuntimeError("Missing Slack token. Authenticate via OAuth.")
        client = WebClient(token=token)
        st.session_state["channel_intelligence"] = ChannelIntelligence(client)
    intelligence = st.session_state["channel_intelligence"]
    intelligence.initialize()
    return intelligence


__all__ = [
    "ChannelInfo",
    "ChannelPattern",
    "ChannelIntelligence",
    "get_channel_intelligence",
]
