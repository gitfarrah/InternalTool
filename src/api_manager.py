"""
API Manager for Gemini API Key Rotation and Load Balancing.

This module provides functionality to manage multiple API keys,
rotate between them, and handle quota/rate limit issues automatically.
"""

import random
import time
from typing import List, Dict, Optional, Any, Callable
import logging
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)


class GeminiAPIManager:
    """
    Manages multiple Gemini API keys with automatic rotation and fallback.
    
    Features:
    - Round-robin key rotation
    - Automatic retry on quota errors
    - Success/failure tracking
    - Statistics reporting
    """
    
    def __init__(
        self,
        api_keys: List[str],
        model: str = "gemini-2.5-flash",
        rotation_strategy: str = "round_robin"
    ):
        """
        Initialize API manager with multiple keys.

        Args:
            api_keys: List of Gemini API keys
            model: Model to use (e.g., "gemini-2.0-flash-lite")
            rotation_strategy: "round_robin" or "random"
        """
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        self.api_keys = api_keys
        self.model = model
        self.rotation_strategy = rotation_strategy
        self.current_index = 0
        self.key_stats = {key: {"success": 0, "failed": 0, "quota_errors": 0} for key in api_keys}
        
    def get_llm(self, model_override: Optional[str] = None, **kwargs) -> ChatGoogleGenerativeAI:
        """
        Get LLM instance with current or selected API key.
        
        Args:
            model_override: Override the default model for this call (e.g., "gemini-2.5-pro")
            **kwargs: Additional arguments to pass to ChatGoogleGenerativeAI
            
        Returns:
            Configured ChatGoogleGenerativeAI instance
        """
        # Use override model or default
        actual_model = model_override or self.model
        
        # Select key based on strategy
        if self.rotation_strategy == "random":
            selected_key = random.choice(self.api_keys)
        else:  # round_robin
            selected_key = self.api_keys[self.current_index]
        
        logger.info(f"Using model: {actual_model}, API key index: {self.api_keys.index(selected_key)}")
        
        return ChatGoogleGenerativeAI(
            model=actual_model,
            temperature=0.1,
            api_key=selected_key,
            **kwargs
        )
    
    def rotate_key(self):
        """Move to next API key (round-robin)."""
        old_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        logger.info(f"Rotated from key {old_index} to key {self.current_index}")
        
    def mark_success(self):
        """Mark current key as successful."""
        if self.rotation_strategy == "round_robin":
            current_key = self.api_keys[self.current_index]
            self.key_stats[current_key]["success"] += 1
    
    def mark_failure(self, error_type: str = "generic"):
        """
        Mark current key as failed and rotate.
        
        Args:
            error_type: Type of failure ("quota", "rate_limit", "generic")
        """
        if self.rotation_strategy == "round_robin":
            current_key = self.api_keys[self.current_index]
            self.key_stats[current_key]["failed"] += 1
            
            if error_type in ["quota", "rate_limit"]:
                self.key_stats[current_key]["quota_errors"] += 1
            
            self.rotate_key()
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get usage statistics for all keys.
        
        Returns:
            Dictionary mapping keys to their statistics
        """
        return self.key_stats.copy()
    
    def get_best_key(self) -> str:
        """Select key with best success rate."""
        rates = {}
        for key, stats in self.key_stats.items():
            total = stats["success"] + stats["failed"]
            if total > 0:
                rate = stats["success"] / total
                rates[key] = rate
            else:
                rates[key] = 1.0
        
        sorted_keys = sorted(rates.items(), key=lambda x: x[1], reverse=True)
        return sorted_keys[0][0]
    
    def execute_with_retry(
        self, 
        func: Callable, 
        max_retries: Optional[int] = None,
        retry_delay: int = 30
    ) -> Any:
        """
        Execute function with automatic retry and key rotation on failure.
        
        Args:
            func: Function to execute
            max_retries: Maximum retries (default: number of keys)
            retry_delay: Delay in seconds between retries
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        if max_retries is None:
            max_retries = len(self.api_keys)
        
        for attempt in range(max_retries):
            try:
                result = func()
                self.mark_success()
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a quota/rate limit error
                if any(keyword in error_str for keyword in ["quota", "429", "rate limit"]):
                    logger.warning(f"Quota/rate limit error on attempt {attempt + 1}/{max_retries}")
                    
                    if attempt < max_retries - 1:
                        self.mark_failure(error_type="quota")
                        logger.info(f"Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error("All API keys exhausted quota limits")
                        raise Exception("All API keys exhausted quota limits")
                else:
                    # Non-quota error, re-raise immediately
                    logger.error(f"Non-quota error: {e}")
                    raise
        
        raise Exception(f"Failed after {max_retries} attempts")


def create_api_manager_from_env() -> GeminiAPIManager:
    """
    Create API manager by loading keys from environment variables or Streamlit secrets.

    Looks for: GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    Falls back to: GEMINI_API_KEY

    Returns:
        Configured GeminiAPIManager instance
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Helper function to get key from env or Streamlit secrets
    def get_key(name: str) -> Optional[str]:
        # Try Streamlit secrets first (if available)
        try:
            import streamlit as st
            if name in st.secrets:
                return st.secrets[name]
        except (ImportError, FileNotFoundError, KeyError):
            pass
        # Fall back to environment variables
        return os.getenv(name)

    # Collect all numbered keys
    keys = []
    i = 1
    while True:
        key = get_key(f"GEMINI_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1

    # Fallback to single key
    if not keys:
        single_key = get_key("GEMINI_API_KEY")
        if single_key:
            keys.append(single_key)

    if not keys:
        raise ValueError("No GEMINI_API_KEY found in environment variables or Streamlit secrets")

    logger.info(f"Loaded {len(keys)} API key(s)")
    return GeminiAPIManager(api_keys=keys)


__all__ = ["GeminiAPIManager", "create_api_manager_from_env"]

