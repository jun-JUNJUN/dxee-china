#!/usr/bin/env python3
"""
Token Management and Optimization Utilities
Extracted from test_deepseek_advanced_web_research4_01.py

This module provides token counting, content optimization, cost estimation,
and batching capabilities for managing AI API interactions efficiently.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Token counting
try:
    import tiktoken
except ImportError:
    tiktoken = None
    logging.warning("tiktoken not available, using fallback token counting")

# Constants from the test algorithm
MAX_RESEARCH_TIME = 600  # 10 minutes in seconds
MAX_CONTENT_LENGTH = 2000  # Max characters per content piece for DeepSeek
MAX_TOTAL_TOKENS = 50000  # Conservative limit for DeepSeek input

# Pricing constants (as of 2024)
DEEPSEEK_INPUT_COST_PER_1M = 0.14
DEEPSEEK_OUTPUT_COST_PER_1M = 0.28
GENERIC_COST_PER_1M = 0.20

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    try:
        if tiktoken:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
    except:
        pass
    # Fallback: rough estimation (1 token ≈ 4 characters)
    return len(text) // 4


def summarize_content(content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Summarize content to fit within max_length"""
    if len(content) <= max_length:
        return content

    # Extract key sentences (first, last, and middle parts)
    sentences = content.split('. ')
    if len(sentences) <= 3:
        return content[:max_length] + "..."

    # Take first 2 sentences, middle sentence, and last sentence
    first_part = '. '.join(sentences[:2])
    middle_idx = len(sentences) // 2
    middle_part = sentences[middle_idx]
    last_part = sentences[-1]

    summarized = f"{first_part}. ... {middle_part}. ... {last_part}"

    if len(summarized) > max_length:
        return content[:max_length] + "..."

    return summarized


def check_time_limit(start_time: float, max_duration: float = MAX_RESEARCH_TIME) -> bool:
    """Check if time limit has been exceeded"""
    return (time.time() - start_time) >= max_duration


@dataclass
class TimingMetrics:
    """Track timing metrics for performance analysis with time limits"""
    start_time: float
    end_time: Optional[float] = None
    phase_times: Dict[str, float] = None
    time_limit_exceeded: bool = False

    def __post_init__(self):
        if self.phase_times is None:
            self.phase_times = {}

    def start_phase(self, phase_name: str):
        """Start timing a phase"""
        if check_time_limit(self.start_time):
            self.time_limit_exceeded = True
        self.phase_times[phase_name] = time.time()


class TokenManager:
    """Manage token counting and optimization"""

    def __init__(self, max_tokens: int = MAX_TOTAL_TOKENS):
        self.max_tokens = max_tokens
        self.encoding = None
        try:
            if tiktoken:
                self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.encoding = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except:
                pass
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4

    def optimize_content(self, content: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
        """Optimize content to fit within token limits"""
        return summarize_content(content, max_length)

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = "deepseek-chat") -> float:
        """Estimate cost in USD based on token usage"""
        if model.startswith("deepseek"):
            input_cost = (input_tokens / 1_000_000) * DEEPSEEK_INPUT_COST_PER_1M
            output_cost = (output_tokens / 1_000_000) * DEEPSEEK_OUTPUT_COST_PER_1M
            return input_cost + output_cost

        # Fallback generic pricing
        return (input_tokens + output_tokens) / 1_000_000 * GENERIC_COST_PER_1M

    def create_batches(self, content_list: List[str], max_tokens_per_batch: int = 15000) -> List[List[str]]:
        """Create batches of content that fit within token limits"""
        batches = []
        current_batch = []
        current_tokens = 0

        for content in content_list:
            content_tokens = self.count_tokens(content)

            # If single content exceeds limit, optimize it
            if content_tokens > max_tokens_per_batch:
                # Process current batch if not empty
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

                # Optimize large content and add as single batch
                optimized = self.optimize_content(content, max_tokens_per_batch * 4)  # 4 chars per token
                batches.append([optimized])
                continue

            # Check if adding this content would exceed limit
            if current_tokens + content_tokens > max_tokens_per_batch:
                # Save current batch and start new one
                if current_batch:
                    batches.append(current_batch)
                current_batch = [content]
                current_tokens = content_tokens
            else:
                # Add to current batch
                current_batch.append(content)
                current_tokens += content_tokens

        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)

        return batches

    def calculate_batch_stats(self, batches: List[List[str]]) -> Dict[str, Any]:
        """Calculate statistics for batching"""
        total_items = sum(len(batch) for batch in batches)
        total_tokens = sum(
            sum(self.count_tokens(content) for content in batch)
            for batch in batches
        )

        batch_sizes = [len(batch) for batch in batches]
        batch_tokens = [
            sum(self.count_tokens(content) for content in batch)
            for batch in batches
        ]

        return {
            "total_batches": len(batches),
            "total_items": total_items,
            "total_tokens": total_tokens,
            "avg_batch_size": sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0,
            "avg_tokens_per_batch": sum(batch_tokens) / len(batch_tokens) if batch_tokens else 0,
            "min_batch_size": min(batch_sizes) if batch_sizes else 0,
            "max_batch_size": max(batch_sizes) if batch_sizes else 0,
            "estimated_cost": self.estimate_cost(total_tokens, total_tokens // 4)  # Assume 1:4 input:output ratio
        }

    def optimize_for_cost(self, content_list: List[str], target_cost: float = 1.0) -> List[str]:
        """Optimize content list to stay within target cost"""
        optimized = []
        current_cost = 0.0

        for content in content_list:
            tokens = self.count_tokens(content)
            estimated_cost = self.estimate_cost(tokens, tokens // 4)

            if current_cost + estimated_cost > target_cost:
                # Try to optimize content to fit budget
                remaining_budget = target_cost - current_cost
                if remaining_budget > 0.01:  # At least 1 cent
                    # Calculate max tokens for remaining budget
                    max_tokens = int((remaining_budget / 0.20) * 1_000_000)  # Conservative estimate
                    optimized_content = self.optimize_content(content, max_tokens * 4)
                    optimized.append(optimized_content)
                break
            else:
                optimized.append(content)
                current_cost += estimated_cost

        return optimized

    def get_usage_report(self, token_usage_history: List[Dict[str, int]]) -> Dict[str, Any]:
        """Generate usage and cost report"""
        if not token_usage_history:
            return {"error": "No usage history provided"}

        total_input_tokens = sum(usage.get("input_tokens", 0) for usage in token_usage_history)
        total_output_tokens = sum(usage.get("output_tokens", 0) for usage in token_usage_history)
        total_cost = self.estimate_cost(total_input_tokens, total_output_tokens)

        return {
            "total_requests": len(token_usage_history),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_tokens_per_request": (total_input_tokens + total_output_tokens) / len(token_usage_history),
            "avg_cost_per_request": round(total_cost / len(token_usage_history), 6),
            "cost_breakdown": {
                "input_cost": round(self.estimate_cost(total_input_tokens, 0), 4),
                "output_cost": round(self.estimate_cost(0, total_output_tokens), 4)
            }
        }