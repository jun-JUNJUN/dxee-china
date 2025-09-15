#!/usr/bin/env python3
"""
Statistical Data Extraction from Research Content
Extracted from test_deepseek_advanced_web_research4_01.py

This module extracts numerical values, percentages, dates, and other statistical
information from web content to populate the statistics section of research logs.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class StatisticalExtractor:
    """Extract statistical data from research content"""

    def extract_numbers(self, content: str) -> List[str]:
        """Extract numerical values from content"""
        # Pattern matches numbers with optional thousand separators, decimals, and units
        pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand|K|M|B))?\b'
        numbers = re.findall(pattern, content, re.IGNORECASE)

        # Clean and deduplicate
        clean_numbers = []
        seen = set()
        for num in numbers:
            clean_num = num.strip()
            if clean_num and clean_num not in seen:
                clean_numbers.append(clean_num)
                seen.add(clean_num)

        return clean_numbers[:10]  # Limit to 10 most significant numbers

    def extract_percentages(self, content: str) -> List[str]:
        """Extract percentage values from content"""
        # Pattern matches percentages with optional decimal points
        pattern = r'\b\d+(?:\.\d+)?%'
        percentages = re.findall(pattern, content)

        # Deduplicate while preserving order
        unique_percentages = []
        seen = set()
        for pct in percentages:
            if pct not in seen:
                unique_percentages.append(pct)
                seen.add(pct)

        return unique_percentages[:10]  # Limit to 10 percentages

    def extract_dates(self, content: str) -> List[str]:
        """Extract year values from content"""
        # Pattern matches 4-digit years starting with 20 (2000-2099)
        pattern = r'\b20\d{2}\b'
        years = re.findall(pattern, content)

        # Deduplicate and sort
        unique_years = sorted(list(set(years)))

        return unique_years[:5]  # Limit to 5 years

    def extract_statistics(self, content: str) -> Dict[str, Any]:
        """Extract complete statistics matching reference format"""
        return {
            "numbers_found": self.extract_numbers(content),
            "percentages": self.extract_percentages(content),
            "dates": self.extract_dates(content),
            "metrics": {}  # Empty dict for additional metrics
        }

    def extract_statistics_from_multiple_sources(self, contents: List[str]) -> Dict[str, Any]:
        """Extract statistics from multiple content sources"""
        all_numbers = []
        all_percentages = []
        all_dates = []

        for content in contents:
            stats = self.extract_statistics(content)
            all_numbers.extend(stats["numbers_found"])
            all_percentages.extend(stats["percentages"])
            all_dates.extend(stats["dates"])

        # Deduplicate across all sources
        unique_numbers = []
        seen_numbers = set()
        for num in all_numbers:
            if num not in seen_numbers:
                unique_numbers.append(num)
                seen_numbers.add(num)

        unique_percentages = []
        seen_percentages = set()
        for pct in all_percentages:
            if pct not in seen_percentages:
                unique_percentages.append(pct)
                seen_percentages.add(pct)

        unique_dates = sorted(list(set(all_dates)))

        return {
            "numbers_found": unique_numbers[:10],
            "percentages": unique_percentages[:10],
            "dates": unique_dates[:5],
            "metrics": {}
        }