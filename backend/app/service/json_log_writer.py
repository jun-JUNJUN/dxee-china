#!/usr/bin/env python3
"""
JSON Log Writer for Research Results
Generates timestamped JSON log files matching research_results_20250904_104734.json format.

This module ensures exact compliance with the reference JSON schema for research
result logging and analysis.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class JSONLogWriter:
    """Write research results to timestamped JSON log files"""

    def __init__(self, log_directory: str = "logs"):
        """
        Initialize JSON log writer

        Args:
            log_directory: Directory to store log files (default: "logs")
        """
        self.log_directory = log_directory
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Ensure log directory exists"""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            logger.info(f"Created log directory: {self.log_directory}")

    def get_log_filename(self, timestamp: datetime) -> str:
        """
        Generate log filename following pattern: research_results_YYYYMMDD_HHMMSS.json

        Args:
            timestamp: Datetime to use for filename

        Returns:
            Filename string
        """
        return f"research_results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

    def write_research_log(self, research_result: Dict[str, Any], timestamp: Optional[datetime] = None) -> str:
        """
        Write research result to JSON log file

        Args:
            research_result: Research result data to log
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Log filename that was created
        """
        if timestamp is None:
            timestamp = datetime.now()

        filename = self.get_log_filename(timestamp)
        full_path = os.path.join(self.log_directory, filename)

        # Validate format before writing
        if not self.validate_log_format(research_result):
            logger.warning("Research result format validation failed, writing anyway")

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(research_result, f, indent=2, ensure_ascii=False)

            logger.info(f"Research log written to: {filename}")
            return filename

        except Exception as e:
            logger.error(f"Failed to write research log: {e}")
            raise

    def validate_log_format(self, data: Dict[str, Any]) -> bool:
        """
        Validate research result matches expected JSON schema

        Args:
            data: Research result data to validate

        Returns:
            True if format is valid, False otherwise
        """
        try:
            # Check required top-level keys
            required_keys = ["question", "answer", "confidence", "sources",
                           "statistics", "metadata", "duration"]

            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key: {key}")
                    return False

            # Validate data types
            if not isinstance(data["question"], str):
                logger.error("question must be string")
                return False

            if not isinstance(data["answer"], str):
                logger.error("answer must be string")
                return False

            if not isinstance(data["confidence"], (int, float)):
                logger.error("confidence must be number")
                return False

            if not isinstance(data["sources"], list):
                logger.error("sources must be list")
                return False

            if not isinstance(data["duration"], (int, float)):
                logger.error("duration must be number")
                return False

            # Validate statistics structure
            statistics = data.get("statistics", {})
            if not isinstance(statistics, dict):
                logger.error("statistics must be dict")
                return False

            stats_keys = ["numbers_found", "percentages", "dates", "metrics"]
            for key in stats_keys:
                if key not in statistics:
                    logger.error(f"Missing statistics key: {key}")
                    return False

                if key == "metrics":
                    if not isinstance(statistics[key], dict):
                        logger.error("metrics must be dict")
                        return False
                else:
                    if not isinstance(statistics[key], list):
                        logger.error(f"{key} must be list")
                        return False

            # Validate metadata structure
            metadata = data.get("metadata", {})
            if not isinstance(metadata, dict):
                logger.error("metadata must be dict")
                return False

            metadata_keys = ["relevance_threshold", "timeout_reached", "serper_requests"]
            for key in metadata_keys:
                if key not in metadata:
                    logger.error(f"Missing metadata key: {key}")
                    return False

            # Type checks for metadata
            if not isinstance(metadata["relevance_threshold"], (int, float)):
                logger.error("relevance_threshold must be number")
                return False

            if not isinstance(metadata["timeout_reached"], bool):
                logger.error("timeout_reached must be boolean")
                return False

            if not isinstance(metadata["serper_requests"], int):
                logger.error("serper_requests must be integer")
                return False

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def read_research_log(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Read research log from file

        Args:
            filename: Log filename to read

        Returns:
            Research result data or None if file doesn't exist
        """
        full_path = os.path.join(self.log_directory, filename)

        if not os.path.exists(full_path):
            logger.error(f"Log file not found: {filename}")
            return None

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data

        except Exception as e:
            logger.error(f"Failed to read log file {filename}: {e}")
            return None

    def list_log_files(self) -> list:
        """
        List all research log files in the log directory

        Returns:
            List of log filenames
        """
        if not os.path.exists(self.log_directory):
            return []

        try:
            files = [f for f in os.listdir(self.log_directory)
                    if f.startswith("research_results_") and f.endswith(".json")]
            return sorted(files, reverse=True)  # Most recent first

        except Exception as e:
            logger.error(f"Failed to list log files: {e}")
            return []