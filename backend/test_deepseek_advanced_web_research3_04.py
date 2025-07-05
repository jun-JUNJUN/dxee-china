#!/usr/bin/env python3
"""
Enhanced DeepSeek Web Research with MongoDB Caching and Multi-Query Strategy v3.04
This script implements an advanced research workflow with MongoDB caching:
1. Multi-angle search query generation
2. MongoDB caching for scraped web content
3. Content deduplication and smart caching
4. Enhanced content filtering and source diversification
5. Iterative query refinement based on gaps
6. Comprehensive logging and performance analysis

New Features in v3.04:
- 10-minute time limit for research sessions
- Content summarization to handle DeepSeek token limits (65536 tokens)
- Intelligent input size management and token counting
- Early termination when relevance targets are met
- Batch processing for large content sets
- Optimized error handling for timeout and token limit scenarios
- Replaced web scraping with Bright Data API for reliable content extraction
- MongoDB integration for caching scraped web content
- Smart URL matching to avoid duplicate scraping
- Keywords tracking for better cache management
- Content freshness checking
- Professional content extraction via Bright Data API

Usage:
    python test_deepseek_advanced_web_research3_04.py

Environment variables:
    DEEPSEEK_API_KEY: Your DeepSeek API key
    DEEPSEEK_API_URL: The base URL for the DeepSeek API (default: https://api.deepseek.com)
    GOOGLE_API_KEY: Your Google Custom Search API key
    GOOGLE_CSE_ID: Your Google Custom Search Engine ID
    BRIGHTDATA_API_KEY: Your Bright Data API key for content extraction
    BRIGHTDATA_API_URL: Bright Data API endpoint (default: https://api.brightdata.com/datasets/v3/scrape)
    MONGODB_URI: MongoDB connection string (default: mongodb://localhost:27017)
"""

import os
import sys
import json
import asyncio
import logging
import requests
import re
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from collections import defaultdict

# Bright Data API for content extraction
try:
    import aiohttp
    import tiktoken  # For token counting
except ImportError as e:
    print(f"❌ Missing required libraries: {e}")
    print("📦 Please install: pip install aiohttp tiktoken")
    sys.exit(1)

# MongoDB libraries
try:
    from pymongo import MongoClient
    from motor.motor_asyncio import AsyncIOMotorClient
except ImportError as e:
    print(f"❌ Missing MongoDB libraries: {e}")
    print("📦 Please install: pip install pymongo motor")
    sys.exit(1)

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='deepseek_enhanced_research_v304.log',
    filemode='a'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants for v3.04 optimizations
MAX_RESEARCH_TIME = 600  # 10 minutes in seconds
MAX_CONTENT_LENGTH = 2000  # Max characters per content piece for DeepSeek
MAX_TOTAL_TOKENS = 50000  # Conservative limit for DeepSeek input
TARGET_SOURCES_PER_ITERATION = 8  # Optimal number of sources per analysis