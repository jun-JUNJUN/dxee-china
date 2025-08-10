# Environment Configuration Guide

## Overview

This guide covers all environment variables and configuration options for the DeepSeek Research integration, including setup instructions, validation, and best practices for different deployment environments.

## Core Environment Variables

### Required Variables

These variables must be set for the application to function properly:

```bash
# Basic Application Settings
PORT=8100
MONGODB_URI=mongodb://localhost:27017
MEILISEARCH_HOST=http://localhost:7701
DEEPSEEK_API_KEY=your_deepseek_api_key_here
AUTH_SECRET_KEY=your_secure_secret_key_here
```

### DeepSeek Research Variables

**Required for DeepSeek Research functionality**:

```bash
# Google Custom Search API
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here

# Bright Data Content Extraction API  
BRIGHTDATA_API_KEY=your_bright_data_api_key_here
```

**Optional DeepSeek Research Configuration**:

```bash
# Research timeout in seconds (default: 600 = 10 minutes)
DEEPSEEK_RESEARCH_TIMEOUT=600

# Cache expiry in days (default: 30 days)
CACHE_EXPIRY_DAYS=30

# Maximum concurrent research sessions per server (default: 3)
MAX_CONCURRENT_RESEARCH=3

# Relevance threshold for filtering results (default: 7.0 out of 10)
RELEVANCE_THRESHOLD=7.0

# Enable/disable DeepSeek research feature (default: true)
DEEPSEEK_RESEARCH_ENABLED=true

# Token limit for content processing (default: 50000)
MAX_CONTENT_TOKENS=50000

# Content length limit per source (default: 2000 characters)
MAX_CONTENT_LENGTH=2000
```

## Environment Variable Details

### DEEPSEEK_API_KEY
- **Purpose**: Authentication for DeepSeek AI API
- **Required**: Yes
- **Format**: String key from DeepSeek platform
- **Example**: `sk-1234567890abcdef...`
- **Setup**: 
  1. Visit https://platform.deepseek.com
  2. Create account and generate API key
  3. Ensure sufficient credits for usage

### GOOGLE_API_KEY
- **Purpose**: Google Custom Search API access
- **Required**: Yes (for research feature)
- **Format**: Google Cloud API key
- **Example**: `AIzaSyC1234567890abcdef...`
- **Setup**:
  1. Visit Google Cloud Console
  2. Enable Custom Search API
  3. Create API key with Custom Search permission
  4. Set daily quota limits as needed

### GOOGLE_CSE_ID
- **Purpose**: Custom Search Engine identifier
- **Required**: Yes (for research feature)
- **Format**: Custom Search Engine ID
- **Example**: `a12345678901234567:abcdefghijk`
- **Setup**:
  1. Visit https://cse.google.com
  2. Create new search engine
  3. Enable "Search the entire web"
  4. Copy the Search Engine ID (cx parameter)

### BRIGHTDATA_API_KEY
- **Purpose**: High-quality content extraction
- **Required**: Yes (for research feature)
- **Format**: Bright Data API token
- **Example**: `Bearer abc123def456...`
- **Setup**:
  1. Sign up at https://brightdata.com
  2. Subscribe to Web Scraper API
  3. Generate API token
  4. Monitor usage quotas

### CACHE_EXPIRY_DAYS
- **Purpose**: Control MongoDB cache retention
- **Required**: No
- **Default**: 30 days
- **Range**: 1-365 days
- **Format**: Integer
- **Example**: `CACHE_EXPIRY_DAYS=90`
- **Impact**: 
  - Higher values = better performance, more storage
  - Lower values = fresher data, less storage

### DEEPSEEK_RESEARCH_TIMEOUT
- **Purpose**: Maximum research session duration
- **Required**: No
- **Default**: 600 seconds (10 minutes)
- **Range**: 60-1800 seconds (1-30 minutes)
- **Format**: Integer (seconds)
- **Example**: `DEEPSEEK_RESEARCH_TIMEOUT=900`
- **Impact**:
  - Longer timeouts allow more comprehensive research
  - Shorter timeouts provide faster responses with less detail

## Environment Files

### .env File Structure

Create a `.env` file in the project root:

```bash
# Copy from template
cp .env.example .env

# Edit with your values
nano .env
```

### Development Environment (.env.development)

```bash
# Development settings
PORT=8100
NODE_ENV=development
DEBUG=true
LOG_LEVEL=DEBUG

# Local services
MONGODB_URI=mongodb://localhost:27017/deepschina_dev
MEILISEARCH_HOST=http://localhost:7701

# API Keys (development/testing)
DEEPSEEK_API_KEY=your_dev_api_key
GOOGLE_API_KEY=your_dev_google_key
GOOGLE_CSE_ID=your_dev_cse_id
BRIGHTDATA_API_KEY=your_dev_brightdata_key

# Development-specific settings
CACHE_EXPIRY_DAYS=7
DEEPSEEK_RESEARCH_TIMEOUT=300
MAX_CONCURRENT_RESEARCH=1
```

### Production Environment (.env.production)

```bash
# Production settings
PORT=8100
NODE_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Production services
MONGODB_URI=mongodb://prod-mongo-cluster/deepschina
MEILISEARCH_HOST=https://search.yourdomain.com

# API Keys (production)
DEEPSEEK_API_KEY=your_prod_api_key
GOOGLE_API_KEY=your_prod_google_key
GOOGLE_CSE_ID=your_prod_cse_id
BRIGHTDATA_API_KEY=your_prod_brightdata_key

# Production-specific settings
CACHE_EXPIRY_DAYS=30
DEEPSEEK_RESEARCH_TIMEOUT=600
MAX_CONCURRENT_RESEARCH=5

# Security
AUTH_SECRET_KEY=your_very_secure_secret_key
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

### Staging Environment (.env.staging)

```bash
# Staging settings
PORT=8100
NODE_ENV=staging
DEBUG=false
LOG_LEVEL=DEBUG

# Staging services
MONGODB_URI=mongodb://staging-mongo/deepschina_staging
MEILISEARCH_HOST=http://staging-search:7701

# API Keys (staging/testing)
DEEPSEEK_API_KEY=your_staging_api_key
GOOGLE_API_KEY=your_staging_google_key
GOOGLE_CSE_ID=your_staging_cse_id
BRIGHTDATA_API_KEY=your_staging_brightdata_key

# Staging-specific settings
CACHE_EXPIRY_DAYS=14
DEEPSEEK_RESEARCH_TIMEOUT=450
MAX_CONCURRENT_RESEARCH=3
```

## Configuration Validation

### Startup Validation Script

Create `scripts/validate_config.py`:

```python
#!/usr/bin/env python3
"""
Configuration Validation Script
Validates all environment variables and API connections
"""

import os
import sys
import asyncio
import aiohttp
from datetime import datetime

class ConfigValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def validate_required_vars(self):
        """Validate required environment variables"""
        required_vars = [
            'DEEPSEEK_API_KEY',
            'MONGODB_URI',
            'AUTH_SECRET_KEY'
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                self.errors.append(f"Missing required variable: {var}")
    
    def validate_research_vars(self):
        """Validate DeepSeek research variables"""
        research_vars = [
            'GOOGLE_API_KEY',
            'GOOGLE_CSE_ID', 
            'BRIGHTDATA_API_KEY'
        ]
        
        missing_vars = [var for var in research_vars if not os.getenv(var)]
        
        if missing_vars:
            self.warnings.append(f"DeepSeek research disabled - missing: {', '.join(missing_vars)}")
        
    def validate_numeric_config(self):
        """Validate numeric configuration values"""
        numeric_configs = {
            'CACHE_EXPIRY_DAYS': (1, 365),
            'DEEPSEEK_RESEARCH_TIMEOUT': (60, 1800),
            'MAX_CONCURRENT_RESEARCH': (1, 10),
            'RELEVANCE_THRESHOLD': (0.0, 10.0)
        }
        
        for var, (min_val, max_val) in numeric_configs.items():
            value = os.getenv(var)
            if value:
                try:
                    num_val = float(value)
                    if not (min_val <= num_val <= max_val):
                        self.warnings.append(f"{var}={value} outside recommended range [{min_val}, {max_val}]")
                except ValueError:
                    self.errors.append(f"Invalid numeric value for {var}: {value}")
    
    async def test_api_connections(self):
        """Test external API connections"""
        
        # Test DeepSeek API
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {'Authorization': f'Bearer {api_key}'}
                    async with session.get('https://api.deepseek.com/v1/models', headers=headers, timeout=10) as resp:
                        if resp.status != 200:
                            self.warnings.append(f"DeepSeek API connection issue: HTTP {resp.status}")
                        else:
                            print("✅ DeepSeek API connection successful")
            except Exception as e:
                self.warnings.append(f"DeepSeek API test failed: {e}")
        
        # Test Google Custom Search API
        google_key = os.getenv('GOOGLE_API_KEY')
        cse_id = os.getenv('GOOGLE_CSE_ID')
        if google_key and cse_id:
            try:
                async with aiohttp.ClientSession() as session:
                    params = {
                        'key': google_key,
                        'cx': cse_id,
                        'q': 'test',
                        'num': 1
                    }
                    async with session.get('https://www.googleapis.com/customsearch/v1', params=params, timeout=10) as resp:
                        if resp.status != 200:
                            self.warnings.append(f"Google Search API connection issue: HTTP {resp.status}")
                        else:
                            print("✅ Google Search API connection successful")
            except Exception as e:
                self.warnings.append(f"Google Search API test failed: {e}")
    
    async def validate_all(self):
        """Run all validations"""
        print("🔍 Validating configuration...")
        print(f"📅 Validation timestamp: {datetime.now().isoformat()}")
        print()
        
        self.validate_required_vars()
        self.validate_research_vars() 
        self.validate_numeric_config()
        await self.test_api_connections()
        
        # Print results
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"   • {error}")
            print()
        
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   • {warning}")
            print()
        
        if not self.errors and not self.warnings:
            print("✅ Configuration validation passed!")
        elif not self.errors:
            print("✅ Configuration validation passed with warnings")
        else:
            print("❌ Configuration validation failed")
            sys.exit(1)

if __name__ == "__main__":
    validator = ConfigValidator()
    asyncio.run(validator.validate_all())
```

### Running Validation

```bash
# Make script executable
chmod +x scripts/validate_config.py

# Run validation
python scripts/validate_config.py

# Run with environment file
python -m dotenv run python scripts/validate_config.py
```

## Configuration Management

### Environment-Specific Configuration

Use different configuration files for each environment:

```bash
# Load development config
export ENV_FILE=.env.development
source .env.development

# Load production config  
export ENV_FILE=.env.production
source .env.production

# Load staging config
export ENV_FILE=.env.staging
source .env.staging
```

### Docker Configuration

#### Docker Compose Environment

```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8100:8100"
    environment:
      - PORT=8100
      - MONGODB_URI=mongodb://mongo:27017/deepschina
      - MEILISEARCH_HOST=http://meilisearch:7700
    env_file:
      - .env
    depends_on:
      - mongo
      - meilisearch
  
  mongo:
    image: mongo:latest
    volumes:
      - mongo_data:/data/db
    
  meilisearch:
    image: getmeili/meilisearch:latest
    volumes:
      - meili_data:/meili_data

volumes:
  mongo_data:
  meili_data:
```

#### Docker Secrets

For production deployment with Docker Swarm:

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  web:
    image: deepschina:latest
    secrets:
      - deepseek_api_key
      - google_api_key
      - brightdata_api_key
    environment:
      - DEEPSEEK_API_KEY_FILE=/run/secrets/deepseek_api_key
      - GOOGLE_API_KEY_FILE=/run/secrets/google_api_key
      - BRIGHTDATA_API_KEY_FILE=/run/secrets/brightdata_api_key

secrets:
  deepseek_api_key:
    external: true
  google_api_key:
    external: true
  brightdata_api_key:
    external: true
```

### Kubernetes Configuration

#### ConfigMap for Non-Sensitive Config

```yaml
# k8s-configmap.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: deepschina-config
data:
  PORT: "8100"
  MONGODB_URI: "mongodb://mongo-service:27017/deepschina"
  MEILISEARCH_HOST: "http://meilisearch-service:7700"
  CACHE_EXPIRY_DAYS: "30"
  DEEPSEEK_RESEARCH_TIMEOUT: "600"
  MAX_CONCURRENT_RESEARCH: "5"
```

#### Secrets for Sensitive Config

```yaml
# k8s-secrets.yml
apiVersion: v1
kind: Secret
metadata:
  name: deepschina-secrets
type: Opaque
data:
  DEEPSEEK_API_KEY: <base64-encoded-key>
  GOOGLE_API_KEY: <base64-encoded-key>
  GOOGLE_CSE_ID: <base64-encoded-id>
  BRIGHTDATA_API_KEY: <base64-encoded-key>
  AUTH_SECRET_KEY: <base64-encoded-key>
```

## Best Practices

### Security

1. **Never commit API keys to version control**
2. **Use different keys for development/staging/production**
3. **Rotate API keys regularly**
4. **Monitor API usage and set alerts**
5. **Use environment-specific configuration files**

### Performance Tuning

1. **Cache Expiry**: Adjust based on data freshness needs
   - News/current events: 1-7 days
   - Reference material: 30-90 days
   - Stable documentation: 180+ days

2. **Timeout Configuration**:
   - Development: 300-600 seconds for debugging
   - Production: 600-900 seconds for user experience
   - Batch processing: 900+ seconds if needed

3. **Concurrency Limits**:
   - Start with conservative limits (3-5 sessions)
   - Monitor resource usage and adjust upward
   - Consider server capacity and API rate limits

### Monitoring

```bash
# Monitor configuration in production
cat <<EOF > scripts/config_monitor.sh
#!/bin/bash
echo "Configuration Status Report - $(date)"
echo "=================================="

echo "Environment: $NODE_ENV"
echo "Port: $PORT"
echo "MongoDB: $MONGODB_URI"
echo "Cache Expiry: $CACHE_EXPIRY_DAYS days"
echo "Research Timeout: $DEEPSEEK_RESEARCH_TIMEOUT seconds"
echo "Max Concurrent: $MAX_CONCURRENT_RESEARCH"

echo ""
echo "API Status:"
echo "- DeepSeek: $([ -n "$DEEPSEEK_API_KEY" ] && echo "Configured" || echo "Missing")"
echo "- Google Search: $([ -n "$GOOGLE_API_KEY" ] && echo "Configured" || echo "Missing")"
echo "- Bright Data: $([ -n "$BRIGHTDATA_API_KEY" ] && echo "Configured" || echo "Missing")"

echo ""
echo "System Resources:"
echo "- Memory: $(free -h | awk '/^Mem/ {print $3 "/" $2}')"
echo "- Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
EOF

chmod +x scripts/config_monitor.sh
```

### Backup Configuration

```bash
# Backup current configuration
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

# Create configuration snapshot
cat <<EOF > config_snapshot_$(date +%Y%m%d).json
{
  "timestamp": "$(date -Iseconds)",
  "environment": "$NODE_ENV",
  "cache_expiry_days": "$CACHE_EXPIRY_DAYS", 
  "research_timeout": "$DEEPSEEK_RESEARCH_TIMEOUT",
  "max_concurrent": "$MAX_CONCURRENT_RESEARCH",
  "apis_configured": {
    "deepseek": $([ -n "$DEEPSEEK_API_KEY" ] && echo "true" || echo "false"),
    "google": $([ -n "$GOOGLE_API_KEY" ] && echo "true" || echo "false"),
    "brightdata": $([ -n "$BRIGHTDATA_API_KEY" ] && echo "true" || echo "false")
  }
}
EOF
```

This configuration guide ensures proper setup and management of the DeepSeek Research environment across all deployment scenarios.