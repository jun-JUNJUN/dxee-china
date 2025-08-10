# DeepSeek Research Deployment Guide

## Overview

This guide covers deployment procedures for the DeepSeek Research functionality across different environments, including dependencies, database migrations, monitoring setup, and rollback procedures.

## Pre-Deployment Checklist

### Requirements Verification

Before deploying, ensure all requirements are met:

- [ ] Python >=3.11 installed
- [ ] MongoDB running and accessible
- [ ] Required API keys obtained and validated
- [ ] Network access to external APIs verified
- [ ] Sufficient server resources available
- [ ] Backup procedures in place

### API Key Setup

Obtain and configure the following API keys:

1. **DeepSeek API Key**
   - Visit: https://platform.deepseek.com
   - Create account and generate API key
   - Verify sufficient credits/quota

2. **Google Custom Search API**
   - Visit: Google Cloud Console
   - Enable Custom Search API
   - Create API key with appropriate permissions
   - Set up Custom Search Engine at https://cse.google.com

3. **Bright Data API Key**
   - Visit: https://brightdata.com
   - Subscribe to Web Scraper API
   - Generate API token
   - Configure usage limits

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 20GB available space
- **Network**: Stable internet with access to external APIs

#### Recommended Production Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 50GB+ SSD
- **Network**: High-speed connection with redundancy

## Database Setup and Migration

### MongoDB Collections Setup

Create required indexes and collections:

```javascript
// Connect to MongoDB
use deepschina

// Create research cache collection with TTL index
db.createCollection("web_content_cache")
db.web_content_cache.createIndex(
  { "timestamp": 1 }, 
  { expireAfterSeconds: 2592000 }  // 30 days default
)
db.web_content_cache.createIndex({ "url": 1 }, { unique: true })
db.web_content_cache.createIndex({ "keywords": 1 })

// Create research sessions collection
db.createCollection("research_sessions")
db.research_sessions.createIndex({ "chat_id": 1 })
db.research_sessions.createIndex({ "timestamp": 1 })
db.research_sessions.createIndex({ "user_id": 1 })

// Create API usage logs collection
db.createCollection("api_usage_logs")
db.api_usage_logs.createIndex({ "timestamp": 1 })
db.api_usage_logs.createIndex({ "api_name": 1 })
db.api_usage_logs.createIndex({ "user_id": 1 })
```

### Database Migration Script

Create `scripts/migrate_database.py`:

```python
#!/usr/bin/env python3
"""
Database Migration Script for DeepSeek Research
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMigrator:
    def __init__(self, mongodb_uri: str):
        self.client = AsyncIOMotorClient(mongodb_uri)
        self.db = self.client.deepschina
        
    async def create_collections(self):
        """Create required collections if they don't exist"""
        collections = ['web_content_cache', 'research_sessions', 'api_usage_logs']
        
        existing = await self.db.list_collection_names()
        
        for collection in collections:
            if collection not in existing:
                await self.db.create_collection(collection)
                logger.info(f"✅ Created collection: {collection}")
            else:
                logger.info(f"📋 Collection exists: {collection}")
    
    async def create_indexes(self):
        """Create required indexes"""
        
        # Web content cache indexes
        await self.db.web_content_cache.create_index([("url", 1)], unique=True)
        await self.db.web_content_cache.create_index([("timestamp", 1)])
        await self.db.web_content_cache.create_index([("keywords", 1)])
        
        # TTL index for cache expiry (configurable)
        cache_expiry_days = int(os.getenv('CACHE_EXPIRY_DAYS', 30))
        cache_expiry_seconds = cache_expiry_days * 24 * 60 * 60
        
        await self.db.web_content_cache.create_index(
            [("timestamp", 1)], 
            expireAfterSeconds=cache_expiry_seconds
        )
        
        # Research sessions indexes
        await self.db.research_sessions.create_index([("chat_id", 1)])
        await self.db.research_sessions.create_index([("timestamp", 1)])
        await self.db.research_sessions.create_index([("user_id", 1)])
        
        # API usage logs indexes
        await self.db.api_usage_logs.create_index([("timestamp", 1)])
        await self.db.api_usage_logs.create_index([("api_name", 1)])
        await self.db.api_usage_logs.create_index([("user_id", 1)])
        
        logger.info("✅ Created all required indexes")
    
    async def update_ttl_index(self, new_expiry_days: int):
        """Update TTL index with new expiry period"""
        
        # Drop existing TTL index
        try:
            await self.db.web_content_cache.drop_index("timestamp_1")
        except:
            pass  # Index might not exist
        
        # Create new TTL index
        new_expiry_seconds = new_expiry_days * 24 * 60 * 60
        await self.db.web_content_cache.create_index(
            [("timestamp", 1)],
            expireAfterSeconds=new_expiry_seconds
        )
        
        logger.info(f"✅ Updated TTL index to {new_expiry_days} days")
    
    async def migrate(self):
        """Run complete migration"""
        logger.info("🚀 Starting database migration...")
        
        await self.create_collections()
        await self.create_indexes()
        
        # Verify migration
        stats = await self.get_migration_stats()
        logger.info(f"📊 Migration completed: {stats}")
    
    async def get_migration_stats(self):
        """Get migration statistics"""
        collections = await self.db.list_collection_names()
        
        stats = {
            'collections_created': len([c for c in collections if c in ['web_content_cache', 'research_sessions', 'api_usage_logs']]),
            'total_collections': len(collections),
            'cache_entries': await self.db.web_content_cache.count_documents({}),
            'research_sessions': await self.db.research_sessions.count_documents({}),
            'api_logs': await self.db.api_usage_logs.count_documents({})
        }
        
        return stats
    
    async def close(self):
        """Close database connection"""
        self.client.close()

async def main():
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    
    migrator = DatabaseMigrator(mongodb_uri)
    
    try:
        await migrator.migrate()
    finally:
        await migrator.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Running Migration

```bash
# Run migration script
python scripts/migrate_database.py

# Run with specific cache expiry
CACHE_EXPIRY_DAYS=60 python scripts/migrate_database.py
```

## Deployment Procedures

### Development Deployment

```bash
# 1. Clone repository
git clone <repository-url>
cd deepschina

# 2. Setup environment
cp .env.example .env.development
# Edit .env.development with development API keys

# 3. Setup Python environment
./setup_venvs.sh
./activate_backend.sh

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run database migration
python scripts/migrate_database.py

# 6. Start development server
cd backend && ./run.sh

# 7. Test DeepSeek functionality
python backend/test_deepseek_integration.py
```

### Staging Deployment

```bash
# 1. Prepare staging environment
export ENV=staging
cp .env.example .env.staging

# 2. Configure staging-specific settings
cat <<EOF > .env.staging
NODE_ENV=staging
PORT=8100
MONGODB_URI=mongodb://staging-mongo:27017/deepschina_staging
DEEPSEEK_API_KEY=staging_api_key_here
GOOGLE_API_KEY=staging_google_key_here
GOOGLE_CSE_ID=staging_cse_id_here
BRIGHTDATA_API_KEY=staging_brightdata_key_here
CACHE_EXPIRY_DAYS=14
DEEPSEEK_RESEARCH_TIMEOUT=450
MAX_CONCURRENT_RESEARCH=2
EOF

# 3. Deploy with Docker Compose
docker-compose -f docker-compose.staging.yml up -d

# 4. Run database migration
docker-compose -f docker-compose.staging.yml exec web python scripts/migrate_database.py

# 5. Run validation tests
docker-compose -f docker-compose.staging.yml exec web python scripts/validate_config.py
```

### Production Deployment

#### Method 1: Docker Deployment

```bash
# 1. Build production image
docker build -t deepschina:latest -f Dockerfile.prod .

# 2. Create production configuration
cat <<EOF > .env.production
NODE_ENV=production
PORT=8100
MONGODB_URI=mongodb://prod-mongo-cluster:27017/deepschina
DEEPSEEK_API_KEY=prod_api_key_here
GOOGLE_API_KEY=prod_google_key_here
GOOGLE_CSE_ID=prod_cse_id_here
BRIGHTDATA_API_KEY=prod_brightdata_key_here
CACHE_EXPIRY_DAYS=30
DEEPSEEK_RESEARCH_TIMEOUT=600
MAX_CONCURRENT_RESEARCH=5
AUTH_SECRET_KEY=very_secure_secret_key
EOF

# 3. Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# 4. Run database migration
docker-compose -f docker-compose.prod.yml exec web python scripts/migrate_database.py

# 5. Verify deployment
docker-compose -f docker-compose.prod.yml exec web python scripts/validate_config.py
```

#### Method 2: Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepschina-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: deepschina
  template:
    metadata:
      labels:
        app: deepschina
    spec:
      containers:
      - name: deepschina
        image: deepschina:latest
        ports:
        - containerPort: 8100
        env:
        - name: PORT
          value: "8100"
        - name: NODE_ENV
          value: "production"
        envFrom:
        - configMapRef:
            name: deepschina-config
        - secretRef:
            name: deepschina-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi" 
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8100
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8100
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: deepschina-service
spec:
  selector:
    app: deepschina
  ports:
  - port: 80
    targetPort: 8100
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash
# Apply configuration
kubectl apply -f k8s-configmap.yml
kubectl apply -f k8s-secrets.yml
kubectl apply -f k8s-deployment.yml

# Run database migration
kubectl exec -it deployment/deepschina-app -- python scripts/migrate_database.py

# Monitor deployment
kubectl get pods
kubectl logs -f deployment/deepschina-app
```

## Cache Configuration Management

### Updating Cache Expiry in Production

When changing cache expiry settings in production:

```python
#!/usr/bin/env python3
"""
Script to update cache expiry configuration in production
"""

import asyncio
import os
from scripts.migrate_database import DatabaseMigrator

async def update_cache_expiry(new_expiry_days: int):
    """Update cache expiry in production"""
    
    print(f"🔄 Updating cache expiry to {new_expiry_days} days...")
    
    mongodb_uri = os.getenv('MONGODB_URI')
    migrator = DatabaseMigrator(mongodb_uri)
    
    try:
        # Update TTL index
        await migrator.update_ttl_index(new_expiry_days)
        
        # Update environment variable
        print(f"✅ TTL index updated")
        print(f"🔧 Update CACHE_EXPIRY_DAYS={new_expiry_days} in environment and restart application")
        
    finally:
        await migrator.close()

if __name__ == "__main__":
    new_days = int(input("Enter new cache expiry in days: "))
    asyncio.run(update_cache_expiry(new_days))
```

### Cache Maintenance

```bash
# Clean up expired cache entries manually
python -c "
import asyncio
from app.service.mongodb_service import MongoDBService

async def cleanup():
    service = MongoDBService()
    result = await service.cleanup_expired_cache(days=30)
    print(f'Cleaned up {result} expired entries')

asyncio.run(cleanup())
"

# Monitor cache size and performance
python -c "
import asyncio
from app.service.mongodb_service import MongoDBService

async def stats():
    service = MongoDBService()
    stats = await service.get_cache_stats()
    print(f'Cache stats: {stats}')

asyncio.run(stats())
"
```

## Monitoring and Health Checks

### Application Health Check

```python
# scripts/health_check.py
#!/usr/bin/env python3
"""
Comprehensive health check for DeepSeek Research
"""

import asyncio
import aiohttp
import os
import sys
from datetime import datetime

class HealthChecker:
    def __init__(self, base_url="http://localhost:8100"):
        self.base_url = base_url
        self.checks = []
        
    async def check_api_health(self):
        """Check API health endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health", timeout=10) as response:
                    if response.status == 200:
                        self.checks.append(("API Health", True, "OK"))
                        return True
                    else:
                        self.checks.append(("API Health", False, f"HTTP {response.status}"))
                        return False
        except Exception as e:
            self.checks.append(("API Health", False, str(e)))
            return False
    
    async def check_deepseek_config(self):
        """Check DeepSeek research configuration"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/deepseek/config", timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('enabled', False):
                            self.checks.append(("DeepSeek Config", True, "Enabled"))
                            return True
                        else:
                            self.checks.append(("DeepSeek Config", False, "Disabled"))
                            return False
                    else:
                        self.checks.append(("DeepSeek Config", False, f"HTTP {response.status}"))
                        return False
        except Exception as e:
            self.checks.append(("DeepSeek Config", False, str(e)))
            return False
    
    async def check_database(self):
        """Check database connectivity"""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
            client = AsyncIOMotorClient(mongodb_uri)
            
            # Test connection
            await client.admin.command('ping')
            
            # Check collections
            db = client.deepschina
            collections = await db.list_collection_names()
            required = ['web_content_cache', 'research_sessions']
            missing = [c for c in required if c not in collections]
            
            if not missing:
                self.checks.append(("Database", True, f"{len(collections)} collections"))
                return True
            else:
                self.checks.append(("Database", False, f"Missing: {missing}"))
                return False
                
        except Exception as e:
            self.checks.append(("Database", False, str(e)))
            return False
    
    async def run_all_checks(self):
        """Run all health checks"""
        print(f"🏥 Health Check Report - {datetime.now().isoformat()}")
        print("=" * 50)
        
        checks = [
            self.check_api_health(),
            self.check_deepseek_config(),
            self.check_database()
        ]
        
        await asyncio.gather(*checks, return_exceptions=True)
        
        # Print results
        all_passed = True
        for check_name, passed, message in self.checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} {check_name}: {message}")
            if not passed:
                all_passed = False
        
        print("=" * 50)
        if all_passed:
            print("🎉 All health checks passed!")
            return 0
        else:
            print("⚠️  Some health checks failed!")
            return 1

if __name__ == "__main__":
    checker = HealthChecker()
    exit_code = asyncio.run(checker.run_all_checks())
    sys.exit(exit_code)
```

### Monitoring Script

```bash
# scripts/monitor.sh
#!/bin/bash

echo "📊 DeepSeek Research Monitoring Report"
echo "======================================="
echo "Timestamp: $(date)"
echo ""

# System resources
echo "💻 System Resources:"
echo "- CPU Usage: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1)%"
echo "- Memory: $(free -h | awk '/^Mem/ {print $3 "/" $2}')"
echo "- Disk: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}')"
echo ""

# Application status
echo "🚀 Application Status:"
if pgrep -f "tornado_main.py" > /dev/null; then
    echo "- Status: Running ✅"
    echo "- PID: $(pgrep -f tornado_main.py)"
    echo "- Uptime: $(ps -o etime= -p $(pgrep -f tornado_main.py) | tr -d ' ')"
else
    echo "- Status: Not Running ❌"
fi
echo ""

# Database status
echo "💾 Database Status:"
if pgrep mongod > /dev/null; then
    echo "- MongoDB: Running ✅"
    echo "- Connections: $(mongo --eval 'db.serverStatus().connections' --quiet | grep current)"
else
    echo "- MongoDB: Not Running ❌"
fi
echo ""

# Cache statistics
echo "📋 Cache Statistics:"
python3 -c "
import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient

async def get_stats():
    client = AsyncIOMotorClient(os.getenv('MONGODB_URI', 'mongodb://localhost:27017'))
    db = client.deepschina
    
    cache_count = await db.web_content_cache.count_documents({})
    session_count = await db.research_sessions.count_documents({})
    
    print(f'- Cache Entries: {cache_count}')
    print(f'- Research Sessions: {session_count}')

asyncio.run(get_stats())
" 2>/dev/null || echo "- Cache: Unable to connect"

echo ""

# Recent errors
echo "⚠️  Recent Errors (last 100 lines):"
if [ -f logs/deepseek_research.log ]; then
    tail -100 logs/deepseek_research.log | grep -i error | tail -5 || echo "- No recent errors"
else
    echo "- Log file not found"
fi
```

Make monitoring script executable:

```bash
chmod +x scripts/monitor.sh
```

## Performance Optimization

### Production Performance Settings

```bash
# Production environment variables
export CACHE_EXPIRY_DAYS=30
export DEEPSEEK_RESEARCH_TIMEOUT=600
export MAX_CONCURRENT_RESEARCH=5
export MAX_CONTENT_TOKENS=50000
export MAX_CONTENT_LENGTH=2000

# System tuning
ulimit -n 65536  # Increase file descriptor limit
export PYTHONOPTIMIZE=1  # Enable Python optimizations
```

### Load Balancing Configuration

```nginx
# nginx.conf for load balancing
upstream deepschina_backend {
    server 127.0.0.1:8100;
    server 127.0.0.1:8101;
    server 127.0.0.1:8102;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://deepschina_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # SSE specific settings
        proxy_buffering off;
        proxy_cache off;
    }
}
```

## Rollback Procedures

See the dedicated rollback plan document for detailed rollback procedures including database rollback, configuration restoration, and emergency procedures.

## Post-Deployment Validation

### Validation Checklist

After deployment, verify:

- [ ] Application starts successfully
- [ ] Health check endpoint responds
- [ ] DeepSeek button appears in UI
- [ ] Research functionality works end-to-end
- [ ] Cache operations function correctly
- [ ] API limits and quotas are sufficient
- [ ] Monitoring and logging are operational
- [ ] Performance meets expectations

### Automated Validation

```bash
# Run complete validation suite
python scripts/validate_config.py
python scripts/health_check.py
python backend/test_deepseek_performance_benchmark.py
```

This deployment guide ensures reliable and consistent deployment of the DeepSeek Research functionality across all environments.