# DeepSeek Research Rollback Plan

## Overview

This document outlines comprehensive rollback procedures for the DeepSeek Research integration, covering emergency scenarios, database rollback, configuration restoration, and recovery processes.

## Emergency Contacts

**Primary Contact**: Development Team Lead
**Secondary Contact**: System Administrator  
**Escalation Contact**: Technical Director

**Response Times**:
- Critical (Service Down): 15 minutes
- High (Functionality Impacted): 1 hour
- Medium (Performance Issues): 4 hours

## Rollback Triggers

### When to Execute Rollback

**Immediate Rollback Required**:
- Complete service outage lasting >15 minutes
- Data corruption in research cache
- Security breach or vulnerability exploitation
- Critical API failures affecting core functionality
- Memory leaks or resource exhaustion

**Planned Rollback Scenarios**:
- Performance degradation >50% from baseline
- Error rates >20% for research requests
- User complaints about functionality loss
- Failed deployment validation
- External API quota exhaustion

## Pre-Rollback Assessment

### Quick Diagnostic Checklist

```bash
# 1. Check service status
curl -I http://localhost:8100/health
echo "Service Status: $?"

# 2. Check DeepSeek configuration
curl http://localhost:8100/api/deepseek/config | jq '.enabled'

# 3. Check database connectivity
mongo --eval "db.adminCommand('ping')" --quiet

# 4. Check system resources
free -m | head -2
df -h | head -2

# 5. Check recent errors
tail -50 logs/deepseek_research.log | grep -i error | wc -l
```

### Assessment Decision Matrix

| Issue Type | Severity | Action |
|------------|----------|---------|
| Service completely down | Critical | Immediate rollback |
| Research not working | High | Disable feature rollback |
| Slow performance | Medium | Monitor and plan rollback |
| Cache issues | Low-Medium | Cache rollback only |

## Rollback Procedures

### Level 1: Feature Disable (Fastest - 2 minutes)

**When to Use**: Research functionality broken but main application works

```bash
# 1. Disable DeepSeek research feature
export DEEPSEEK_RESEARCH_ENABLED=false

# 2. Restart application with feature disabled
pkill -f tornado_main.py
cd backend && nohup ./run.sh &

# 3. Verify feature is disabled
curl http://localhost:8100/api/deepseek/config | jq '.enabled'
# Should return: false

# 4. Test main application functionality
curl http://localhost:8100/chat/message -X POST \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "chat_id": "test"}'
```

**Verification**:
- DeepSeek button disappears from UI
- Regular chat continues to work
- No research-related errors in logs

### Level 2: Configuration Rollback (5 minutes)

**When to Use**: Configuration changes caused issues

```bash
# 1. Backup current configuration
cp .env .env.failed.$(date +%Y%m%d_%H%M%S)

# 2. Restore previous configuration
cp .env.backup .env
# OR restore from git
git checkout HEAD~1 -- .env.example

# 3. Restore database configuration
python scripts/restore_db_config.py --backup-date=$(date -d "1 hour ago" +%Y%m%d_%H)

# 4. Restart application
./restart.sh

# 5. Run validation
python scripts/validate_config.py
```

### Level 3: Code Rollback (10 minutes)

**When to Use**: New code deployment caused issues

```bash
# 1. Identify last working commit
git log --oneline -10
LAST_GOOD_COMMIT="abc1234"  # Replace with actual commit

# 2. Create rollback branch
git checkout -b rollback-$(date +%Y%m%d_%H%M%S)
git reset --hard $LAST_GOOD_COMMIT

# 3. Backup current deployment
mv backend backend.failed.$(date +%Y%m%d_%H%M%S)

# 4. Deploy rolled back version
git archive HEAD | tar -x -C backend/

# 5. Restore dependencies if needed
cd backend && pip install -r requirements.txt

# 6. Restart services
./restart.sh

# 7. Verify rollback
python scripts/health_check.py
```

### Level 4: Database Rollback (15-30 minutes)

**When to Use**: Database corruption or migration issues

#### Database Backup Verification

```bash
# 1. Check available backups
ls -la backups/mongodb/
# Look for: deepschina_backup_YYYYMMDD_HHMMSS/

# 2. Verify backup integrity
mongodump --db deepschina --collection web_content_cache --out /tmp/test_restore
echo "Backup verification: $?"

# 3. Check backup size and date
du -sh backups/mongodb/deepschina_backup_*
```

#### Database Restoration Process

```bash
# 1. Stop application to prevent new writes
pkill -f tornado_main.py

# 2. Create emergency backup of current state
mongodump --db deepschina --out backups/emergency_$(date +%Y%m%d_%H%M%S)

# 3. Identify restore point
RESTORE_BACKUP="backups/mongodb/deepschina_backup_20250115_1200"

# 4. Drop affected collections
mongo deepschina --eval "
db.web_content_cache.drop();
db.research_sessions.drop();
db.api_usage_logs.drop();
"

# 5. Restore from backup
mongorestore --db deepschina $RESTORE_BACKUP/deepschina/

# 6. Recreate indexes
python scripts/migrate_database.py

# 7. Restart application
cd backend && ./run.sh

# 8. Verify restoration
python scripts/health_check.py
```

### Level 5: Complete System Rollback (30-60 minutes)

**When to Use**: Multiple systems affected, complete rollback needed

```bash
# 1. Stop all services
docker-compose down
# OR
systemctl stop deepschina
systemctl stop mongodb
systemctl stop meilisearch

# 2. Restore application code
git checkout $LAST_WORKING_TAG
git clean -fd

# 3. Restore database
mongorestore --drop --db deepschina backups/full_backup_$(date +%Y%m%d)/

# 4. Restore search index
curl -X DELETE http://localhost:7701/indexes/messages
curl -X POST http://localhost:7701/indexes \
  -H "Content-Type: application/json" \
  -d '{"uid": "messages", "primaryKey": "id"}'

# 5. Restore configuration
cp config_backups/production_$(date +%Y%m%d).env .env

# 6. Start services
docker-compose up -d
# OR
systemctl start mongodb
systemctl start meilisearch
systemctl start deepschina

# 7. Run complete validation
python scripts/health_check.py
python backend/test_api.sh
```

## Database Rollback Procedures

### Automated Database Backup

```python
# scripts/backup_database.py
#!/usr/bin/env python3
"""
Automated database backup for rollback purposes
"""

import os
import subprocess
import logging
from datetime import datetime

def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/mongodb/deepschina_backup_{timestamp}"
    
    try:
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)
        
        # Run mongodump
        cmd = [
            "mongodump",
            "--db", "deepschina", 
            "--out", backup_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f"✅ Backup created: {backup_dir}")
            return backup_dir
        else:
            logging.error(f"❌ Backup failed: {result.stderr}")
            return None
            
    except Exception as e:
        logging.error(f"❌ Backup error: {e}")
        return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backup_path = create_backup()
    print(backup_path)
```

### Incremental Cache Rollback

```python
# scripts/rollback_cache.py
#!/usr/bin/env python3
"""
Rollback cache data to specific timestamp
"""

import asyncio
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient

async def rollback_cache(hours_back=24):
    """Remove cache entries newer than specified hours"""
    
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.deepschina
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    
    # Remove recent cache entries
    result = await db.web_content_cache.delete_many(
        {"timestamp": {"$gte": cutoff_time}}
    )
    
    print(f"Removed {result.deleted_count} cache entries newer than {cutoff_time}")
    
    # Remove recent research sessions
    result = await db.research_sessions.delete_many(
        {"timestamp": {"$gte": cutoff_time}}
    )
    
    print(f"Removed {result.deleted_count} research sessions newer than {cutoff_time}")

if __name__ == "__main__":
    hours = int(input("Hours to rollback (default 24): ") or "24")
    asyncio.run(rollback_cache(hours))
```

## Recovery Procedures

### Post-Rollback Validation

```bash
# scripts/post_rollback_validation.sh
#!/bin/bash

echo "🔍 Post-Rollback Validation Report"
echo "=================================="
echo "Timestamp: $(date)"
echo ""

# 1. Service health
echo "🏥 Service Health:"
if python scripts/health_check.py > /dev/null 2>&1; then
    echo "- Health Check: PASS ✅"
else
    echo "- Health Check: FAIL ❌"
fi

# 2. Basic functionality
echo "🚀 Basic Functionality:"
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8100/)
if [ "$response" = "200" ]; then
    echo "- Main Application: PASS ✅"
else
    echo "- Main Application: FAIL ❌ (HTTP $response)"
fi

# 3. Database connectivity
echo "💾 Database:"
if mongo --eval "db.adminCommand('ping')" --quiet > /dev/null 2>&1; then
    echo "- MongoDB Connection: PASS ✅"
else
    echo "- MongoDB Connection: FAIL ❌"
fi

# 4. Feature status
echo "🔬 DeepSeek Research:"
enabled=$(curl -s http://localhost:8100/api/deepseek/config | jq -r '.enabled' 2>/dev/null)
if [ "$enabled" = "true" ]; then
    echo "- DeepSeek Feature: ENABLED ✅"
elif [ "$enabled" = "false" ]; then
    echo "- DeepSeek Feature: DISABLED ⚠️"
else
    echo "- DeepSeek Feature: ERROR ❌"
fi

# 5. Performance baseline
echo "⚡ Performance:"
start_time=$(date +%s%N)
curl -s http://localhost:8100/health > /dev/null
end_time=$(date +%s%N)
response_time=$(echo "scale=3; ($end_time - $start_time) / 1000000" | bc)
echo "- Response Time: ${response_time}ms"

echo ""
echo "📊 Validation Complete"
```

### Recovery Monitoring

```python
# scripts/recovery_monitor.py
#!/usr/bin/env python3
"""
Monitor system recovery after rollback
"""

import asyncio
import aiohttp
import time
import logging
from datetime import datetime

class RecoveryMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.checks = []
        
    async def monitor_recovery(self, duration_minutes=30):
        """Monitor system for specified duration after rollback"""
        
        print(f"🔍 Starting recovery monitoring for {duration_minutes} minutes...")
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            await self.run_health_checks()
            await self.check_error_rates()
            await self.monitor_performance()
            
            print(f"✅ Check completed at {datetime.now().strftime('%H:%M:%S')}")
            await asyncio.sleep(60)  # Check every minute
        
        print("📊 Recovery monitoring completed")
        self.generate_report()
    
    async def run_health_checks(self):
        """Run basic health checks"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8100/health', timeout=10) as resp:
                    if resp.status == 200:
                        self.checks.append(('health', True, time.time()))
                    else:
                        self.checks.append(('health', False, time.time()))
        except Exception as e:
            self.checks.append(('health', False, time.time()))
    
    def generate_report(self):
        """Generate recovery monitoring report"""
        total_checks = len(self.checks)
        successful_checks = sum(1 for check, success, _ in self.checks if success)
        
        print(f"""
🎯 Recovery Monitoring Report
============================
Duration: {(time.time() - self.start_time)/60:.1f} minutes
Total Checks: {total_checks}
Successful: {successful_checks}
Success Rate: {(successful_checks/total_checks*100):.1f}%
        """)

if __name__ == "__main__":
    monitor = RecoveryMonitor()
    duration = int(input("Monitor duration in minutes (default 30): ") or "30")
    asyncio.run(monitor.monitor_recovery(duration))
```

## Communication Plan

### Internal Communication

**Rollback Initiated**:
```
SUBJECT: [URGENT] DeepSeek Research Rollback Initiated

Team,

DeepSeek Research rollback has been initiated due to: [REASON]

Rollback Level: [1-5]
Expected Duration: [TIME]
Impact: [USER IMPACT]

Status updates will be provided every 15 minutes.

- [NAME], [TIME]
```

**Rollback Completed**:
```
SUBJECT: [RESOLVED] DeepSeek Research Rollback Completed

Team,

DeepSeek Research rollback has been completed successfully.

Resolution: [WHAT WAS DONE]
Current Status: [FUNCTIONAL STATUS]
Monitoring: [ONGOING MONITORING PLAN]

Post-mortem scheduled for: [DATE/TIME]

- [NAME], [TIME]
```

### User Communication

**Service Disruption Notice**:
```
🔧 Temporary Service Maintenance

We're currently experiencing issues with our advanced research feature. 
We're working to resolve this quickly.

- Regular chat functionality continues to work normally
- Research feature temporarily unavailable
- Estimated resolution: [TIME]

We apologize for any inconvenience.
```

**Resolution Notice**:
```
✅ Service Restored

Our advanced research feature has been restored and is functioning normally.

Thank you for your patience during the maintenance period.
```

## Prevention Measures

### Pre-Deployment Validation

```bash
# scripts/pre_deploy_validation.sh
#!/bin/bash

echo "🔍 Pre-Deployment Validation"
echo "============================"

# 1. Run all tests
echo "🧪 Running Tests:"
python backend/test_deepseek_performance_benchmark.py > /dev/null 2>&1
echo "- Performance: $([[ $? -eq 0 ]] && echo "PASS ✅" || echo "FAIL ❌")"

python backend/test_deepseek_timeout.py > /dev/null 2>&1
echo "- Timeout: $([[ $? -eq 0 ]] && echo "PASS ✅" || echo "FAIL ❌")"

# 2. Configuration validation
echo "⚙️ Configuration:"
python scripts/validate_config.py > /dev/null 2>&1
echo "- Config: $([[ $? -eq 0 ]] && echo "PASS ✅" || echo "FAIL ❌")"

# 3. Database migration validation
echo "💾 Database:"
python scripts/migrate_database.py --dry-run > /dev/null 2>&1
echo "- Migration: $([[ $? -eq 0 ]] && echo "PASS ✅" || echo "FAIL ❌")"

echo ""
echo "✅ Pre-deployment validation complete"
```

### Automated Backup Schedule

```bash
# Create automated backup cron job
echo "0 */6 * * * /path/to/scripts/backup_database.py" | crontab -

# Weekly full backup
echo "0 2 * * 0 /path/to/scripts/full_backup.sh" | crontab -

# Daily configuration backup
echo "0 1 * * * cp .env backups/config/env_$(date +\%Y\%m\%d)" | crontab -
```

This rollback plan ensures rapid recovery from any issues with the DeepSeek Research integration while maintaining service continuity and data integrity.