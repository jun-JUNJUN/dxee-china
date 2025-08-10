# DeepSeek Research Troubleshooting Guide

## Overview

This guide provides solutions for common issues encountered with the DeepSeek Research functionality, including setup problems, runtime errors, performance issues, and configuration challenges.

## Quick Diagnostic Checklist

Before diving into specific issues, run through this checklist:

1. ✅ All required environment variables are set
2. ✅ External APIs (Google, Bright Data, DeepSeek) are accessible  
3. ✅ MongoDB is running and accessible
4. ✅ DeepSeek button appears in the UI
5. ✅ Network connectivity is stable
6. ✅ No rate limits have been exceeded

## Environment Setup Issues

### Missing Environment Variables

**Symptom**: DeepSeek button is disabled or research fails to start

**Error Messages**:
- `"DEEPSEEK_API_KEY environment variable is required"`
- `"Google Search API not configured"`
- `"Configuration incomplete"`

**Solution**:
1. Check your `.env` file contains all required variables:
   ```bash
   DEEPSEEK_API_KEY=your_deepseek_key
   GOOGLE_API_KEY=your_google_key
   GOOGLE_CSE_ID=your_cse_id
   BRIGHTDATA_API_KEY=your_brightdata_key
   MONGODB_URI=mongodb://localhost:27017
   CACHE_EXPIRY_DAYS=30
   DEEPSEEK_RESEARCH_TIMEOUT=600
   ```

2. Verify environment variables are loaded:
   ```bash
   # In your backend environment
   echo $DEEPSEEK_API_KEY
   echo $GOOGLE_API_KEY
   echo $GOOGLE_CSE_ID
   echo $BRIGHTDATA_API_KEY
   ```

3. Restart the application after adding variables

### Invalid API Keys

**Symptom**: Research starts but fails during web search or content extraction

**Error Messages**:
- `"HTTP 403: API key invalid"`
- `"Authentication failed"`
- `"Rate limit exceeded"`

**Solution**:
1. **DeepSeek API Key**:
   - Verify key is active at https://platform.deepseek.com
   - Check account has sufficient credits
   - Ensure key has chat completions permission

2. **Google Search API Key**:
   - Enable Custom Search API in Google Cloud Console
   - Verify API key has Custom Search permission
   - Check daily quota limits

3. **Google Custom Search Engine ID**:
   - Create CSE at https://cse.google.com
   - Enable "Search the entire web" option
   - Copy the Search Engine ID (not the API key)

4. **Bright Data API Key**:
   - Verify subscription is active
   - Check API quota and usage limits
   - Test key with a simple API call

### MongoDB Connection Issues

**Symptom**: Research fails to initialize or cache statistics show errors

**Error Messages**:
- `"MongoDB connection failed"`
- `"Cache initialization error"`
- `"Database timeout"`

**Solution**:
1. Verify MongoDB is running:
   ```bash
   # Check MongoDB status
   brew services list | grep mongodb  # macOS
   sudo systemctl status mongod       # Linux
   ```

2. Test connection manually:
   ```bash
   # Connect to MongoDB
   mongosh mongodb://localhost:27017
   
   # List databases
   show dbs
   
   # Test write operation
   use testdb
   db.test.insertOne({test: "connection"})
   ```

3. Check MongoDB configuration:
   - Ensure MongoDB is listening on correct port (default: 27017)
   - Verify no authentication issues
   - Check disk space and memory availability

## Runtime Errors

### Research Timeout Issues

**Symptom**: Research stops after exactly 10 minutes with partial results

**Error Messages**:
- `"Research partially completed due to timeout"`
- `"Time limit exceeded"`
- `"⏰ Time limit reached"`

**Solution**:
1. **For Users**:
   - Break complex questions into simpler parts
   - Try more specific search terms
   - Retry with better network connection

2. **For Administrators**:
   ```bash
   # Increase timeout (up to 15 minutes max recommended)
   export DEEPSEEK_RESEARCH_TIMEOUT=900
   
   # Monitor performance
   tail -f logs/deepseek_research.log
   ```

3. **Optimization Tips**:
   - Clean up old cache entries to improve performance
   - Monitor external API response times
   - Consider upgrading API plans for better rate limits

### Relevance Evaluation Failures

**Symptom**: Research finds sources but fails to evaluate relevance

**Error Messages**:
- `"Relevance evaluation failed"`
- `"No high-relevance content found"`
- `"Threshold not met"`

**Solution**:
1. **Check DeepSeek API Status**:
   ```python
   # Test DeepSeek API connection
   import openai
   client = openai.AsyncOpenAI(api_key="your_key", base_url="https://api.deepseek.com")
   
   response = await client.chat.completions.create(
       model="deepseek-chat",
       messages=[{"role": "user", "content": "Test message"}]
   )
   print(response.choices[0].message.content)
   ```

2. **Adjust Relevance Threshold**:
   - Default threshold is 7.0/10
   - For broader results, temporarily lower threshold
   - Monitor relevance scores in logs to calibrate

3. **Improve Question Quality**:
   - Use specific, well-defined questions
   - Include relevant keywords and context
   - Avoid overly broad or vague queries

### Content Extraction Failures

**Symptom**: Web search succeeds but content extraction fails

**Error Messages**:
- `"Content extraction failed"`
- `"Bright Data API error"`
- `"Empty API response"`

**Solution**:
1. **Test Bright Data API**:
   ```bash
   curl -X POST https://api.brightdata.com/datasets/v3/scrape \
     -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
       "urls": ["https://example.com"],
       "format": "json"
     }'
   ```

2. **Check API Limits**:
   - Verify monthly/daily request limits
   - Monitor API usage in Bright Data dashboard
   - Consider upgrading plan if limits exceeded

3. **Fallback Options**:
   - System falls back to basic extraction automatically
   - Users see notification about limited extraction
   - Results may have less detailed content

## Performance Issues

### Slow Research Response

**Symptom**: Research takes longer than expected to complete

**Expected Times**:
- Cache hits: 5-15 seconds
- Fresh research: 30-90 seconds
- Complex queries: 60-300 seconds

**Solution**:
1. **Check Cache Performance**:
   ```python
   # Monitor cache hit/miss ratio
   cache_stats = await mongodb_service.get_cache_stats()
   hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses'])
   print(f"Cache hit rate: {hit_rate:.2%}")
   ```

2. **Optimize Cache Configuration**:
   ```bash
   # Increase cache retention
   export CACHE_EXPIRY_DAYS=60
   
   # Pre-populate cache for common topics
   python scripts/cache_warmup.py
   ```

3. **Network Optimization**:
   - Check internet connection speed
   - Verify DNS resolution for external APIs
   - Consider using CDN for static assets

### High Memory Usage

**Symptom**: Server memory usage increases during research sessions

**Solution**:
1. **Monitor Memory Usage**:
   ```bash
   # Check memory usage during research
   ps aux | grep python
   top -p $(pgrep -f "tornado_main.py")
   ```

2. **Optimize Content Processing**:
   - Research service automatically summarizes large content
   - Token limits prevent excessive memory usage
   - Cleanup happens after each session

3. **System Limits**:
   ```bash
   # Set memory limits for the process
   ulimit -m 2097152  # 2GB limit
   
   # Monitor with system tools
   htop
   ```

## Configuration Issues

### Cache Expiry Problems

**Symptom**: Old cached content affects research quality

**Error Messages**:
- `"Stale cache detected"`
- `"Cache consistency issues"`

**Solution**:
1. **Manual Cache Cleanup**:
   ```python
   # Clear expired cache entries
   python -c "
   import asyncio
   from app.service.mongodb_service import MongoDBService
   
   async def cleanup():
       service = MongoDBService()
       await service.cleanup_expired_cache(days=30)
   
   asyncio.run(cleanup())
   "
   ```

2. **Adjust Cache Settings**:
   ```bash
   # Shorter cache expiry for rapidly changing topics
   export CACHE_EXPIRY_DAYS=7
   
   # Longer cache for stable reference material
   export CACHE_EXPIRY_DAYS=90
   ```

3. **Cache Monitoring**:
   - Regular cache statistics review
   - Alert on unusually low hit rates
   - Periodic cache cleanup scheduling

### Concurrent Session Limits

**Symptom**: Research requests are queued or rejected

**Error Messages**:
- `"Maximum concurrent sessions reached"`
- `"Request queued"`
- `"Rate limit exceeded"`

**Solution**:
1. **Adjust Limits**:
   ```bash
   # Increase concurrent session limit (be cautious)
   export MAX_CONCURRENT_RESEARCH=5
   ```

2. **Load Balancing**:
   - Monitor active session count
   - Implement session queuing with timeout
   - Consider horizontal scaling

3. **User Communication**:
   - Display current session count to users
   - Provide estimated wait times
   - Allow session cancellation

## User Interface Issues

### DeepSeek Button Not Visible

**Symptom**: Research functionality not accessible in UI

**Solution**:
1. **Check Feature Flags**:
   ```javascript
   // In browser console
   console.log(window.deepseekConfig);
   ```

2. **Verify API Configuration**:
   - Visit `/api/deepseek/config` endpoint
   - Check `enabled: true` in response
   - Verify all APIs show as configured

3. **Browser Issues**:
   - Clear browser cache and cookies
   - Disable ad blockers temporarily
   - Try different browser or incognito mode

### Progress Updates Not Showing

**Symptom**: Research runs but no progress indicators appear

**Solution**:
1. **Check SSE Connection**:
   ```javascript
   // Monitor Server-Sent Events in browser dev tools
   // Network tab -> EventSource connections
   ```

2. **Firewall/Proxy Issues**:
   - SSE requires persistent HTTP connection
   - Check corporate firewalls
   - Verify proxy settings don't block streams

3. **Fallback Mode**:
   - System should fall back to polling mode
   - Check console for fallback notifications

## API Integration Issues

### External API Rate Limits

**Symptom**: Research fails with rate limit errors

**Error Messages**:
- `"Google API quota exceeded"`
- `"Bright Data rate limit"`
- `"Too many requests"`

**Solution**:
1. **Monitor API Usage**:
   ```bash
   # Check API usage logs
   grep "rate limit" logs/deepseek_research.log
   
   # Monitor quota usage
   curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
     "https://www.googleapis.com/customsearch/v1?key=$GOOGLE_API_KEY&cx=$GOOGLE_CSE_ID&q=test"
   ```

2. **Implement Backoff Strategy**:
   - Service automatically retries with exponential backoff
   - Temporary degradation to cached results only
   - User notification of reduced functionality

3. **Upgrade API Plans**:
   - Consider higher quota tiers
   - Monitor usage patterns to optimize
   - Implement usage alerts

### Network Connectivity Issues

**Symptom**: Intermittent failures connecting to external APIs

**Solution**:
1. **Test Connectivity**:
   ```bash
   # Test API endpoints
   curl -I https://api.deepseek.com
   curl -I https://www.googleapis.com/customsearch/v1
   curl -I https://api.brightdata.com
   ```

2. **DNS Resolution**:
   ```bash
   # Check DNS resolution
   nslookup api.deepseek.com
   nslookup www.googleapis.com
   nslookup api.brightdata.com
   ```

3. **Firewall Configuration**:
   - Ensure outbound HTTPS (443) is allowed
   - Whitelist API domains if necessary
   - Check for corporate proxy requirements

## Logging and Debugging

### Enable Debug Logging

```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG

# View real-time logs
tail -f logs/deepseek_research.log | grep -E "(ERROR|WARNING|DEBUG)"
```

### Key Log Locations

- **Application logs**: `logs/tornado.log`
- **Research logs**: `logs/deepseek_research.log`  
- **MongoDB logs**: Check MongoDB log directory
- **System logs**: `/var/log/syslog` (Linux) or Console app (macOS)

### Performance Profiling

```python
# Add performance profiling
import cProfile
import pstats

# Profile research session
profiler = cProfile.Profile()
profiler.enable()

# ... run research ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Getting Help

### Self-Service Resources

1. **Check System Status**:
   - Visit `/health` endpoint for system status
   - Monitor `/api/deepseek/config` for configuration issues

2. **Review Logs**:
   - Application logs contain detailed error information
   - Research logs show step-by-step progress
   - MongoDB logs reveal database issues

3. **Test Components**:
   - Run individual test files to isolate issues
   - Use provided diagnostic scripts
   - Test external API connections manually

### Support Information to Provide

When contacting support, include:

1. **Environment Details**:
   - Operating system and version
   - Python version and dependencies
   - MongoDB version
   - Browser and version (for UI issues)

2. **Error Information**:
   - Complete error messages
   - Timestamp of issue
   - Chat ID or session identifier
   - Steps to reproduce

3. **Configuration**:
   - Environment variables (sanitized)
   - API configuration status
   - Cache statistics
   - System resource usage

4. **Logs** (last 100 lines):
   ```bash
   tail -100 logs/deepseek_research.log
   tail -100 logs/tornado.log
   ```

### Emergency Recovery

If DeepSeek Research is completely non-functional:

1. **Disable Feature**:
   ```bash
   # Temporarily disable DeepSeek research
   export DEEPSEEK_RESEARCH_ENABLED=false
   ```

2. **Restart Services**:
   ```bash
   # Restart application
   ./restart.sh
   
   # Or manual restart
   pkill -f tornado_main.py
   cd backend && ./run.sh
   ```

3. **Database Recovery**:
   ```bash
   # If MongoDB issues
   sudo systemctl restart mongod
   
   # Check database integrity
   mongosh --eval "db.runCommand({dbStats: 1})"
   ```

4. **Clean Restart**:
   ```bash
   # Clear all caches and restart
   rm -rf logs/*
   rm -rf __pycache__/
   ./setup_venvs.sh
   cd backend && ./run.sh
   ```

This troubleshooting guide covers the most common issues. For additional support or complex problems, check the project documentation or contact the development team with the diagnostic information outlined above.