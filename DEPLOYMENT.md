# 🚀 Production Deployment Guide

This guide covers deploying the Multi-modal RAG system in production.

## 📋 Pre-Deployment Checklist

### Security

- [ ] Generate strong random `SECRET_KEY`
- [ ] Change `API_KEY` from default
- [ ] Review CORS settings
- [ ] Enable HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Disable debug mode
- [ ] Review file upload limits

### Infrastructure

- [ ] Redis setup with persistence
- [ ] ChromaDB persistent storage
- [ ] Backup strategy
- [ ] Monitoring setup
- [ ] Logging configuration
- [ ] Error tracking (Sentry)

### Performance

- [ ] Configure worker count
- [ ] Set appropriate timeouts
- [ ] Enable caching
- [ ] Configure rate limiting
- [ ] Optimize database indexes

## 🔐 Security Configuration

### 1. Generate Secure Keys
```bash
# Generate SECRET_KEY (64 hex characters)
python -c "import secrets; print(secrets.token_hex(32))"

# Generate API_KEY (32 hex characters)
python -c "import secrets; print(secrets.token_hex(16))"
```

Update `.env`:
```bash
SECRET_KEY=<generated-secret-key>
API_KEY=<generated-api-key>
```

### 2. CORS Configuration

In production, specify exact origins:
```python
# config.py
ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://www.yourdomain.com"
]
```

### 3. API Key Security

Store API keys securely:
```bash
# Use environment variables
export API_KEY="your-secure-key"

# Or use secrets management
# AWS Secrets Manager, HashiCorp Vault, etc.
```

## 🐳 Docker Deployment

### 1. Create `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data figures chroma_db

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "run_server.py"]
```

### 2. Create `docker-compose.yml`
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: rag_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  app:
    build: .
    container_name: rag_app
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    volumes:
      - ./data:/app/data
      - ./figures:/app/figures
      - ./chroma_db:/app/chroma_db
    depends_on:
      - redis
    env_file:
      - .env

volumes:
  redis_data:
```

### 3. Deploy with Docker
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

## 🖥️ Systemd Service (Linux)

### 1. Create Service File
```bash
sudo nano /etc/systemd/system/rag-system.service
```
```ini
[Unit]
Description=Multi-modal RAG System
After=network.target redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/multimodal_rag
Environment="PATH=/opt/multimodal_rag/venv/bin"
ExecStart=/opt/multimodal_rag/venv/bin/python run_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 2. Enable and Start
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable rag-system

# Start service
sudo systemctl start rag-system

# Check status
sudo systemctl status rag-system

# View logs
sudo journalctl -u rag-system -f
```

## 🔄 Nginx Reverse Proxy

### 1. Install Nginx
```bash
sudo apt-get install nginx
```

### 2. Configure Nginx
```bash
sudo nano /etc/nginx/sites-available/rag-system
```
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Max upload size
    client_max_body_size 50M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
    }
}
```

### 3. Enable Site
```bash
# Create symlink
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### 4. SSL with Let's Encrypt
```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

## 📊 Monitoring

### 1. Application Monitoring

Use **Prometheus + Grafana**:
```python
# Add to requirements.txt
prometheus-client==0.17.1

# Add to app
from prometheus_client import Counter, Histogram, generate_latest

upload_counter = Counter('uploads_total', 'Total uploads')
query_counter = Counter('queries_total', 'Total queries')
query_duration = Histogram('query_duration_seconds', 'Query duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 2. Health Checks
```bash
# Add to cron for health monitoring
*/5 * * * * curl -f http://localhost:8000/health || systemctl restart rag-system
```

### 3. Log Monitoring
```bash
# Setup log rotation
sudo nano /etc/logrotate.d/rag-system
```
```
/var/log/rag-system/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        systemctl reload rag-system
    endscript
}
```

## 💾 Backup Strategy

### 1. Database Backups
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/rag-system"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup Redis
redis-cli SAVE
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis_$DATE.rdb"

# Backup ChromaDB
tar -czf "$BACKUP_DIR/chroma_$DATE.tar.gz" /opt/multimodal_rag/chroma_db

# Backup figures
tar -czf "$BACKUP_DIR/figures_$DATE.tar.gz" /opt/multimodal_rag/figures

# Remove old backups (keep 30 days)
find $BACKUP_DIR -name "*.rdb" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### 2. Schedule Backups
```bash
# Add to crontab
0 2 * * * /opt/scripts/backup.sh
```

## 📈 Performance Optimization

### 1. Increase Workers
```python
# run_server.py
uvicorn.run(
    "app.server.api:app",
    host=settings.API_HOST,
    port=settings.API_PORT,
    workers=4,  # CPU cores
    reload=False
)
```

### 2. Enable Caching
```python
# Add Redis caching for queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return model.encode(text)
```

### 3. Database Optimization
```bash
# Redis maxmemory policy
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

## 🔍 Troubleshooting

### Check Logs
```bash
# Application logs
sudo journalctl -u rag-system -n 100

# Nginx logs
sudo tail -f /var/log/nginx/error.log
sudo tail -f /var/log/nginx/access.log
```

### Monitor Resources
```bash
# CPU/Memory usage
htop

# Disk space
df -h

# Redis memory
redis-cli INFO memory
```

### Common Issues

**High Memory Usage:**
```bash
# Restart services
sudo systemctl restart rag-system
sudo systemctl restart redis
```

**Slow Queries:**
```bash
# Check ChromaDB size
du -sh chroma_db/

# Optimize if needed
# Consider sharding or archiving old data
```

## 📧 Support

For production support:
- Email: support@yourdomain.com
- Slack: #rag-support
- On-call: PagerDuty

---

Last updated: 2024
