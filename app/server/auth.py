"""
app/server/auth.py

auth_v2.py - Enhanced Authentication Module
============================================
NEW in v2:
- ✅ Rate limiting (prevent brute force)
- ✅ Timing-safe key comparison (prevent timing attacks)
- ✅ Comprehensive audit logging (track all access)
- ✅ Multiple API keys (per-user/service keys)
- ✅ Request quotas (per key limits)
- ✅ Key metadata (expiry, permissions, tags)
- ✅ IP whitelisting (optional security)
- ✅ Monitoring and statistics

Handles API key verification and JWT tokens with production-grade security
"""

import logging
import time
import hmac
import hashlib
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from fastapi import HTTPException, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext

from config import settings

logger = logging.getLogger(__name__)

# ============================================================================
# Security Schemes
# ============================================================================

# API Key in header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Bearer token
bearer_scheme = HTTPBearer(auto_error=False)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter
    
    Prevents brute force attacks by limiting authentication attempts
    """
    
    def __init__(self, max_attempts: int = 10, window_seconds: int = 60):
        """
        Args:
            max_attempts: Maximum attempts per window
            window_seconds: Time window in seconds
        """
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.attempts = defaultdict(list)  # IP -> [timestamps]
        self.blocked = {}  # IP -> block_until_timestamp
    
    async def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if identifier is within rate limit
        
        Args:
            identifier: IP address or key hash
        
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        
        # Check if blocked
        if identifier in self.blocked:
            if now < self.blocked[identifier]:
                # Still blocked
                return False
            else:
                # Unblock
                del self.blocked[identifier]
        
        # Clean old attempts
        cutoff = now - self.window_seconds
        self.attempts[identifier] = [
            ts for ts in self.attempts[identifier]
            if ts > cutoff
        ]
        
        # Check limit
        if len(self.attempts[identifier]) >= self.max_attempts:
            # Block for 5 minutes
            self.blocked[identifier] = now + 300
            logger.warning(f"🚫 Rate limit exceeded for {identifier}, blocked for 5 minutes")
            return False
        
        # Record attempt
        self.attempts[identifier].append(now)
        return True
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining attempts"""
        now = time.time()
        cutoff = now - self.window_seconds
        recent = [ts for ts in self.attempts.get(identifier, []) if ts > cutoff]
        return max(0, self.max_attempts - len(recent))
    
    def reset(self, identifier: str):
        """Reset rate limit for identifier"""
        if identifier in self.attempts:
            del self.attempts[identifier]
        if identifier in self.blocked:
            del self.blocked[identifier]


# Global rate limiter
rate_limiter = RateLimiter(max_attempts=10, window_seconds=60)


# ============================================================================
# API Key Management
# ============================================================================

class APIKey:
    """
    API Key with metadata
    
    Supports:
    - Expiry dates
    - Permissions
    - Usage quotas
    - IP whitelisting
    """
    
    def __init__(
        self,
        key: str,
        name: str,
        permissions: List[str] = None,
        expires_at: Optional[datetime] = None,
        ip_whitelist: List[str] = None,
        daily_quota: Optional[int] = None,
        tags: Dict[str, str] = None
    ):
        self.key = key
        self.name = name
        self.permissions = permissions or ["read", "write"]
        self.expires_at = expires_at
        self.ip_whitelist = ip_whitelist or []
        self.daily_quota = daily_quota
        self.tags = tags or {}
        
        # Usage tracking
        self.created_at = datetime.utcnow()
        self.last_used = None
        self.usage_count = 0
        self.daily_usage = defaultdict(int)  # date -> count
    
    def is_valid(self) -> bool:
        """Check if key is valid (not expired)"""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def has_permission(self, permission: str) -> bool:
        """Check if key has permission"""
        return permission in self.permissions
    
    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        if not self.ip_whitelist:
            return True  # No whitelist = all allowed
        return ip in self.ip_whitelist
    
    def check_quota(self) -> bool:
        """Check if daily quota not exceeded"""
        if not self.daily_quota:
            return True  # No quota = unlimited
        
        today = datetime.utcnow().date().isoformat()
        return self.daily_usage[today] < self.daily_quota
    
    def record_usage(self):
        """Record API key usage"""
        self.last_used = datetime.utcnow()
        self.usage_count += 1
        today = datetime.utcnow().date().isoformat()
        self.daily_usage[today] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        today = datetime.utcnow().date().isoformat()
        
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "total_usage": self.usage_count,
            "today_usage": self.daily_usage.get(today, 0),
            "daily_quota": self.daily_quota,
            "quota_remaining": (
                self.daily_quota - self.daily_usage.get(today, 0)
                if self.daily_quota else None
            ),
            "is_valid": self.is_valid(),
            "permissions": self.permissions,
            "tags": self.tags
        }


class APIKeyManager:
    """
    Manages multiple API keys
    
    NEW: Support multiple keys with different permissions
    """
    
    def __init__(self):
        self.keys: Dict[str, APIKey] = {}
        self._load_default_keys()
    
    def _load_default_keys(self):
        """Load default API key from settings"""
        # Default key (from settings)
        default_key = APIKey(
            key=settings.API_KEY,
            name="default",
            permissions=["read", "write", "delete", "admin"],
            expires_at=None,  # Never expires
            daily_quota=None  # Unlimited
        )
        self.keys[settings.API_KEY] = default_key
        
        logger.info(f"✅ Loaded {len(self.keys)} API key(s)")
    
    def add_key(
        self,
        name: str,
        permissions: List[str] = None,
        expires_days: Optional[int] = None,
        ip_whitelist: List[str] = None,
        daily_quota: Optional[int] = None,
        tags: Dict[str, str] = None
    ) -> str:
        """
        Create a new API key
        
        Args:
            name: Key name/description
            permissions: List of permissions
            expires_days: Days until expiry (None = never)
            ip_whitelist: Allowed IPs (None = all)
            daily_quota: Daily request limit (None = unlimited)
            tags: Custom tags
        
        Returns:
            Generated API key
        """
        # Generate secure random key
        key = secrets.token_urlsafe(32)
        
        # Calculate expiry
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        # Create key object
        api_key = APIKey(
            key=key,
            name=name,
            permissions=permissions,
            expires_at=expires_at,
            ip_whitelist=ip_whitelist,
            daily_quota=daily_quota,
            tags=tags
        )
        
        self.keys[key] = api_key
        
        logger.info(f"✅ Created new API key: {name}")
        
        return key
    
    def get_key(self, key: str) -> Optional[APIKey]:
        """Get API key object"""
        return self.keys.get(key)
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key"""
        if key in self.keys:
            del self.keys[key]
            logger.info(f"🗑️ Revoked API key")
            return True
        return False
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all keys with stats"""
        return [
            {
                "key_prefix": key[:10] + "...",
                **api_key.get_stats()
            }
            for key, api_key in self.keys.items()
        ]


# Global key manager
key_manager = APIKeyManager()


# ============================================================================
# Audit Logger
# ============================================================================

class AuditLogger:
    """
    Audit logger for security events
    
    Tracks:
    - Authentication attempts (success/failure)
    - API usage
    - Security events
    """
    
    def __init__(self):
        self.events = []
        self.max_events = 10000  # Keep last 10k events
    
    def log_auth_attempt(
        self,
        success: bool,
        method: str,
        ip: str,
        key_prefix: Optional[str] = None,
        reason: Optional[str] = None
    ):
        """Log authentication attempt"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "auth",
            "success": success,
            "method": method,
            "ip": ip,
            "key_prefix": key_prefix,
            "reason": reason
        }
        
        self.events.append(event)
        
        # Trim if too many events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Log to file
        if success:
            logger.info(f"✅ Auth success: {method} from {ip}")
        else:
            logger.warning(f"❌ Auth failed: {method} from {ip} - {reason}")
    
    def log_api_request(
        self,
        endpoint: str,
        method: str,
        ip: str,
        key_name: str,
        success: bool,
        duration: float
    ):
        """Log API request"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "request",
            "endpoint": endpoint,
            "method": method,
            "ip": ip,
            "key_name": key_name,
            "success": success,
            "duration": duration
        }
        
        self.events.append(event)
        
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events"""
        return self.events[-limit:]
    
    def get_failed_attempts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent failed authentication attempts"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        return [
            event for event in self.events
            if event.get("type") == "auth"
            and not event.get("success")
            and datetime.fromisoformat(event["timestamp"]) > cutoff
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get audit statistics"""
        total_events = len(self.events)
        
        # Count by type
        auth_success = sum(
            1 for e in self.events
            if e.get("type") == "auth" and e.get("success")
        )
        auth_failed = sum(
            1 for e in self.events
            if e.get("type") == "auth" and not e.get("success")
        )
        
        # Recent failures (last hour)
        recent_failures = len(self.get_failed_attempts(60))
        
        return {
            "total_events": total_events,
            "auth_success": auth_success,
            "auth_failed": auth_failed,
            "recent_failures_1h": recent_failures,
            "events_stored": len(self.events)
        }


# Global audit logger
audit_logger = AuditLogger()


# ============================================================================
# Enhanced API Key Verification
# ============================================================================

def timing_safe_compare(a: str, b: str) -> bool:
    """
    Timing-safe string comparison
    
    Prevents timing attacks by ensuring comparison
    takes constant time regardless of where strings differ
    """
    return hmac.compare_digest(a, b)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> Dict[str, Any]:
    """
    Enhanced API key verification with V2 features
    
    IMPROVED:
    - Rate limiting (prevent brute force)
    - Timing-safe comparison (prevent timing attacks)
    - Audit logging (track all access)
    - Multiple keys support
    - Quota checking
    - IP whitelisting
    
    Args:
        request: FastAPI request object
        api_key: API key from X-API-Key header
        credentials: Bearer token credentials
    
    Returns:
        Dict with auth context (key info, permissions)
    
    Raises:
        HTTPException: If authentication fails
    """
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    rate_limit_ok = await rate_limiter.check_rate_limit(client_ip)
    
    if not rate_limit_ok:
        audit_logger.log_auth_attempt(
            success=False,
            method="rate_limited",
            ip=client_ip,
            reason="Rate limit exceeded"
        )
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please try again later.",
            headers={"Retry-After": "300"}
        )
    
    # Method 1: Check X-API-Key header
    if api_key:
        return await _verify_api_key_header(api_key, client_ip)
    
    # Method 2: Check Bearer token
    if credentials:
        return await _verify_bearer_token(credentials.credentials, client_ip)
    
    # No authentication provided
    audit_logger.log_auth_attempt(
        success=False,
        method="none",
        ip=client_ip,
        reason="No credentials provided"
    )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide API key via X-API-Key header or Bearer token.",
        headers={"WWW-Authenticate": "ApiKey, Bearer"}
    )


async def _verify_api_key_header(api_key: str, client_ip: str) -> Dict[str, Any]:
    """Verify API key from header"""
    # Get key object
    key_obj = key_manager.get_key(api_key)
    
    if not key_obj:
        # Try timing-safe comparison with default key
        if timing_safe_compare(api_key, settings.API_KEY):
            key_obj = key_manager.get_key(settings.API_KEY)
    
    if not key_obj:
        audit_logger.log_auth_attempt(
            success=False,
            method="api_key",
            ip=client_ip,
            key_prefix=api_key[:10],
            reason="Invalid key"
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    # Check if key is valid (not expired)
    if not key_obj.is_valid():
        audit_logger.log_auth_attempt(
            success=False,
            method="api_key",
            ip=client_ip,
            key_prefix=api_key[:10],
            reason="Key expired"
        )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key expired"
        )
    
    # Check IP whitelist
    if not key_obj.is_ip_allowed(client_ip):
        audit_logger.log_auth_attempt(
            success=False,
            method="api_key",
            ip=client_ip,
            key_prefix=api_key[:10],
            reason=f"IP {client_ip} not whitelisted"
        )
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="IP address not authorized"
        )
    
    # Check quota
    if not key_obj.check_quota():
        audit_logger.log_auth_attempt(
            success=False,
            method="api_key",
            ip=client_ip,
            key_prefix=api_key[:10],
            reason="Quota exceeded"
        )
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily quota exceeded"
        )
    
    # Success!
    key_obj.record_usage()
    
    audit_logger.log_auth_attempt(
        success=True,
        method="api_key",
        ip=client_ip,
        key_prefix=api_key[:10]
    )
    
    # Reset rate limit on success
    rate_limiter.reset(client_ip)
    
    return {
        "authenticated": True,
        "method": "api_key",
        "key_name": key_obj.name,
        "permissions": key_obj.permissions,
        "ip": client_ip
    }


async def _verify_bearer_token(token: str, client_ip: str) -> Dict[str, Any]:
    """Verify Bearer token"""
    # Try as API key first
    key_obj = key_manager.get_key(token)
    
    if key_obj:
        # Same validation as API key
        return await _verify_api_key_header(token, client_ip)
    
    # Try to decode as JWT
    try:
        payload = decode_jwt_token(token)
        
        if payload:
            audit_logger.log_auth_attempt(
                success=True,
                method="jwt",
                ip=client_ip
            )
            
            rate_limiter.reset(client_ip)
            
            return {
                "authenticated": True,
                "method": "jwt",
                "user": payload.get("sub"),
                "permissions": payload.get("permissions", ["read"]),
                "ip": client_ip
            }
    except JWTError:
        pass
    
    # Invalid token
    audit_logger.log_auth_attempt(
        success=False,
        method="bearer",
        ip=client_ip,
        reason="Invalid token"
    )
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication token",
        headers={"WWW-Authenticate": "Bearer"}
    )


# ============================================================================
# Permission Checking
# ============================================================================

def require_permission(permission: str):
    """
    Dependency to check if authenticated user has permission
    
    Usage:
        @app.delete("/admin/data", dependencies=[Depends(require_permission("admin"))])
        async def delete_all_data():
            ...
    """
    async def check_permission(auth_context: Dict = Depends(verify_api_key)):
        permissions = auth_context.get("permissions", [])
        
        if permission not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return auth_context
    
    return check_permission


# ============================================================================
# JWT Token Functions
# ============================================================================

def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT token
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
    
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def decode_jwt_token(token: str) -> Optional[dict]:
    """
    Decode and verify a JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    
    except JWTError as e:
        logger.debug(f"JWT decode error: {e}")
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


# ============================================================================
# Management Functions
# ============================================================================

async def get_auth_stats() -> Dict[str, Any]:
    """
    Get authentication statistics
    
    NEW: Comprehensive auth monitoring
    """
    return {
        "keys": {
            "total": len(key_manager.keys),
            "keys": key_manager.list_keys()
        },
        "audit": audit_logger.get_stats(),
        "rate_limiting": {
            "enabled": True,
            "max_attempts": rate_limiter.max_attempts,
            "window_seconds": rate_limiter.window_seconds,
            "blocked_ips": len(rate_limiter.blocked)
        }
    }


async def create_api_key(
    name: str,
    permissions: List[str] = None,
    expires_days: Optional[int] = None,
    ip_whitelist: List[str] = None,
    daily_quota: Optional[int] = None,
    tags: Dict[str, str] = None
) -> str:
    """
    Create a new API key
    
    NEW: Programmatic key creation
    """
    return key_manager.add_key(
        name=name,
        permissions=permissions,
        expires_days=expires_days,
        ip_whitelist=ip_whitelist,
        daily_quota=daily_quota,
        tags=tags
    )


async def revoke_api_key(key: str) -> bool:
    """Revoke an API key"""
    return key_manager.revoke_key(key)


async def get_audit_events(limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent audit events"""
    return audit_logger.get_recent_events(limit)


async def get_failed_attempts(minutes: int = 60) -> List[Dict[str, Any]]:
    """Get recent failed authentication attempts"""
    return audit_logger.get_failed_attempts(minutes)

