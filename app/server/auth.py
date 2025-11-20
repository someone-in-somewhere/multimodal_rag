"""
Authentication module
Handles API key verification and JWT tokens
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import HTTPException, Security, status
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
# API Key Verification
# ============================================================================

async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> bool:
    """
    Verify API key from either X-API-Key header or Bearer token
    
    Supports two authentication methods:
    1. X-API-Key header: X-API-Key: your-api-key
    2. Bearer token: Authorization: Bearer your-api-key
    
    Args:
        api_key: API key from X-API-Key header
        credentials: Bearer token credentials
    
    Returns:
        True if authenticated
    
    Raises:
        HTTPException: If authentication fails
    """
    # Method 1: Check X-API-Key header
    if api_key:
        if api_key == settings.API_KEY:
            logger.debug("✅ Authenticated via X-API-Key header")
            return True
        else:
            logger.warning(f"⚠️  Invalid API key attempt: {api_key[:10]}...")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"}
            )
    
    # Method 2: Check Bearer token
    if credentials:
        token = credentials.credentials
        
        # Simple token comparison (can be enhanced with JWT)
        if token == settings.API_KEY:
            logger.debug("✅ Authenticated via Bearer token")
            return True
        
        # Try to decode as JWT (if it's a JWT token)
        try:
            payload = decode_jwt_token(token)
            if payload:
                logger.debug(f"✅ Authenticated via JWT token (user: {payload.get('sub')})")
                return True
        except JWTError:
            pass
        
        logger.warning(f"⚠️  Invalid bearer token attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # No authentication provided
    logger.warning("⚠️  No authentication credentials provided")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide API key via X-API-Key header or Bearer token.",
        headers={"WWW-Authenticate": "ApiKey, Bearer"}
    )


# ============================================================================
# JWT Token Functions (Optional - for future use)
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
    """
    Verify a password against a hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
    
    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


# ============================================================================
# Example Usage for User Authentication (Future Enhancement)
# ============================================================================

"""
# Example: Create a JWT token for a user
token = create_jwt_token(
    data={"sub": "user@example.com", "role": "admin"},
    expires_delta=timedelta(hours=24)
)

# Example: Verify token in endpoint
async def protected_endpoint(token: str = Depends(bearer_scheme)):
    payload = decode_jwt_token(token.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_email = payload.get("sub")
    # ... rest of the endpoint logic
"""
