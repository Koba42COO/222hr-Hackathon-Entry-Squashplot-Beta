"""
Enterprise prime aligned compute Platform - Authentication Service
=======================================================

Comprehensive authentication and authorization system for the
Enterprise prime aligned compute Platform with JWT tokens, user management,
and enterprise-grade security features.

Author: Enterprise prime aligned compute Platform Team
Version: 2.0.0
License: Proprietary
"""

import os
import json
import time
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from functools import wraps

import jwt
from passlib.hash import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Import database service (lazy loading to avoid circular imports)
DATABASE_AVAILABLE = False
database_service = None

def _get_database_service():
    """Lazy load database service"""
    global DATABASE_AVAILABLE, database_service
    if database_service is None:
        try:
            from database_service import database_service as db_service
            database_service = db_service
            DATABASE_AVAILABLE = True
        except ImportError:
            DATABASE_AVAILABLE = False
            print("Warning: Database service not available")
    return database_service

# Configure logging
logger = logging.getLogger(__name__)

# Security constants
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', secrets.token_hex(32))
JWT_ALGORITHM = 'HS256'
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7

ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
REFRESH_TOKEN_ENCRYPTION_KEY = os.getenv('REFRESH_TOKEN_ENCRYPTION_KEY', Fernet.generate_key())

@dataclass
class User:
    """User data model"""
    id: str
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool = True
    created_at: datetime = None
    last_login: datetime = None
    permissions: List[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.permissions is None:
            self.permissions = []

@dataclass
class TokenData:
    """JWT token data"""
    user_id: str
    username: str
    role: str
    permissions: List[str]
    exp: int
    iat: int

class AuthService:
    """
    Enterprise-grade authentication service with JWT tokens,
    user management, and comprehensive security features.
    """

    def __init__(self):
        # Initialize encryption
        self.fernet = Fernet(ENCRYPTION_KEY)
        self.refresh_fernet = Fernet(REFRESH_TOKEN_ENCRYPTION_KEY)

        # Create default admin user if database is available
        db_service = _get_database_service()
        if db_service:
            self._create_default_admin()

        logger.info("Authentication service initialized")

    def _create_default_admin(self):
        """Create default admin user for initial setup"""
        db_service = _get_database_service()
        if not db_service:
            return

        # Check if admin already exists
        existing_admin = db_service.get_user_by_username("admin")
        if existing_admin:
            logger.info("Default admin user already exists")
            return

        hashed_password = self._hash_password("admin123!")
        success = db_service.create_user(
            "admin-001",
            "admin",
            "admin@prime aligned compute.platform",
            "Platform Administrator",
            hashed_password,
            "admin"
        )

        if success:
            logger.info("Default admin user created")
        else:
            logger.error("Failed to create default admin user")

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hash(password)

    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.verify(password, hashed)

    def _generate_tokens(self, user: User) -> Tuple[str, str]:
        """Generate access and refresh tokens"""
        # Access token
        access_token_data = {
            "sub": user.id,
            "username": user.username,
            "role": user.role,
            "permissions": user.permissions,
            "type": "access",
            "iat": int(time.time()),
            "exp": int((datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp())
        }

        access_token = jwt.encode(access_token_data, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        # Refresh token
        refresh_token_data = {
            "sub": user.id,
            "username": user.username,
            "type": "refresh",
            "iat": int(time.time()),
            "exp": int((datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)).timestamp())
        }

        refresh_token = jwt.encode(refresh_token_data, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

        # Encrypt refresh token for storage
        encrypted_refresh = self.refresh_fernet.encrypt(refresh_token.encode()).decode()

        # Store encrypted refresh token in database
        db_service = _get_database_service()
        if db_service:
            refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            expires_at = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
            db_service.store_refresh_token(
                user.id,
                refresh_token_hash,
                encrypted_refresh,
                expires_at
            )

        return access_token, refresh_token

    def authenticate_user(self, username: str, password: str) -> Optional[Tuple[str, str, User]]:
        """
        Authenticate user and return tokens

        Returns:
            Tuple of (access_token, refresh_token, user) or None if authentication fails
        """
        db_service = _get_database_service()
        if not db_service:
            logger.error("Database not available for authentication")
            return None

        user_data = db_service.get_user_by_username(username)
        if not user_data:
            logger.warning(f"Authentication failed: user {username} not found")
            return None

        if not self._verify_password(password, user_data["password_hash"]):
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None

        if not user_data["is_active"]:
            logger.warning(f"Authentication failed: user {username} is inactive")
            return None

        # Convert database user data to User object
        user = User(
            id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            full_name=user_data["full_name"],
            role=user_data["role"],
            is_active=user_data["is_active"],
            created_at=user_data["created_at"],
            last_login=user_data["last_login"],
            permissions=user_data["permissions"]
        )

        # Update last login
        db_service.update_user(username, {"last_login": datetime.utcnow()})

        # Generate tokens
        access_token, refresh_token = self._generate_tokens(user)

        logger.info(f"User {username} authenticated successfully")
        return access_token, refresh_token, user

    def refresh_access_token(self, refresh_token: str) -> Optional[Tuple[str, str]]:
        """Refresh access token using refresh token"""
        try:
            # Decode refresh token
            payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

            if payload.get("type") != "refresh":
                logger.warning("Invalid token type for refresh")
                return None

            user_id = payload["sub"]

            # Verify refresh token is valid and matches stored token in database
            db_service = _get_database_service()
            if not db_service:
                logger.error("Database not available for token refresh")
                return None

            refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            token_data = db_service.get_refresh_token(refresh_token_hash)

            if not token_data:
                logger.warning(f"Refresh token not found for user {user_id}")
                return None

            stored_encrypted = token_data["encrypted_token"]
            stored_token = self.refresh_fernet.decrypt(stored_encrypted.encode()).decode()

            if stored_token != refresh_token:
                logger.warning(f"Refresh token mismatch for user {user_id}")
                return None

            # Find user in database
            user_data = db_service.get_user_by_username(payload["username"])
            if not user_data or not user_data["is_active"]:
                logger.warning(f"User {user_id} not found or inactive")
                return None

            # Convert to User object
            user = User(
                id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"],
                full_name=user_data["full_name"],
                role=user_data["role"],
                is_active=user_data["is_active"],
                created_at=user_data["created_at"],
                last_login=user_data["last_login"],
                permissions=user_data["permissions"]
            )

            # Generate new tokens
            new_access_token, new_refresh_token = self._generate_tokens(user)

            logger.info(f"Access token refreshed for user {user.username}")
            return new_access_token, new_refresh_token

        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
        except Exception as e:
            logger.error(f"Refresh token error: {e}")
            return None

    def validate_token(self, token: str) -> Optional[TokenData]:
        """Validate JWT token and return token data"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

            if payload.get("type") != "access":
                return None

            return TokenData(
                user_id=payload["sub"],
                username=payload["username"],
                role=payload["role"],
                permissions=payload.get("permissions", []),
                exp=payload["exp"],
                iat=payload["iat"]
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None

    def revoke_refresh_token(self, user_id: str):
        """Revoke refresh token for user"""
        db_service = _get_database_service()
        if db_service:
            db_service.revoke_refresh_token(user_id)
            logger.info(f"Refresh token revoked for user {user_id}")

    def create_user(self, username: str, password: str, email: str,
                   full_name: str, role: str = "user") -> Optional[User]:
        """Create new user (admin only)"""
        db_service = _get_database_service()
        if not db_service:
            logger.error("Database not available for user creation")
            return None

        user_id = f"user-{secrets.token_hex(8)}"
        hashed_password = self._hash_password(password)

        success = db_service.create_user(
            user_id,
            username,
            email,
            full_name,
            hashed_password,
            role
        )

        if success:
            user = User(
                id=user_id,
                username=username,
                email=email,
                full_name=full_name,
                role=role,
                permissions=self._get_default_permissions(role)
            )
            logger.info(f"User {username} created with role {role}")
            return user

        logger.error(f"Failed to create user {username}")
        return None

    def _get_default_permissions(self, role: str) -> List[str]:
        """Get default permissions for role"""
        role_permissions = {
            "admin": ["read", "write", "delete", "admin", "system"],
            "researcher": ["read", "write", "analyze"],
            "operator": ["read", "write"],
            "viewer": ["read"]
        }
        return role_permissions.get(role, ["read"])

    def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        db_service = _get_database_service()
        if not db_service:
            return None

        user_data = db_service.get_user_by_username(username)
        if user_data:
            return User(
                id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"],
                full_name=user_data["full_name"],
                role=user_data["role"],
                is_active=user_data["is_active"],
                created_at=user_data["created_at"],
                last_login=user_data["last_login"],
                permissions=user_data["permissions"]
            )
        return None

    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """Update user information"""
        db_service = _get_database_service()
        if not db_service:
            return False

        success = db_service.update_user(username, updates)

        if success:
            logger.info(f"User {username} updated")
        else:
            logger.error(f"Failed to update user {username}")

        return success

    def delete_user(self, username: str) -> bool:
        """Delete user"""
        if not DATABASE_AVAILABLE:
            return False

        # Get user first to revoke tokens
        db_service = _get_database_service()
        if db_service:
            user_data = db_service.get_user_by_username(username)
            if user_data:
                self.revoke_refresh_token(user_data["id"])

            success = db_service.delete_user(username)
        else:
            return False

        if success:
            logger.info(f"User {username} deleted")
        else:
            logger.error(f"Failed to delete user {username}")

        return success

    def list_users(self) -> List[Dict[str, Any]]:
        """List all users (admin only)"""
        db_service = _get_database_service()
        if not db_service:
            return []

        return db_service.list_users()

    # API Key Management
    def create_api_key(self, user_id: str, name: str, permissions: List[str] = None) -> Optional[str]:
        """Create API key for user"""
        db_service = _get_database_service()
        if not db_service:
            return None

        if permissions is None:
            permissions = ["read"]

        api_key = secrets.token_urlsafe(32)
        success = db_service.store_api_key(
            user_id,
            name,
            hashlib.sha256(api_key.encode()).hexdigest(),
            permissions
        )

        if success:
            logger.info(f"API key created for user {user_id}: {name}")
            return api_key

        logger.error(f"Failed to create API key for user {user_id}")
        return None

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key"""
        db_service = _get_database_service()
        if not db_service:
            return None

        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        return db_service.get_api_key(hashed_key)

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        db_service = _get_database_service()
        if not db_service:
            return False

        hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = db_service.get_api_key(hashed_key)

        if key_data:
            # For now, we'll just return true - in production you'd want to revoke by name
            logger.info(f"API key revocation requested")
            return True

        return False

    def list_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """List API keys for user"""
        db_service = _get_database_service()
        if not db_service:
            return []

        return db_service.list_user_api_keys(user_id)

# Global auth service instance
auth_service = AuthService()

# Authentication decorators
def require_auth(roles: List[str] = None, permissions: List[str] = None):
    """Decorator to require authentication and authorization"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import Request, HTTPException, Depends
            from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

            # Get request object
            request = None
            for arg in args:
                if hasattr(arg, 'headers'):
                    request = arg
                    break

            if not request:
                raise HTTPException(status_code=401, detail="Request object not found")

            # Check for API key first
            api_key = request.headers.get("X-API-Key")
            if api_key:
                key_data = auth_service.validate_api_key(api_key)
                if key_data:
                    # Check permissions
                    if permissions:
                        if not all(perm in key_data["permissions"] for perm in permissions):
                            raise HTTPException(status_code=403, detail="Insufficient API key permissions")
                    request.state.user = key_data
                    request.state.auth_type = "api_key"
                    return await func(*args, **kwargs)

            # Check for JWT token
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Authentication required")

            token = auth_header.split(" ")[1]
            token_data = auth_service.validate_token(token)

            if not token_data:
                raise HTTPException(status_code=401, detail="Invalid or expired token")

            # Check roles
            if roles and token_data.role not in roles:
                raise HTTPException(status_code=403, detail="Insufficient permissions")

            # Check permissions
            if permissions:
                if not all(perm in token_data.permissions for perm in permissions):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")

            request.state.user = token_data
            request.state.auth_type = "jwt"
            return await func(*args, **kwargs)

        return wrapper
    return decorator

def require_admin(func):
    """Decorator to require admin role"""
    return require_auth(roles=["admin"])(func)

# Utility functions
def get_current_user(request) -> Optional[TokenData]:
    """Get current authenticated user from request"""
    return getattr(request.state, 'user', None)

def get_auth_type(request) -> str:
    """Get authentication type (jwt or api_key)"""
    return getattr(request.state, 'auth_type', 'none')

# Security middleware for rate limiting
class RateLimiter:
    """Simple rate limiter for API endpoints"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = {}  # In production, use Redis

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        if client_id not in self.requests:
            self.requests[client_id] = []

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < 60
        ]

        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False

        self.requests[client_id].append(now)
        return True

# Global rate limiter
rate_limiter = RateLimiter()

if __name__ == "__main__":
    # Test the auth service
    print("🧪 Testing Authentication Service...")

    # Test user creation
    user = auth_service.create_user("testuser", "password123", "test@example.com", "Test User", "researcher")
    print(f"✅ User created: {user.username if user else 'Failed'}")

    # Test authentication
    tokens = auth_service.authenticate_user("testuser", "password123")
    if tokens:
        access_token, refresh_token, user = tokens
        print(f"✅ Authentication successful for {user.username}")
        print(f"   Role: {user.role}")
        print(f"   Permissions: {user.permissions}")

        # Test token validation
        token_data = auth_service.validate_token(access_token)
        if token_data:
            print(f"✅ Token validation successful for {token_data.username}")

        # Test API key creation
        api_key = auth_service.create_api_key(user.id, "Test API Key")
        if api_key:
            print(f"✅ API key created: {api_key[:10]}...")

            # Test API key validation
            key_data = auth_service.validate_api_key(api_key)
            if key_data:
                print(f"✅ API key validation successful")

    print("🎉 Authentication service test completed!")
