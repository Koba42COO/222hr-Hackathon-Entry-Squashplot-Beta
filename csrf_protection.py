
import secrets
import hashlib
import hmac
from typing import Dict, Optional

class CSRFProtection:
    """CSRF protection system"""
    
    def __init__(self):
        self.tokens: Dict[str, str] = {}
        self.secret_key = secrets.token_hex(32)
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        token = secrets.token_urlsafe(32)
        self.tokens[session_id] = token
        return token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token"""
        if session_id not in self.tokens:
            return False
        
        stored_token = self.tokens[session_id]
        return hmac.compare_digest(stored_token, token)
    
    def invalidate_token(self, session_id: str):
        """Invalidate CSRF token"""
        if session_id in self.tokens:
            del self.tokens[session_id]
    
    def get_token(self, session_id: str) -> Optional[str]:
        """Get current token for session"""
        return self.tokens.get(session_id)
    
    def rotate_token(self, session_id: str) -> str:
        """Rotate CSRF token"""
        return self.generate_token(session_id)
