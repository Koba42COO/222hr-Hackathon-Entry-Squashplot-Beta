
import uuid
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

class SessionManager:
    """Secure session management system"""
    
    def __init__(self, session_timeout: int = 3600):
        self.sessions: Dict[str, dict] = {}
        self.session_timeout = session_timeout
    
    def create_session(self, user_id: str, user_data: dict = None) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "data": user_data or {},
            "is_active": True
        }
        self.sessions[session_id] = session_data
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session is expired
        if self._is_session_expired(session):
            self.invalidate_session(session_id)
            return None
        
        # Update last activity
        session["last_activity"] = datetime.now().isoformat()
        return session
    
    def update_session(self, session_id: str, data: dict):
        """Update session data"""
        if session_id in self.sessions:
            self.sessions[session_id]["data"].update(data)
            self.sessions[session_id]["last_activity"] = datetime.now().isoformat()
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.sessions:
            self.sessions[session_id]["is_active"] = False
            del self.sessions[session_id]
    
    def _is_session_expired(self, session: dict) -> bool:
        """Check if session is expired"""
        last_activity = datetime.fromisoformat(session["last_activity"])
        return datetime.now() - last_activity > timedelta(seconds=self.session_timeout)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.invalidate_session(session_id)
    
    def get_active_sessions(self) -> Dict[str, dict]:
        """Get all active sessions"""
        self.cleanup_expired_sessions()
        return {sid: session for sid, session in self.sessions.items() if session.get("is_active", False)}
