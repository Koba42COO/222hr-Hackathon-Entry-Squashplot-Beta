
import re
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

class DataValidator:
    """Comprehensive data validation system"""
    
    def __init__(self):
        self.validation_rules = {}
        self.error_messages = {}
    
    def validate_string(self, value: Any, min_length: int = 0, max_length: int = None, 
                       pattern: str = None, required: bool = True) -> Dict[str, Any]:
        """Validate string data"""
        errors = []
        
        if not value and required:
            errors.append("Field is required")
            return {"valid": False, "errors": errors}
        
        if value is None:
            return {"valid": True, "errors": []}
        
        if not isinstance(value, str):
            errors.append("Must be a string")
            return {"valid": False, "errors": errors}
        
        if len(value) < min_length:
            errors.append(f"Must be at least {min_length} characters")
        
        if max_length and len(value) > max_length:
            errors.append(f"Must be no more than {max_length} characters")
        
        if pattern and not re.match(pattern, value):
            errors.append("Invalid format")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_email(self, email: str) -> Dict[str, Any]:
        """Validate email address"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return self.validate_string(email, pattern=pattern, required=True)
    
    def validate_password(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        errors = []
        
        if len(password) < 8:
            errors.append("Password must be at least 8 characters")
        
        if not re.search(r'[A-Z]', password):
            errors.append("Password must contain uppercase letter")
        
        if not re.search(r'[a-z]', password):
            errors.append("Password must contain lowercase letter")
        
        if not re.search(r'\d', password):
            errors.append("Password must contain number")
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain special character")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_number(self, value: Any, min_value: float = None, max_value: float = None, 
                       integer_only: bool = False) -> Dict[str, Any]:
        """Validate numeric data"""
        errors = []
        
        try:
            if integer_only:
                num_value = int(value)
            else:
                num_value = float(value)
        except (ValueError, TypeError):
            errors.append("Must be a valid number")
            return {"valid": False, "errors": errors}
        
        if min_value is not None and num_value < min_value:
            errors.append(f"Must be at least {min_value}")
        
        if max_value is not None and num_value > max_value:
            errors.append(f"Must be no more than {max_value}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_json(self, value: Any) -> Dict[str, Any]:
        """Validate JSON data"""
        errors = []
        
        if isinstance(value, str):
            try:
                json.loads(value)
            except json.JSONDecodeError:
                errors.append("Invalid JSON format")
        elif not isinstance(value, (dict, list)):
            errors.append("Must be JSON object or array")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def validate_file_upload(self, filename: str, allowed_extensions: List[str] = None,
                           max_size: int = None) -> Dict[str, Any]:
        """Validate file upload"""
        errors = []
        
        if not filename:
            errors.append("Filename is required")
            return {"valid": False, "errors": errors}
        
        if allowed_extensions:
            file_ext = filename.split('.')[-1].lower()
            if file_ext not in allowed_extensions:
                errors.append(f"File type not allowed. Allowed: {', '.join(allowed_extensions)}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def sanitize_input(self, value: str) -> str:
        """Sanitize user input"""
        if not isinstance(value, str):
            return str(value)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"']', '', value)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized
    
    def validate_complete_form(self, data: Dict[str, Any], rules: Dict[str, Dict]) -> Dict[str, Any]:
        """Validate complete form data"""
        results = {}
        all_valid = True
        
        for field, rule in rules.items():
            value = data.get(field)
            validation_result = self.validate_string(value, **rule)
            results[field] = validation_result
            
            if not validation_result["valid"]:
                all_valid = False
        
        return {
            "valid": all_valid,
            "results": results,
            "errors": [error for result in results.values() for error in result.get("errors", [])]
        }
