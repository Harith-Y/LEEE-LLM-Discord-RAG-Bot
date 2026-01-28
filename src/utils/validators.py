"""
Input validation and sanitization utilities
"""
import re
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class QueryValidator:
    """Validator for user queries"""
    
    def __init__(self, max_length: int = 500, min_length: int = 3):
        """
        Initialize query validator
        
        Args:
            max_length: Maximum allowed query length
            min_length: Minimum allowed query length
        """
        self.max_length = max_length
        self.min_length = min_length
        
        # Patterns to detect spam/abuse
        self.spam_patterns = [
            r'(.)\1{10,}',  # Repeated characters (10+ times)
            r'^[^a-zA-Z0-9\s]+$',  # Only special characters
        ]
    
    def validate(self, query: str) -> Tuple[bool, str]:
        """
        Validate user query
        
        Args:
            query: User's input query
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if query is valid, False otherwise
            - error_message: Empty string if valid, error description if invalid
        """
        # Check if empty or only whitespace
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        # Strip whitespace for further checks
        query = query.strip()
        
        # Check length constraints
        if len(query) < self.min_length:
            return False, f"Query too short (minimum {self.min_length} characters)"
        
        if len(query) > self.max_length:
            return False, f"Query too long (maximum {self.max_length} characters)"
        
        # Check for spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, query):
                return False, "Query contains invalid patterns"
        
        # Check if query contains at least one alphanumeric character
        if not re.search(r'[a-zA-Z0-9]', query):
            return False, "Query must contain at least one letter or number"
        
        return True, ""
    
    def sanitize(self, query: str) -> str:
        """
        Sanitize query by removing/replacing problematic characters
        
        Args:
            query: Raw user query
            
        Returns:
            Sanitized query
        """
        # Strip leading/trailing whitespace
        query = query.strip()
        
        # Replace multiple spaces with single space
        query = re.sub(r'\s+', ' ', query)
        
        # Remove null bytes and other control characters
        query = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', query)
        
        return query
    
    def is_appropriate(self, query: str) -> Tuple[bool, str]:
        """
        Check if query is appropriate (not offensive/spam)
        
        Args:
            query: User query
            
        Returns:
            Tuple of (is_appropriate, reason)
        """
        query_lower = query.lower()
        
        # Basic profanity check (expand as needed)
        profanity_indicators = [
            # Add specific words if needed, or use external library
        ]
        
        for word in profanity_indicators:
            if word in query_lower:
                return False, "Query contains inappropriate language"
        
        # Check for URL spam
        url_pattern = r'https?://|www\.'
        url_matches = re.findall(url_pattern, query, re.IGNORECASE)
        if len(url_matches) > 2:  # Allow 1-2 URLs, flag excessive ones
            return False, "Query contains excessive URLs"
        
        return True, ""


class InputValidator:
    """General input validator for various input types"""
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """
        Validate Discord user ID format
        
        Args:
            user_id: Discord user ID
            
        Returns:
            True if valid format
        """
        if not user_id:
            return False
        
        # Discord IDs are numeric strings of 17-19 digits
        return bool(re.match(r'^\d{17,19}$', str(user_id)))
    
    @staticmethod
    def validate_channel_id(channel_id: str) -> bool:
        """
        Validate Discord channel ID format
        
        Args:
            channel_id: Discord channel ID
            
        Returns:
            True if valid format
        """
        if not channel_id:
            return False
        
        # Discord channel IDs are numeric strings
        return bool(re.match(r'^\d{17,19}$', str(channel_id)))
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal attacks
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove directory separators and null bytes
        filename = re.sub(r'[/\\:\x00]', '', filename)
        
        # Remove leading dots to prevent hidden files
        filename = filename.lstrip('.')
        
        # Limit length
        max_length = 255
        if len(filename) > max_length:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            filename = name[:max_length - len(ext) - 1] + '.' + ext if ext else name[:max_length]
        
        return filename


def create_query_validator(max_length: int = 500, min_length: int = 3) -> QueryValidator:
    """
    Factory function to create QueryValidator with custom settings
    
    Args:
        max_length: Maximum query length
        min_length: Minimum query length
        
    Returns:
        Configured QueryValidator instance
    """
    return QueryValidator(max_length=max_length, min_length=min_length)
