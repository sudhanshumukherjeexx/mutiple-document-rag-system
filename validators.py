"""
Input validation and security module for RAG system.
Provides input sanitization and validation functions.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Validates and sanitizes user inputs for security."""
    
    def __init__(
        self,
        max_query_length: int = 1000,
        enable_sanitization: bool = True
    ):
        """
        Initialize input validator.
        
        Args:
            max_query_length: Maximum allowed query length
            enable_sanitization: Whether to sanitize inputs
        """
        self.max_query_length = max_query_length
        self.enable_sanitization = enable_sanitization
        
        # Patterns that might indicate prompt injection attempts
        self.suspicious_patterns = [
            r"ignore\s+previous\s+instructions",
            r"ignore\s+all\s+previous",
            r"disregard\s+previous",
            r"system\s*:\s*you\s+are",
            r"<\s*script\s*>",
            r"javascript\s*:",
        ]
    
    def validate_query(self, query: str) -> str:
        """
        Validate and sanitize a user query.
        
        Args:
            query: User's query string
            
        Returns:
            Validated and sanitized query
            
        Raises:
            ValidationError: If query is invalid
        """
        if not query:
            raise ValidationError("Query cannot be empty")
        
        if not isinstance(query, str):
            raise ValidationError("Query must be a string")
        
        # Check length
        if len(query) > self.max_query_length:
            raise ValidationError(
                f"Query exceeds maximum length of {self.max_query_length} characters"
            )
        
        # Check for suspicious patterns
        if self.enable_sanitization:
            for pattern in self.suspicious_patterns:
                if re.search(pattern, query.lower()):
                    logger.warning(f"Suspicious pattern detected in query: {pattern}")
                    # We don't reject it, but log it for monitoring
        
        # Sanitize query
        if self.enable_sanitization:
            query = self._sanitize_input(query)
        
        return query.strip()
    
    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize input text.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text
    
    def validate_file_path(self, file_path: str) -> str:
        """
        Validate a file path for security.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Validated file path
            
        Raises:
            ValidationError: If file path is invalid
        """
        if not file_path:
            raise ValidationError("File path cannot be empty")
        
        # Check for path traversal attempts
        if '..' in file_path or file_path.startswith('/'):
            raise ValidationError("Invalid file path: path traversal detected")
        
        return file_path
    
    def validate_config_value(self, value: any, value_type: type, min_val: Optional[float] = None, max_val: Optional[float] = None) -> any:
        """
        Validate a configuration value.
        
        Args:
            value: Value to validate
            value_type: Expected type
            min_val: Minimum value (for numeric types)
            max_val: Maximum value (for numeric types)
            
        Returns:
            Validated value
            
        Raises:
            ValidationError: If value is invalid
        """
        if not isinstance(value, value_type):
            raise ValidationError(f"Expected {value_type}, got {type(value)}")
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"Value {value} is below minimum {min_val}")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"Value {value} exceeds maximum {max_val}")
        
        return value


def validate_query(query: str, max_length: int = 1000) -> str:
    """
    Convenience function to validate a query.
    
    Args:
        query: Query to validate
        max_length: Maximum query length
        
    Returns:
        Validated query
    """
    validator = InputValidator(max_query_length=max_length)
    return validator.validate_query(query)
