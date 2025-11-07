"""
Configuration management module for RAG system.
Loads and validates configuration from YAML file.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the RAG system."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        """Singleton pattern to ensure only one config instance."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def load(self, config_path: str = "config.yaml") -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            self._validate_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate required configuration keys."""
        required_sections = ['api', 'models', 'document_processing', 'vector_store', 'rag']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        logger.debug("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., 'models.summarize.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config = Config()
            >>> config.get('models.summarize.name')
            'gpt-4o-mini'
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., 'models.summarize.name')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key_path} = {value}")
    
    def get_openai_api_key(self) -> str:
        """
        Get OpenAI API key from environment variable.
        
        Returns:
            API key string
            
        Raises:
            ValueError: If API key not found
        """
        env_var = self.get('api.openai_api_key_env', 'OPENAI_API_KEY')
        api_key = os.environ.get(env_var)
        
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found in environment variable '{env_var}'. "
                "Please set it before running the application."
            )
        
        return api_key
    
    def create_directories(self) -> None:
        """Create necessary directories for logs, cache, etc."""
        directories = [
            'logs',
            self.get('cache.cache_dir', '.cache'),
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return self._config.copy()


# Global config instance
config = Config()


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    config.load(config_path)
    config.create_directories()
    return config
