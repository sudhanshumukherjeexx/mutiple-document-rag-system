"""
Document loading module for RAG system.
Handles loading of PDF, TXT, and Markdown files.
"""

import os
from pathlib import Path
from typing import List, Optional
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from ..validators import InputValidator

logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Custom exception for document loading errors."""
    pass


class DocumentLoader:
    """Loads documents from various file formats."""
    
    SUPPORTED_FORMATS = {
        '.pdf': 'PDF',
        '.txt': 'Text',
        '.md': 'Markdown'
    }
    
    def __init__(self, validate_paths: bool = True):
        """
        Initialize document loader.
        
        Args:
            validate_paths: Whether to validate file paths
        """
        self.validate_paths = validate_paths
        self.validator = InputValidator() if validate_paths else None
    
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
            
        Raises:
            DocumentLoaderError: If document cannot be loaded
        """
        # Validate path
        if self.validate_paths and self.validator:
            try:
                file_path = self.validator.validate_file_path(file_path)
            except Exception as e:
                raise DocumentLoaderError(f"Invalid file path: {e}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise DocumentLoaderError(f"File not found: {file_path}")
        
        # Get file extension
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext not in self.SUPPORTED_FORMATS:
            raise DocumentLoaderError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )
        
        # Load document based on format
        try:
            if file_ext == '.pdf':
                return self._load_pdf(file_path)
            elif file_ext in ['.txt', '.md']:
                return self._load_text(file_path)
        except Exception as e:
            raise DocumentLoaderError(f"Error loading {file_path}: {e}")
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects (one per page)
        """
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from {file_path}")
        return documents
    
    def _load_text(self, file_path: str) -> List[Document]:
        """
        Load a text or markdown document.
        
        Args:
            file_path: Path to text file
            
        Returns:
            List with single Document object
        """
        logger.info(f"Loading text file: {file_path}")
        loader = TextLoader(file_path)
        documents = loader.load()
        logger.info(f"Loaded {file_path}")
        return documents
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple documents.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Combined list of Document objects
        """
        if not file_paths:
            raise DocumentLoaderError("No file paths provided")
        
        all_documents = []
        errors = []
        
        for file_path in file_paths:
            try:
                docs = self.load_document(file_path)
                all_documents.extend(docs)
            except DocumentLoaderError as e:
                logger.error(f"Error loading {file_path}: {e}")
                errors.append((file_path, str(e)))
        
        if not all_documents and errors:
            raise DocumentLoaderError(
                f"Failed to load any documents. Errors: {errors}"
            )
        
        if errors:
            logger.warning(f"Loaded {len(all_documents)} documents with {len(errors)} errors")
        else:
            logger.info(f"Successfully loaded {len(all_documents)} documents from {len(file_paths)} files")
        
        return all_documents
    
    def load_from_directory(
        self,
        directory: str,
        recursive: bool = False,
        file_pattern: Optional[str] = None
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            file_pattern: Optional glob pattern for filtering files
            
        Returns:
            List of Document objects
        """
        if not os.path.isdir(directory):
            raise DocumentLoaderError(f"Directory not found: {directory}")
        
        path = Path(directory)
        
        # Get all files matching the pattern
        if recursive:
            pattern = f"**/{file_pattern}" if file_pattern else "**/*"
            files = path.rglob(pattern.split('/')[-1]) if file_pattern else path.rglob("*")
        else:
            pattern = file_pattern if file_pattern else "*"
            files = path.glob(pattern)
        
        # Filter by supported extensions
        supported_files = [
            str(f) for f in files
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS
        ]
        
        if not supported_files:
            logger.warning(f"No supported files found in {directory}")
            return []
        
        logger.info(f"Found {len(supported_files)} files in {directory}")
        return self.load_documents(supported_files)
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported extensions
        """
        return list(DocumentLoader.SUPPORTED_FORMATS.keys())


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Convenience function to load documents.
    
    Args:
        file_paths: List of file paths
        
    Returns:
        List of Document objects
    """
    loader = DocumentLoader()
    return loader.load_documents(file_paths)
