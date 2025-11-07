"""
Vector store module for RAG system.
Manages document embeddings and similarity search using FAISS.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional
from functools import lru_cache
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from config_loader import config
from metrics import Timer

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store errors."""
    pass


class VectorStoreManager:
    """Manages vector store for document retrieval."""
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize vector store manager.
        
        Args:
            embedding_model: Name of the embedding model
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            cache_enabled: Whether to enable caching
        """
        self.embedding_model_name = embedding_model or config.get('vector_store.embedding_model')
        self.chunk_size = chunk_size or config.get('document_processing.rag_chunk_size')
        self.chunk_overlap = chunk_overlap or config.get('document_processing.rag_chunk_overlap')
        self.cache_enabled = cache_enabled
        self.cache_dir = config.get('cache.cache_dir', '.cache')
        
        # Initialize components
        self.embeddings = self._load_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.vector_store: Optional[FAISS] = None
        self.retriever: Optional[VectorStoreRetriever] = None
        
        logger.info(
            f"Initialized VectorStoreManager with embedding model: {self.embedding_model_name}"
        )
    
    @lru_cache(maxsize=1)
    def _load_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Load embeddings model (cached).
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        logger.info(f"Loading embeddings model: {self.embedding_model_name}")
        
        with Timer("Embeddings model loading") as timer:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                cache_folder=self.cache_dir if self.cache_enabled else None
            )
        
        logger.info(f"Embeddings model loaded in {timer.elapsed_ms:.0f}ms")
        return embeddings
    
    def create_vector_store(
        self,
        documents: List[Document],
        save_path: Optional[str] = None
    ) -> VectorStoreRetriever:
        """
        Create vector store from documents.
        
        Args:
            documents: List of documents to index
            save_path: Optional path to save the vector store
            
        Returns:
            VectorStoreRetriever for querying
            
        Raises:
            VectorStoreError: If vector store creation fails
        """
        if not documents:
            raise VectorStoreError("No documents provided for vector store creation")
        
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        # Split documents
        with Timer("Document splitting"):
            split_docs = self.text_splitter.split_documents(documents)
        
        logger.info(
            f"Split {len(documents)} documents into {len(split_docs)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        
        # Create vector store
        try:
            with Timer("Vector store creation") as timer:
                self.vector_store = FAISS.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings
                )
            
            logger.info(f"Vector store created in {timer.elapsed_ms:.0f}ms")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to create vector store: {e}")
        
        # Save if requested
        if save_path and self.cache_enabled:
            self.save_vector_store(save_path)
        
        # Create retriever
        top_k = config.get('vector_store.top_k', 5)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        
        logger.info(f"Created retriever with top_k={top_k}")
        
        return self.retriever
    
    def save_vector_store(self, save_path: str) -> None:
        """
        Save vector store to disk.
        
        Args:
            save_path: Path to save the vector store
        """
        if self.vector_store is None:
            raise VectorStoreError("No vector store to save")
        
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(save_path)
            logger.info(f"Vector store saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise VectorStoreError(f"Failed to save vector store: {e}")
    
    def load_vector_store(self, load_path: str) -> VectorStoreRetriever:
        """
        Load vector store from disk.
        
        Args:
            load_path: Path to load the vector store from
            
        Returns:
            VectorStoreRetriever for querying
        """
        if not os.path.exists(load_path):
            raise VectorStoreError(f"Vector store not found at {load_path}")
        
        try:
            with Timer("Vector store loading") as timer:
                self.vector_store = FAISS.load_local(
                    load_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            logger.info(f"Vector store loaded from {load_path} in {timer.elapsed_ms:.0f}ms")
            
            top_k = config.get('vector_store.top_k', 5)
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            
            return self.retriever
            
        except Exception as e:
            raise VectorStoreError(f"Failed to load vector store: {e}")
    
    def get_retriever(self, top_k: Optional[int] = None) -> VectorStoreRetriever:
        """
        Get retriever with optional custom top_k.
        
        Args:
            top_k: Number of documents to retrieve (overrides config)
            
        Returns:
            VectorStoreRetriever instance
            
        Raises:
            VectorStoreError: If no vector store exists
        """
        if self.vector_store is None:
            raise VectorStoreError("No vector store available. Create or load one first.")
        
        if top_k is not None:
            return self.vector_store.as_retriever(search_kwargs={"k": top_k})
        
        if self.retriever is None:
            top_k = config.get('vector_store.top_k', 5)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        
        return self.retriever
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise VectorStoreError("No vector store available")
        
        k = k or config.get('vector_store.top_k', 5)
        
        with Timer(f"Similarity search (k={k})"):
            results = self.vector_store.similarity_search(query, k=k)
        
        logger.debug(f"Found {len(results)} similar documents")
        
        return results
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of documents to add
        """
        if self.vector_store is None:
            raise VectorStoreError("No vector store available")
        
        split_docs = self.text_splitter.split_documents(documents)
        
        logger.info(f"Adding {len(split_docs)} chunks to vector store")
        
        with Timer("Adding documents"):
            self.vector_store.add_documents(split_docs)
        
        logger.info("Documents added successfully")


def create_vector_store(
    documents: List[Document],
    save_path: Optional[str] = None
) -> VectorStoreRetriever:
    """
    Convenience function to create a vector store.
    
    Args:
        documents: List of documents to index
        save_path: Optional path to save the vector store
        
    Returns:
        VectorStoreRetriever instance
    """
    manager = VectorStoreManager()
    return manager.create_vector_store(documents, save_path)
