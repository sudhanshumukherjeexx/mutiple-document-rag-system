"""
Document summarization module for RAG system.
Implements MapReduce summarization strategy for multipage documents.
"""

from typing import List, Optional
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config_loader import config
from metrics import Timer

logger = logging.getLogger(__name__)


class SummarizerError(Exception):
    """Custom exception for summarization errors."""
    pass


class DocumentSummarizer:
    """Summarizes documents using MapReduce strategy."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Initialize document summarizer.
        
        Args:
            model_name: OpenAI model name (defaults to config)
            temperature: Model temperature (defaults to config)
            chunk_size: Size of text chunks (defaults to config)
            chunk_overlap: Overlap between chunks (defaults to config)
        """
        # Load configuration
        self.model_name = model_name or config.get('models.summarize.name')
        self.temperature = temperature if temperature is not None else config.get('models.summarize.temperature')
        self.chunk_size = chunk_size or config.get('document_processing.summary_chunk_size')
        self.chunk_overlap = chunk_overlap or config.get('document_processing.summary_chunk_overlap')
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=config.get('models.summarize.max_tokens'),
                timeout=config.get('api.request_timeout')
            )
            logger.info(f"Initialized summarizer with model: {self.model_name}")
        except Exception as e:
            raise SummarizerError(f"Failed to initialize LLM: {e}")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Define prompts
        self._setup_prompts()
    
    def _setup_prompts(self) -> None:
        """Setup prompts for MapReduce summarization."""
        self.map_prompt = PromptTemplate.from_template(
            """You are an expert document summarizer. Write a concise and comprehensive summary of the following text.
Focus on the main ideas, key points, and important details. Keep the summary clear and well-structured.

TEXT:
{text}

CONCISE SUMMARY:"""
        )
        
        self.combine_prompt = PromptTemplate.from_template(
            """You are an expert document summarizer. You have been given summaries of different sections of a long document.
Combine these summaries into a single, coherent, and comprehensive summary. The final summary should:
- Capture all main ideas and key points from all sections
- Be well-organized and flow naturally
- Eliminate redundancy
- Maintain the most important details

SECTION SUMMARIES:
{text}

COMBINED COMPREHENSIVE SUMMARY:"""
        )
    
    def summarize(
        self,
        documents: List[Document],
        verbose: bool = False
    ) -> str:
        """
        Summarize a list of documents using MapReduce strategy.
        
        Args:
            documents: List of Document objects to summarize
            verbose: Whether to print detailed progress
            
        Returns:
            Summary text
            
        Raises:
            SummarizerError: If summarization fails
        """
        if not documents:
            raise SummarizerError("No documents provided for summarization")
        
        logger.info(f"Starting summarization of {len(documents)} documents")
        
        # Split documents into chunks
        with Timer("Document splitting"):
            split_docs = self.text_splitter.split_documents(documents)
        
        logger.info(
            f"Split {len(documents)} documents into {len(split_docs)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        
        # Create summarization chain
        try:
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=self.map_prompt,
                combine_prompt=self.combine_prompt,
                verbose=verbose
            )
        except Exception as e:
            raise SummarizerError(f"Failed to create summarization chain: {e}")
        
        # Run summarization
        try:
            with Timer("Summarization") as timer:
                result = chain.invoke({"input_documents": split_docs})
            
            summary = result.get('output_text', '')
            
            if not summary:
                raise SummarizerError("Summarization produced empty output")
            
            logger.info(f"Summarization completed in {timer.elapsed_ms:.0f}ms")
            logger.info(f"Summary length: {len(summary)} characters")
            
            return summary
            
        except Exception as e:
            raise SummarizerError(f"Summarization failed: {e}")
    
    def summarize_with_metadata(
        self,
        documents: List[Document],
        verbose: bool = False
    ) -> dict:
        """
        Summarize documents and return result with metadata.
        
        Args:
            documents: List of Document objects
            verbose: Whether to print detailed progress
            
        Returns:
            Dictionary with summary and metadata
        """
        summary = self.summarize(documents, verbose)
        
        return {
            'summary': summary,
            'metadata': {
                'num_source_documents': len(documents),
                'num_chunks_processed': len(self.text_splitter.split_documents(documents)),
                'model_used': self.model_name,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
        }


def summarize_documents(
    documents: List[Document],
    model_name: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Convenience function to summarize documents.
    
    Args:
        documents: List of documents to summarize
        model_name: Optional model name override
        verbose: Whether to print detailed progress
        
    Returns:
        Summary text
    """
    summarizer = DocumentSummarizer(model_name=model_name)
    return summarizer.summarize(documents, verbose=verbose)
