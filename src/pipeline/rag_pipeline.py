"""
Self-Corrected RAG Pipeline implementation.
Implements a complete RAG pipeline with self-correction capability.
"""

import uuid
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from ..config_loader import config
from ..metrics import Timer, QueryMetrics, get_metrics_collector
from ..validators import validate_query
from ..agents.agents import GuardrailAgent, GenerationAgent, EvaluationAgent

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG pipeline execution."""
    
    question: str
    answer: str
    score: int
    score_justification: str
    source_context: str
    
    # Metadata
    documents_retrieved: int
    documents_after_filter: int
    correction_attempts: int
    success: bool
    error_message: Optional[str] = None


class RAGPipelineError(Exception):
    """Custom exception for RAG pipeline errors."""
    pass


class SelfCorrectedRAGPipeline:
    """
    Self-corrected RAG pipeline with retrieval, filtering, generation, and evaluation.
    Implements true self-correction by regenerating answers with low scores.
    """
    
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        max_correction_attempts: Optional[int] = None,
        min_acceptable_score: Optional[int] = None,
        enable_guardrail: Optional[bool] = None,
        parallel_guardrail: Optional[bool] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Vector store retriever for document retrieval
            max_correction_attempts: Maximum number of correction attempts
            min_acceptable_score: Minimum acceptable evaluation score
            enable_guardrail: Whether to use guardrail filtering
            parallel_guardrail: Whether to run guardrail checks in parallel
        """
        self.retriever = retriever
        
        # Load configuration
        self.max_correction_attempts = max_correction_attempts or config.get('rag.max_correction_attempts', 3)
        self.min_acceptable_score = min_acceptable_score or config.get('rag.min_acceptable_score', 3)
        self.enable_guardrail = enable_guardrail if enable_guardrail is not None else config.get('rag.guardrail_enabled', True)
        self.parallel_guardrail = parallel_guardrail if parallel_guardrail is not None else config.get('rag.parallel_guardrail_checks', True)
        self.max_context_length = config.get('rag.max_context_length', 8000)
        
        # Initialize agents
        self.guardrail_agent = GuardrailAgent() if self.enable_guardrail else None
        self.generation_agent = GenerationAgent()
        self.evaluation_agent = EvaluationAgent()
        
        # Metrics
        self.metrics_collector = get_metrics_collector()
        
        logger.info(
            f"RAGPipeline initialized (max_attempts={self.max_correction_attempts}, "
            f"min_score={self.min_acceptable_score}, guardrail={self.enable_guardrail})"
        )
    
    async def run(self, question: str, query_id: Optional[str] = None) -> RAGResult:
        """
        Execute the complete RAG pipeline with self-correction.
        
        Args:
            question: User's question
            query_id: Optional unique query identifier
            
        Returns:
            RAGResult with answer and metadata
        """
        # Generate query ID
        query_id = query_id or str(uuid.uuid4())
        
        # Initialize metrics
        metrics = self.metrics_collector.create_query_metrics(query_id, question)
        
        logger.info(f"[{query_id}] Starting RAG pipeline for question: '{question}'")
        
        try:
            # Validate input
            question = validate_query(
                question,
                max_length=config.get('security.max_query_length', 1000)
            )
            
            # Execute pipeline with self-correction
            with Timer("Total pipeline"):
                result = await self._execute_with_correction(question, metrics)
            
            # Record metrics
            metrics.success = True
            metrics.final_score = result.score
            self.metrics_collector.record_query(metrics)
            
            logger.info(
                f"[{query_id}] Pipeline completed: score={result.score}/5, "
                f"attempts={result.correction_attempts}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"[{query_id}] Pipeline failed: {e}")
            metrics.success = False
            metrics.error_message = str(e)
            self.metrics_collector.record_query(metrics)
            
            return RAGResult(
                question=question,
                answer=f"Error: {str(e)}",
                score=0,
                score_justification="Pipeline execution failed",
                source_context="",
                documents_retrieved=0,
                documents_after_filter=0,
                correction_attempts=0,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_with_correction(
        self,
        question: str,
        metrics: QueryMetrics
    ) -> RAGResult:
        """
        Execute pipeline with self-correction loop.
        
        Args:
            question: User's question
            metrics: Metrics object to populate
            
        Returns:
            RAGResult
        """
        # Step 1: Retrieve documents
        documents = await self._retrieve_documents(question, metrics)
        
        if not documents:
            return self._create_no_documents_result(question)
        
        # Step 2: Filter with guardrail (if enabled)
        if self.enable_guardrail and self.guardrail_agent:
            relevant_docs = await self._filter_documents(question, documents, metrics)
            
            if not relevant_docs:
                return self._create_no_relevant_context_result(question, len(documents))
        else:
            relevant_docs = documents
        
        # Prepare context
        context = self._prepare_context(relevant_docs)
        
        # Update metrics
        metrics.documents_retrieved = len(documents)
        metrics.documents_after_filter = len(relevant_docs)
        if len(documents) > 0:
            metrics.filter_rejection_rate = 1 - (len(relevant_docs) / len(documents))
        
        # Step 3-5: Generate and evaluate with self-correction loop
        best_result = None
        best_score = 0
        
        for attempt in range(self.max_correction_attempts):
            logger.info(f"Correction attempt {attempt + 1}/{self.max_correction_attempts}")
            
            # Generate answer
            answer = await self._generate_answer(question, context, metrics)
            
            # Evaluate answer
            evaluation = await self._evaluate_answer(answer, context, metrics)
            
            # Track attempts
            metrics.correction_attempts = attempt + 1
            
            # Create result for this attempt
            current_result = RAGResult(
                question=question,
                answer=answer,
                score=evaluation.score,
                score_justification=evaluation.justification,
                source_context=context,
                documents_retrieved=len(documents),
                documents_after_filter=len(relevant_docs),
                correction_attempts=attempt + 1,
                success=True
            )
            
            # Keep track of best result
            if evaluation.score > best_score:
                best_score = evaluation.score
                best_result = current_result
            
            # Check if score is acceptable
            if evaluation.score >= self.min_acceptable_score:
                logger.info(
                    f"Acceptable score achieved: {evaluation.score}/{self.min_acceptable_score} "
                    f"on attempt {attempt + 1}"
                )
                return current_result
            
            # Log that we're retrying
            if attempt < self.max_correction_attempts - 1:
                logger.warning(
                    f"Score {evaluation.score} below threshold {self.min_acceptable_score}. "
                    f"Retrying... ({attempt + 2}/{self.max_correction_attempts})"
                )
                
                # Could implement strategy adjustment here
                # For example: retrieve more documents, adjust temperature, etc.
        
        # If we exhausted all attempts, return best result
        logger.warning(
            f"Max correction attempts reached. Best score: {best_score}/{self.min_acceptable_score}"
        )
        
        return best_result
    
    async def _retrieve_documents(
        self,
        question: str,
        metrics: QueryMetrics
    ) -> List[Document]:
        """Retrieve documents from vector store."""
        logger.info("Step 1: Retrieving documents...")
        
        with Timer("Retrieval") as timer:
            try:
                documents = await self.retriever.ainvoke(question)
            except Exception as e:
                logger.error(f"Retrieval failed: {e}")
                raise RAGPipelineError(f"Document retrieval failed: {e}")
        
        metrics.retrieval_latency_ms = timer.elapsed_ms
        logger.info(f"Retrieved {len(documents)} documents")
        
        return documents
    
    async def _filter_documents(
        self,
        question: str,
        documents: List[Document],
        metrics: QueryMetrics
    ) -> List[Document]:
        """Filter documents using guardrail agent."""
        logger.info("Step 2: Filtering documents with Guardrail Agent...")
        
        with Timer("Guardrail filtering") as timer:
            relevant_docs, justifications = await self.guardrail_agent.filter_documents(
                question,
                documents,
                parallel=self.parallel_guardrail
            )
        
        metrics.guardrail_latency_ms = timer.elapsed_ms
        logger.info(f"Filtered to {len(relevant_docs)}/{len(documents)} relevant documents")
        
        return relevant_docs
    
    async def _generate_answer(
        self,
        question: str,
        context: str,
        metrics: QueryMetrics
    ) -> str:
        """Generate answer from context."""
        logger.info("Step 3: Generating answer...")
        
        with Timer("Generation") as timer:
            answer = await self.generation_agent.generate_answer(question, context)
        
        metrics.generation_latency_ms += timer.elapsed_ms
        
        return answer
    
    async def _evaluate_answer(
        self,
        answer: str,
        context: str,
        metrics: QueryMetrics
    ) -> Any:
        """Evaluate answer quality."""
        logger.info("Step 4: Evaluating answer...")
        
        with Timer("Evaluation") as timer:
            evaluation = await self.evaluation_agent.evaluate_answer(answer, context)
        
        metrics.evaluation_latency_ms += timer.elapsed_ms
        logger.info(f"Evaluation score: {evaluation.score}/5 - {evaluation.justification}")
        
        return evaluation
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context from documents.
        
        Args:
            documents: List of relevant documents
            
        Returns:
            Combined context string
        """
        # Join documents with separators
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            logger.warning(
                f"Context length ({len(context)}) exceeds maximum ({self.max_context_length}). "
                "Truncating..."
            )
            context = context[:self.max_context_length] + "\n\n[Context truncated...]"
        
        return context
    
    def _create_no_documents_result(self, question: str) -> RAGResult:
        """Create result when no documents are retrieved."""
        return RAGResult(
            question=question,
            answer="I could not find any relevant information to answer this question.",
            score=0,
            score_justification="No documents retrieved",
            source_context="",
            documents_retrieved=0,
            documents_after_filter=0,
            correction_attempts=0,
            success=False,
            error_message="No documents retrieved"
        )
    
    def _create_no_relevant_context_result(
        self,
        question: str,
        total_docs: int
    ) -> RAGResult:
        """Create result when no relevant context is found."""
        return RAGResult(
            question=question,
            answer="I could not find any relevant information to answer this question.",
            score=0,
            score_justification="No relevant context found after filtering",
            source_context="",
            documents_retrieved=total_docs,
            documents_after_filter=0,
            correction_attempts=0,
            success=False,
            error_message="No relevant documents after filtering"
        )


async def run_rag_query(
    question: str,
    retriever: VectorStoreRetriever,
    **kwargs
) -> RAGResult:
    """
    Convenience function to run a RAG query.
    
    Args:
        question: User's question
        retriever: Vector store retriever
        **kwargs: Additional arguments for RAGPipeline
        
    Returns:
        RAGResult
    """
    pipeline = SelfCorrectedRAGPipeline(retriever, **kwargs)
    return await pipeline.run(question)
