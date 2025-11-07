"""
Unit tests for RAG System.
Run with: pytest test_rag_system.py -v
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

# Import modules to test
from config_loader import Config
from validators import InputValidator, ValidationError
from document_loader import DocumentLoader, DocumentLoaderError
from metrics import MetricsCollector, QueryMetrics
from agents import GuardrailCheck, Evaluation


class TestConfig:
    """Tests for configuration management."""
    
    def test_config_singleton(self):
        """Test that Config is a singleton."""
        config1 = Config()
        config2 = Config()
        assert config1 is config2
    
    def test_config_get_with_default(self):
        """Test getting config values with defaults."""
        config = Config()
        config._config = {'test': {'key': 'value'}}
        
        assert config.get('test.key') == 'value'
        assert config.get('missing.key', 'default') == 'default'
    
    def test_config_set(self):
        """Test setting config values."""
        config = Config()
        config._config = {}
        
        config.set('new.nested.key', 'value')
        assert config.get('new.nested.key') == 'value'


class TestValidators:
    """Tests for input validation."""
    
    def test_validate_query_success(self):
        """Test successful query validation."""
        validator = InputValidator(max_query_length=100)
        result = validator.validate_query("What is AI?")
        assert result == "What is AI?"
    
    def test_validate_query_empty(self):
        """Test validation fails on empty query."""
        validator = InputValidator()
        with pytest.raises(ValidationError):
            validator.validate_query("")
    
    def test_validate_query_too_long(self):
        """Test validation fails on too long query."""
        validator = InputValidator(max_query_length=10)
        with pytest.raises(ValidationError):
            validator.validate_query("This is a very long query that exceeds the limit")
    
    def test_validate_query_sanitization(self):
        """Test query sanitization."""
        validator = InputValidator(enable_sanitization=True)
        result = validator.validate_query("What  is   AI?")  # Multiple spaces
        assert result == "What is AI?"
    
    def test_validate_file_path_success(self):
        """Test successful file path validation."""
        validator = InputValidator()
        result = validator.validate_file_path("documents/file.pdf")
        assert result == "documents/file.pdf"
    
    def test_validate_file_path_traversal(self):
        """Test file path traversal is blocked."""
        validator = InputValidator()
        with pytest.raises(ValidationError):
            validator.validate_file_path("../../../etc/passwd")


class TestDocumentLoader:
    """Tests for document loading."""
    
    def test_supported_formats(self):
        """Test that supported formats are correct."""
        formats = DocumentLoader.get_supported_formats()
        assert '.pdf' in formats
        assert '.txt' in formats
        assert '.md' in formats
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        loader = DocumentLoader()
        with pytest.raises(DocumentLoaderError):
            loader.load_document("nonexistent_file.pdf")
    
    def test_unsupported_format(self):
        """Test loading unsupported format raises error."""
        loader = DocumentLoader()
        with pytest.raises(DocumentLoaderError):
            loader.load_document("file.docx")  # Not supported


class TestMetrics:
    """Tests for metrics collection."""
    
    def test_create_query_metrics(self):
        """Test creating query metrics."""
        collector = MetricsCollector(enabled=True)
        metrics = collector.create_query_metrics("test-id", "test question")
        
        assert metrics.query_id == "test-id"
        assert metrics.question == "test question"
        assert metrics.total_latency_ms == 0.0
    
    def test_record_query(self):
        """Test recording query metrics."""
        collector = MetricsCollector(enabled=True)
        metrics = QueryMetrics(
            query_id="test",
            timestamp="2024-01-01",
            question="test",
            final_score=5,
            success=True
        )
        
        collector.record_query(metrics)
        assert len(collector.query_metrics) == 1
    
    def test_aggregate_metrics(self):
        """Test calculating aggregate metrics."""
        collector = MetricsCollector(enabled=True)
        
        # Add some test metrics
        for i in range(3):
            metrics = QueryMetrics(
                query_id=f"test-{i}",
                timestamp="2024-01-01",
                question=f"question {i}",
                total_latency_ms=100.0,
                final_score=5,
                success=True
            )
            collector.record_query(metrics)
        
        agg = collector.get_aggregate_metrics()
        assert agg.total_queries == 3
        assert agg.successful_queries == 3
        assert agg.avg_score == 5.0


class TestAgentSchemas:
    """Tests for agent Pydantic schemas."""
    
    def test_guardrail_check_schema(self):
        """Test GuardrailCheck schema."""
        check = GuardrailCheck(
            is_relevant=True,
            justification="Context discusses the topic directly"
        )
        
        assert check.is_relevant is True
        assert isinstance(check.justification, str)
    
    def test_evaluation_schema(self):
        """Test Evaluation schema."""
        evaluation = Evaluation(
            score=5,
            justification="Answer is fully supported"
        )
        
        assert evaluation.score == 5
        assert 1 <= evaluation.score <= 5
    
    def test_evaluation_score_bounds(self):
        """Test Evaluation score is bounded."""
        # Valid scores
        for score in [1, 2, 3, 4, 5]:
            eval = Evaluation(score=score, justification="test")
            assert eval.score == score
        
        # Invalid scores should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            Evaluation(score=0, justification="test")
        
        with pytest.raises(Exception):
            Evaluation(score=6, justification="test")


class TestRAGPipeline:
    """Tests for RAG pipeline (integration tests)."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="Machine learning is a type of AI.",
                metadata={"source": "test1.txt"}
            ),
            Document(
                page_content="Deep learning uses neural networks.",
                metadata={"source": "test2.txt"}
            )
        ]
    
    @pytest.fixture
    def mock_retriever(self, sample_documents):
        """Create a mock retriever."""
        mock = MagicMock()
        mock.ainvoke = AsyncMock(return_value=sample_documents)
        return mock
    
    def test_rag_result_dataclass(self):
        """Test RAGResult dataclass."""
        from rag_pipeline import RAGResult
        
        result = RAGResult(
            question="test",
            answer="test answer",
            score=5,
            score_justification="perfect",
            source_context="context",
            documents_retrieved=2,
            documents_after_filter=2,
            correction_attempts=1,
            success=True
        )
        
        assert result.question == "test"
        assert result.score == 5
        assert result.success is True


# Helper class for async mocks
class AsyncMock(MagicMock):
    """Mock for async functions."""
    
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
