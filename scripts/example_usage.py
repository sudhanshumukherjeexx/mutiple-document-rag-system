"""
Example usage script for RAG System.
Demonstrates various use cases and features.
"""

import asyncio
import os


import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config_loader import load_config
from src.logger import setup_logging
from src.loaders.document_loader import DocumentLoader
from src.processing.summarizer import DocumentSummarizer
from src.processing.vector_store import VectorStoreManager
from src.pipeline.rag_pipeline import SelfCorrectedRAGPipeline
from src.metrics import get_metrics_collector

async def example_1_document_summarization():
    """Example 1: Summarize multiple documents."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Document Summarization")
    print("="*80 + "\n")
    
    # Load documents (replace with your actual files)
    document_files = [
        "sample_doc1.pdf",
        "sample_doc2.pdf"
    ]
    
    # Check if files exist
    existing_files = [f for f in document_files if os.path.exists(f)]
    
    if not existing_files:
        print("⚠️  No sample documents found. Skipping this example.")
        print("   Please add PDF files to the current directory.\n")
        return
    
    loader = DocumentLoader()
    documents = loader.load_documents(existing_files)
    print(f"✓ Loaded {len(documents)} document pages\n")
    
    # Summarize
    summarizer = DocumentSummarizer()
    print("🔄 Generating summary...\n")
    summary = summarizer.summarize(documents, verbose=False)
    
    print("📝 SUMMARY:")
    print("-" * 80)
    print(summary)
    print("-" * 80 + "\n")


async def example_2_basic_rag():
    """Example 2: Basic RAG query."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Basic RAG Query")
    print("="*80 + "\n")
    
    # Create sample documents for demonstration
    from langchain_core.documents import Document
    
    sample_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that focuses on learning from data. "
                       "It uses algorithms to identify patterns and make predictions without explicit programming.",
            metadata={"source": "intro.txt"}
        ),
        Document(
            page_content="Deep learning is a type of machine learning that uses neural networks with multiple layers. "
                       "It's particularly effective for image recognition, natural language processing, and speech recognition.",
            metadata={"source": "deep_learning.txt"}
        ),
        Document(
            page_content="Supervised learning is a machine learning approach where models are trained on labeled data. "
                       "The model learns to map inputs to outputs based on example input-output pairs.",
            metadata={"source": "supervised.txt"}
        )
    ]
    
    # Create vector store
    print("🔄 Creating knowledge base...\n")
    vector_manager = VectorStoreManager()
    retriever = vector_manager.create_vector_store(sample_docs)
    print("✓ Knowledge base created\n")
    
    # Initialize RAG pipeline
    pipeline = SelfCorrectedRAGPipeline(retriever)
    
    # Query
    question = "What is machine learning?"
    print(f"❓ Question: {question}\n")
    
    result = await pipeline.run(question)
    
    print("💬 Answer:")
    print(result.answer)
    print(f"\n⭐ Score: {result.score}/5")
    print(f"📊 Evaluation: {result.score_justification}\n")


async def example_3_self_correction():
    """Example 3: Demonstrate self-correction mechanism."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Self-Correction Mechanism")
    print("="*80 + "\n")
    
    from langchain_core.documents import Document
    
    # Create documents with limited information
    # This might trigger self-correction if the question requires more detail
    limited_docs = [
        Document(
            page_content="Python is a programming language. It was created by Guido van Rossum.",
            metadata={"source": "python_basic.txt"}
        )
    ]
    
    print("🔄 Creating knowledge base with limited information...\n")
    vector_manager = VectorStoreManager()
    retriever = vector_manager.create_vector_store(limited_docs)
    
    # Create pipeline with custom settings
    pipeline = SelfCorrectedRAGPipeline(
        retriever,
        max_correction_attempts=3,
        min_acceptable_score=4  # High threshold to potentially trigger correction
    )
    
    question = "Tell me everything about Python programming language including its history, features, and use cases."
    print(f"❓ Question: {question}\n")
    print("   (This question might trigger self-correction due to limited context)\n")
    
    result = await pipeline.run(question)
    
    print("💬 Answer:")
    print(result.answer)
    print(f"\n⭐ Score: {result.score}/5")
    print(f"🔄 Correction Attempts: {result.correction_attempts}")
    print(f"📊 Evaluation: {result.score_justification}\n")


async def example_4_guardrail_filtering():
    """Example 4: Demonstrate guardrail filtering."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Guardrail Filtering")
    print("="*80 + "\n")
    
    from langchain_core.documents import Document
    
    # Create documents with mixed relevance
    mixed_docs = [
        Document(
            page_content="Climate change is causing global temperatures to rise. "
                       "This leads to melting ice caps and rising sea levels.",
            metadata={"source": "climate.txt"}
        ),
        Document(
            page_content="Pizza is a popular Italian dish made with dough, tomato sauce, and cheese. "
                       "It can have various toppings like pepperoni, mushrooms, and olives.",
            metadata={"source": "food.txt"}
        ),
        Document(
            page_content="Renewable energy sources include solar, wind, and hydroelectric power. "
                       "They help reduce greenhouse gas emissions and combat climate change.",
            metadata={"source": "energy.txt"}
        )
    ]
    
    print("🔄 Creating knowledge base with mixed content...\n")
    vector_manager = VectorStoreManager()
    retriever = vector_manager.create_vector_store(mixed_docs)
    
    # Create pipeline with guardrail enabled
    pipeline = SelfCorrectedRAGPipeline(retriever, enable_guardrail=True)
    
    question = "What are the effects of climate change?"
    print(f"❓ Question: {question}\n")
    print("   (The guardrail should filter out the irrelevant pizza document)\n")
    
    result = await pipeline.run(question)
    
    print("💬 Answer:")
    print(result.answer)
    print(f"\n📊 Documents Retrieved: {result.documents_retrieved}")
    print(f"✓ Documents After Filtering: {result.documents_after_filter}")
    print(f"⭐ Score: {result.score}/5\n")


async def example_5_metrics_tracking():
    """Example 5: Demonstrate metrics tracking."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Metrics Tracking")
    print("="*80 + "\n")
    
    from langchain_core.documents import Document
    from metrics import get_metrics_collector
    
    # Create sample documents
    docs = [
        Document(
            page_content="Artificial Intelligence enables computers to perform tasks that typically require human intelligence.",
            metadata={"source": "ai.txt"}
        )
    ]
    
    print("🔄 Setting up for multiple queries...\n")
    vector_manager = VectorStoreManager()
    retriever = vector_manager.create_vector_store(docs)
    pipeline = SelfCorrectedRAGPipeline(retriever)
    
    # Run multiple queries
    questions = [
        "What is AI?",
        "Can you explain artificial intelligence?",
        "What does AI stand for?"
    ]
    
    print("Running multiple queries to collect metrics...\n")
    for i, question in enumerate(questions, 1):
        print(f"Query {i}: {question}")
        result = await pipeline.run(question)
        print(f"   → Score: {result.score}/5\n")
    
    # Display metrics
    print("\n📊 COLLECTED METRICS:")
    print("="*80)
    metrics_collector = get_metrics_collector()
    metrics_collector.print_summary()


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("RAG SYSTEM - USAGE EXAMPLES")
    print("="*80)
    
    # Initialize system
    load_config("config.yaml")
    setup_logging(level="INFO")
    
    # Set API key (make sure it's set in environment)
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not set in environment")
        print("   Please set your API key before running examples:")
        print("   export OPENAI_API_KEY='your-key-here'\n")
        return
    
    try:
        # Run examples
        await example_1_document_summarization()
        await example_2_basic_rag()
        await example_3_self_correction()
        await example_4_guardrail_filtering()
        await example_5_metrics_tracking()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
