"""
Main application entry point for RAG System.
Provides CLI interface for document summarization and RAG queries.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import List, Optional
import logging

from config_loader import load_config, config
from logger import setup_logging
from metrics import initialize_metrics, get_metrics_collector
from document_loader import DocumentLoader
from summarizer import DocumentSummarizer
from vector_store import VectorStoreManager
from rag_pipeline import SelfCorrectedRAGPipeline

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
    logger_temp = logging.getLogger(__name__)
    logger_temp.debug(".env file loaded successfully")
except ImportError:
    # python-dotenv not installed, skip
    pass


class RAGApplication:
    """Main application class for RAG system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        load_config(config_path)
        
        # Setup logging
        setup_logging(
            level=config.get('logging.level', 'INFO'),
            log_format=config.get('logging.format'),
            log_to_file=config.get('logging.log_to_file', True),
            log_file=config.get('logging.log_file', 'logs/rag_system.log'),
            max_log_size=config.get('logging.max_log_size', 10485760),
            backup_count=config.get('logging.backup_count', 5)
        )
        
        # Initialize metrics
        initialize_metrics(
            metrics_file=config.get('monitoring.metrics_file', 'logs/metrics.json'),
            enabled=config.get('monitoring.enabled', True)
        )
        
        # Set OpenAI API key
        try:
            # First check if already in environment (from .env or manual export)
            api_key = os.environ.get("OPENAI_API_KEY")
            
            if not api_key:
                # Try to get from config (which checks environment)
                api_key = config.get_openai_api_key()
                os.environ["OPENAI_API_KEY"] = api_key
            
            logger.info("OpenAI API key loaded successfully")
        except ValueError as e:
            logger.error(f"Failed to load API key: {e}")
            sys.exit(1)
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.summarizer = DocumentSummarizer()
        self.vector_store_manager = VectorStoreManager()
        self.rag_pipeline: Optional[SelfCorrectedRAGPipeline] = None
        
        logger.info("RAG Application initialized successfully")
    
    def summarize_documents(
        self,
        file_paths: List[str],
        output_file: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        Summarize documents from file paths.
        
        Args:
            file_paths: List of document file paths
            output_file: Optional file to save summary
            verbose: Whether to print detailed progress
            
        Returns:
            Summary text
        """
        logger.info(f"Starting document summarization for {len(file_paths)} files")
        
        # Load documents
        print("\n📄 Loading documents...")
        documents = self.document_loader.load_documents(file_paths)
        print(f"✓ Loaded {len(documents)} document pages/sections\n")
        
        # Summarize
        print("🔄 Generating summary...")
        summary = self.summarizer.summarize(documents, verbose=verbose)
        print(f"✓ Summary generated ({len(summary)} characters)\n")
        
        # Save if requested
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(summary)
            print(f"💾 Summary saved to: {output_file}\n")
            logger.info(f"Summary saved to {output_file}")
        
        return summary
    
    def create_knowledge_base(
        self,
        file_paths: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create knowledge base from documents.
        
        Args:
            file_paths: List of document file paths
            save_path: Optional path to save vector store
        """
        logger.info(f"Creating knowledge base from {len(file_paths)} files")
        
        # Load documents
        print("\n📄 Loading documents...")
        documents = self.document_loader.load_documents(file_paths)
        print(f"✓ Loaded {len(documents)} document pages/sections\n")
        
        # Create vector store
        print("🔄 Creating vector store...")
        retriever = self.vector_store_manager.create_vector_store(documents, save_path)
        print(f"✓ Vector store created\n")
        
        if save_path:
            print(f"💾 Vector store saved to: {save_path}\n")
        
        # Initialize RAG pipeline
        self.rag_pipeline = SelfCorrectedRAGPipeline(retriever)
        logger.info("Knowledge base created successfully")
    
    def load_knowledge_base(self, load_path: str) -> None:
        """
        Load existing knowledge base.
        
        Args:
            load_path: Path to saved vector store
        """
        logger.info(f"Loading knowledge base from {load_path}")
        
        print(f"\n📂 Loading vector store from {load_path}...")
        retriever = self.vector_store_manager.load_vector_store(load_path)
        print("✓ Vector store loaded\n")
        
        # Initialize RAG pipeline
        self.rag_pipeline = SelfCorrectedRAGPipeline(retriever)
        logger.info("Knowledge base loaded successfully")
    
    async def query(
        self,
        question: str,
        show_context: bool = False,
        show_details: bool = False
    ) -> dict:
        """
        Query the knowledge base.
        
        Args:
            question: Question to ask
            show_context: Whether to show source context
            show_details: Whether to show detailed metrics
            
        Returns:
            Result dictionary
        """
        if self.rag_pipeline is None:
            raise RuntimeError(
                "Knowledge base not initialized. "
                "Create or load a knowledge base first."
            )
        
        logger.info(f"Processing query: {question}")
        
        # Run RAG pipeline
        result = await self.rag_pipeline.run(question)
        
        # Display result
        self._display_result(result, show_context, show_details)
        
        return {
            'question': result.question,
            'answer': result.answer,
            'score': result.score,
            'success': result.success
        }
    
    def _display_result(self, result, show_context: bool, show_details: bool) -> None:
        """Display query result."""
        print("\n" + "="*80)
        print("QUERY RESULT")
        print("="*80)
        print(f"\n❓ Question: {result.question}\n")
        print(f"💬 Answer:\n{result.answer}\n")
        print(f"⭐ Quality Score: {result.score}/5")
        print(f"📊 Evaluation: {result.score_justification}\n")
        
        if show_details:
            print(f"📈 Details:")
            print(f"   - Documents Retrieved: {result.documents_retrieved}")
            print(f"   - Documents After Filter: {result.documents_after_filter}")
            print(f"   - Correction Attempts: {result.correction_attempts}")
            print(f"   - Success: {result.success}\n")
        
        if show_context:
            print(f"📚 Source Context:")
            print("-" * 80)
            print(result.source_context[:500] + "..." if len(result.source_context) > 500 else result.source_context)
            print("-" * 80 + "\n")
        
        print("="*80 + "\n")
    
    async def interactive_mode(self) -> None:
        """Run interactive query mode."""
        if self.rag_pipeline is None:
            print("❌ Error: Knowledge base not initialized.")
            print("Please create or load a knowledge base first.\n")
            return
        
        print("\n" + "="*80)
        print("INTERACTIVE RAG QUERY MODE")
        print("="*80)
        print("Enter your questions (type 'exit' or 'quit' to stop)")
        print("Commands: 'context' - toggle context display, 'details' - toggle details")
        print("="*80 + "\n")
        
        show_context = False
        show_details = False
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break
                
                if question.lower() == 'context':
                    show_context = not show_context
                    print(f"✓ Context display: {'ON' if show_context else 'OFF'}\n")
                    continue
                
                if question.lower() == 'details':
                    show_details = not show_details
                    print(f"✓ Details display: {'ON' if show_details else 'OFF'}\n")
                    continue
                
                if not question:
                    continue
                
                await self.query(question, show_context, show_details)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                logger.error(f"Error in interactive mode: {e}")
    
    def show_metrics(self) -> None:
        """Display collected metrics."""
        metrics_collector = get_metrics_collector()
        metrics_collector.print_summary()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG System - Document Summarization and Question Answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize documents
  python main.py summarize doc1.pdf doc2.pdf --output summary.txt
  
  # Create knowledge base
  python main.py create-kb doc1.pdf doc2.pdf --save kb.index
  
  # Query knowledge base
  python main.py query "What is machine learning?" --kb kb.index
  
  # Interactive mode
  python main.py interactive --kb kb.index
  
  # Show metrics
  python main.py metrics
        """
    )
    
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Summarize documents')
    summarize_parser.add_argument('files', nargs='+', help='Document files to summarize')
    summarize_parser.add_argument('--output', '-o', help='Output file for summary')
    summarize_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Create knowledge base command
    create_kb_parser = subparsers.add_parser('create-kb', help='Create knowledge base')
    create_kb_parser.add_argument('files', nargs='+', help='Document files for knowledge base')
    create_kb_parser.add_argument('--save', '-s', help='Path to save vector store')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query knowledge base')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--kb', required=True, help='Path to knowledge base or documents')
    query_parser.add_argument('--context', action='store_true', help='Show source context')
    query_parser.add_argument('--details', action='store_true', help='Show detailed metrics')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Interactive query mode')
    interactive_parser.add_argument('--kb', required=True, help='Path to knowledge base or documents')
    
    # Metrics command
    subparsers.add_parser('metrics', help='Show metrics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize application
    try:
        app = RAGApplication(config_path=args.config)
    except Exception as e:
        print(f"❌ Error initializing application: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'summarize':
            summary = app.summarize_documents(
                args.files,
                output_file=args.output,
                verbose=args.verbose
            )
            print("📝 SUMMARY")
            print("="*80)
            print(summary)
            print("="*80 + "\n")
        
        elif args.command == 'create-kb':
            app.create_knowledge_base(args.files, save_path=args.save)
            print("✅ Knowledge base created successfully\n")
        
        elif args.command == 'query':
            # Check if kb path is a vector store or documents
            if os.path.isdir(args.kb):
                app.load_knowledge_base(args.kb)
            else:
                # Treat as document files
                files = args.kb.split(',') if ',' in args.kb else [args.kb]
                app.create_knowledge_base(files)
            
            asyncio.run(app.query(args.question, args.context, args.details))
        
        elif args.command == 'interactive':
            # Load knowledge base
            if os.path.isdir(args.kb):
                app.load_knowledge_base(args.kb)
            else:
                files = args.kb.split(',') if ',' in args.kb else [args.kb]
                app.create_knowledge_base(files)
            
            asyncio.run(app.interactive_mode())
        
        elif args.command == 'metrics':
            app.show_metrics()
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
