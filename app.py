"""
Gradio Web Interface for RAG System
Beautiful frontend for document summarization and RAG Q&A
"""

import gradio as gr
import os
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from datetime import datetime

# Import RAG system components

from src.config_loader import load_config, config
from src.logger import setup_logging
from src.loaders.document_loader import DocumentLoader
from src.processing.summarizer import DocumentSummarizer
from src.processing.vector_store import VectorStoreManager
from src.pipeline.rag_pipeline import SelfCorrectedRAGPipeline
from src.metrics import get_metrics_collector

# Load environment and config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

load_config("config/config.yaml")
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Global state
vector_store_manager = None
rag_pipeline = None
current_kb_name = None

# Custom CSS for beautiful, minimalist, adaptive UI
CUSTOM_CSS = """
:root {
  --bg: #0b0c0f;
  --panel: #111318;
  --subtle: #1a1d24;
  --muted: #9aa3af;
  --text: #e5e7eb;
  --accent: #7c8cfb; /* indigo-ish */
  --accent-2: #b889f6; /* violet-ish */
  --success: #22c55e;
  --error: #ef4444;
  --warn: #f59e0b;
  --radius: 14px;
  --shadow: 0 8px 24px rgba(0,0,0,0.25);
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #f7f8fb;
    --panel: #ffffff;
    --subtle: #f2f4f7;
    --muted: #6b7280;
    --text: #0f172a;
    --accent: #6366f1;
    --accent-2: #8b5cf6;
    --shadow: 0 8px 24px rgba(0,0,0,0.08);
  }
}

* { box-sizing: border-box; }

.gradio-container {
  max-width: 1100px;
  margin: 0 auto !important;
  padding: 24px !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Apple Color Emoji","Segoe UI Emoji" !important;
  background: var(--bg);
  color: var(--text);
}

/* Header */
.main-header {
  background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 28%, transparent), transparent 55%);
  border: 1px solid color-mix(in srgb, var(--accent) 25%, transparent);
  color: var(--text);
  padding: 28px 22px;
  border-radius: var(--radius);
  margin-bottom: 18px;
  box-shadow: var(--shadow);
}
.main-header h1 {
  margin: 0 0 8px 0;
  letter-spacing: -0.02em;
  font-weight: 700;
}
.main-header p {
  margin: 4px 0;
  color: var(--muted);
}

/* Panels / boxes */
.feature-box,
.info-box,
.stats-box {
  border-radius: var(--radius);
  padding: 14px 16px;
  border: 1px solid color-mix(in srgb, var(--text) 8%, transparent);
  box-shadow: none;
}

.feature-box { background: var(--panel); }
.info-box    { background: color-mix(in srgb, var(--accent) 7%, var(--panel)); }
.stats-box   { background: color-mix(in srgb, var(--warn) 7%, var(--panel)); }

.success-message { color: var(--success); font-weight: 600; }
.error-message   { color: var(--error); font-weight: 600; }

/* Tabs & sections */
.gr-tabs, .tabitem, .gr-block, .gr-box, .gr-group {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.gr-panel, .gr-box > div, .gr-accordion, .gr-accordion .label-wrap {
  background: var(--panel) !important;
  border-radius: var(--radius) !important;
  border: 1px solid color-mix(in srgb, var(--text) 8%, transparent) !important;
}

.gr-accordion .label-wrap {
  padding: 12px 14px !important;
}

/* Buttons */
.gr-button, button.svelte-1ipelgc {
  border-radius: 12px !important;
  border: 1px solid color-mix(in srgb, var(--text) 8%, transparent) !important;
  background: linear-gradient(180deg, var(--panel), color-mix(in srgb, var(--panel) 70%, black));
  color: var(--text);
  transition: transform .08s ease, box-shadow .12s ease, border-color .2s ease;
}
.gr-button:hover {
  transform: translateY(-1px);
  border-color: color-mix(in srgb, var(--accent) 40%, transparent);
  box-shadow: 0 6px 16px color-mix(in srgb, var(--accent) 24%, transparent);
}
.gr-button.primary, .gr-button--primary {
  background: linear-gradient(180deg, var(--accent), var(--accent-2));
  color: #fff !important;
  border: none !important;
}
.gr-button.secondary, .gr-button--secondary {
  background: var(--subtle);
}

/* Inputs */
textarea, input[type="text"], .gr-textbox, .gr-file, .gr-dropdown, .gr-select {
  background: var(--panel) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
  border: 1px solid color-mix(in srgb, var(--text) 10%, transparent) !important;
  transition: border-color .2s ease, box-shadow .15s ease;
}
textarea:focus, input[type="text"]:focus, .gr-textbox:focus-within {
  border-color: color-mix(in srgb, var(--accent) 45%, transparent) !important;
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--accent) 20%, transparent) !important;
  outline: none !important;
}
.gr-textbox textarea {
  line-height: 1.55;
}

/* Markdown bodies */
.gr-markdown, .gr-markdown p, .gr-markdown li, .gr-markdown code {
  color: var(--text) !important;
}
.gr-markdown code {
  background: var(--subtle);
  border-radius: 8px;
  padding: 2px 6px;
}

/* Chatbot */
.gr-chatbot {
  background: var(--panel) !important;
  border: 1px solid color-mix(in srgb, var(--text) 8%, transparent) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow);
}
.gr-chatbot .message {
  border-radius: 12px !important;
}
.gr-chatbot .message.user {
  background: color-mix(in srgb, var(--accent) 12%, var(--panel)) !important;
}
.gr-chatbot .message.bot {
  background: color-mix(in srgb, var(--accent-2) 10%, var(--panel)) !important;
}

/* Status & statistics blocks */
#status_display, .status, .stats {
  background: var(--panel) !important;
  border: 1px solid color-mix(in srgb, var(--text) 8%, transparent) !important;
  border-radius: var(--radius) !important;
  padding: 12px !important;
}

/* File uploader */
.gr-file {
  background: var(--panel) !important;
}

/* Subtle separators */
hr {
  border: 0;
  height: 1px;
  background: color-mix(in srgb, var(--text) 10%, transparent);
}

/* Responsive tweaks */
@media (max-width: 860px) {
  .gradio-container { padding: 16px !important; }
  .main-header { padding: 20px; }
}
"""

# Initialize components
def initialize_system():
    """Initialize the RAG system"""
    try:
        # Check API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return False, "❌ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file"
        
        logger.info("System initialized successfully")
        return True, "✅ System ready!"
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return False, f"❌ Initialization failed: {str(e)}"


def process_documents_for_summary(files: List) -> Tuple[str, str]:
    """
    Process uploaded files and generate summary
    
    Args:
        files: List of uploaded file objects from Gradio
        
    Returns:
        Tuple of (summary_text, status_message)
    """
    try:
        if not files:
            return "", "❌ Please upload at least one document"
        
        # Initialize components
        loader = DocumentLoader()
        summarizer = DocumentSummarizer()
        
        # Extract file paths
        file_paths = [file.name if hasattr(file, 'name') else file for file in files]
        
        logger.info(f"Processing {len(file_paths)} files for summarization")
        
        # Load documents
        documents = loader.load_documents(file_paths)
        
        if not documents:
            return "", "❌ Failed to load documents"
        
        # Generate summary
        logger.info(f"Generating summary for {len(documents)} document chunks...")
        summary = summarizer.summarize(documents, verbose=False)
        
        # Create status message
        status = f"""✅ **Summary Generated Successfully!**
        
📊 **Statistics:**
- Files processed: {len(file_paths)}
- Document chunks: {len(documents)}
- Summary length: {len(summary)} characters
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        logger.info("Summary generation completed")
        return summary, status
        
    except Exception as e:
        error_msg = f"❌ Error generating summary: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


def create_knowledge_base(files: List, kb_name: str) -> str:
    """
    Create a knowledge base from uploaded files
    
    Args:
        files: List of uploaded file objects
        kb_name: Name for the knowledge base
        
    Returns:
        Status message
    """
    global vector_store_manager, rag_pipeline, current_kb_name
    
    try:
        if not files:
            return "❌ Please upload at least one document"
        
        if not kb_name:
            kb_name = f"kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize components
        loader = DocumentLoader()
        
        # Extract file paths
        file_paths = [file.name if hasattr(file, 'name') else file for file in files]
        
        logger.info(f"Creating knowledge base '{kb_name}' from {len(file_paths)} files")
        
        # Load documents
        documents = loader.load_documents(file_paths)
        
        if not documents:
            return "❌ Failed to load documents"
        
        # Create vector store
        vector_store_manager = VectorStoreManager()
        retriever = vector_store_manager.create_vector_store(documents)
        
        # Initialize RAG pipeline
        rag_pipeline = SelfCorrectedRAGPipeline(retriever)
        current_kb_name = kb_name
        
        # Create status message
        status = f"""✅ **Knowledge Base Created Successfully!**

📚 **Knowledge Base:** {kb_name}

📊 **Statistics:**
- Files processed: {len(file_paths)}
- Document chunks indexed: {len(documents)}
- Vector store: FAISS
- Embeddings: all-MiniLM-L6-v2
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🎯 **You can now ask questions in the chat below!**
"""
        
        logger.info(f"Knowledge base '{kb_name}' created successfully")
        return status
        
    except Exception as e:
        error_msg = f"❌ Error creating knowledge base: {str(e)}"
        logger.error(error_msg)
        return error_msg


async def chat_with_documents(message: str, history: List) -> Tuple[str, List]:
    """
    Chat interface for RAG Q&A
    
    Args:
        message: User's question
        history: Chat history
        
    Returns:
        Tuple of (response, updated_history)
    """
    global rag_pipeline, current_kb_name
    
    try:
        if not rag_pipeline:
            error_msg = "❌ Please create a knowledge base first by uploading documents in the 'RAG Setup' section above."
            history.append((message, error_msg))
            return "", history
        
        if not message or not message.strip():
            return "", history
        
        logger.info(f"Processing question: {message}")
        
        # Run RAG pipeline
        result = await rag_pipeline.run(message)
        
        # Format response with metadata
        response = f"""{result.answer}

---
📊 **Query Statistics:**
- Quality Score: {result.score}/5 ⭐
- Documents Retrieved: {result.documents_retrieved}
- Documents Used: {result.documents_after_filter}
- Correction Attempts: {result.correction_attempts}
- Knowledge Base: {current_kb_name or 'Unnamed'}
"""
        
        # Add to history in messages format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        logger.info(f"Response generated with score: {result.score}/5")
        return "", history
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(error_msg)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return "", history


def clear_knowledge_base() -> str:
    """Clear the current knowledge base"""
    global vector_store_manager, rag_pipeline, current_kb_name
    
    vector_store_manager = None
    rag_pipeline = None
    current_kb_name = None
    
    logger.info("Knowledge base cleared")
    return "✅ Knowledge base cleared. Upload new documents to create a new one."


def get_system_stats() -> str:
    """Get system statistics"""
    try:
        metrics_collector = get_metrics_collector()
        agg_metrics = metrics_collector.get_aggregate_metrics()
        
        if agg_metrics.total_queries == 0:
            return "📊 No queries processed yet."
        
        stats = f"""📊 **System Statistics**

**Query Metrics:**
- Total Queries: {agg_metrics.total_queries}
- Successful: {agg_metrics.successful_queries}
- Failed: {agg_metrics.failed_queries}
- Success Rate: {(agg_metrics.successful_queries/agg_metrics.total_queries*100):.1f}%

**Performance:**
- Avg Latency: {agg_metrics.avg_latency_ms:.0f}ms
- Min Latency: {agg_metrics.min_latency_ms:.0f}ms
- Max Latency: {agg_metrics.max_latency_ms:.0f}ms

**Quality:**
- Avg Score: {agg_metrics.avg_score:.2f}/5 ⭐
- Avg Corrections: {agg_metrics.avg_correction_attempts:.2f}

**Efficiency:**
- Avg Filter Rejection: {agg_metrics.avg_filter_rejection_rate*100:.1f}%
"""
        return stats
        
    except Exception as e:
        return f"❌ Error getting stats: {str(e)}"


# Create Gradio Interface
def create_interface():
    """Create the Gradio web interface"""
    
    with gr.Blocks(
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter")
        ),
        title="RAG System"
    ) as app:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🤖 Intelligent Document Analysis System</h1>
            <p style="font-size: 1.06rem; margin-top: .25rem;">
                AI-Powered Document Summarization & Question Answering
            </p>
            <p style="font-size: .92rem;">
                Powered by GPT-4 | Self-Correcting RAG | Production-Ready
            </p>
        </div>
        """)
        
        # System status
        with gr.Row():
            status_display = gr.Textbox(
                label="System Status",
                value="Initializing...",
                interactive=False,
                scale=3
            )
            stats_btn = gr.Button("📊 View Statistics", scale=1, size="sm")
        
        # Main tabs
        with gr.Tabs() as tabs:
            
            # Tab 1: Document Summarization
            with gr.TabItem("📝 Document Summarization", id=0):
                gr.Markdown("""
                ### Upload Your Documents
                Upload one or multiple documents (PDF, TXT, MD) to generate an AI-powered summary.
                
                **Features:**
                - ✅ Multi-document summarization
                - ✅ MapReduce strategy for long documents
                - ✅ High-quality GPT-4 summaries
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        summary_files = gr.File(
                            label="Upload Documents",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"],
                            type="filepath"
                        )
                        summarize_btn = gr.Button(
                            "✨ Generate Summary",
                            variant="primary",
                            size="lg"
                        )
                        summary_status = gr.Markdown(label="Status")
                    
                    with gr.Column(scale=2):
                        summary_output = gr.Textbox(
                            label="Generated Summary",
                            lines=20,
                            placeholder="Your summary will appear here...",
                            show_copy_button=True
                        )
                
                # Examples
                gr.Markdown("""
                ### 💡 Tips for Best Results
                - Upload clear, well-formatted documents
                - Multiple related documents work great together
                - Longer documents may take a few moments to process
                """)
            
            # Tab 2: RAG Q&A
            with gr.TabItem("💬 RAG Question Answering", id=1):
                gr.Markdown("""
                ### Intelligent Document Q&A
                Upload documents to create a knowledge base, then ask questions!
                
                **Features:**
                - ✅ Self-correcting answers (automatically improves low-quality responses)
                - ✅ Guardrail filtering (removes irrelevant content)
                - ✅ Quality scoring (1-5 stars per answer)
                - ✅ Source tracking
                """)
                
                # Knowledge Base Setup
                with gr.Accordion("🔧 RAG Setup - Create Knowledge Base", open=True):
                    with gr.Row():
                        with gr.Column(scale=2):
                            rag_files = gr.File(
                                label="Upload Documents for Knowledge Base",
                                file_count="multiple",
                                file_types=[".pdf", ".txt", ".md"],
                                type="filepath"
                            )
                        with gr.Column(scale=1):
                            kb_name_input = gr.Textbox(
                                label="Knowledge Base Name (optional)",
                                placeholder="e.g., company_docs, research_papers",
                                value=""
                            )
                    
                    with gr.Row():
                        create_kb_btn = gr.Button(
                            "📚 Create Knowledge Base",
                            variant="primary",
                            size="lg",
                            scale=2
                        )
                        clear_kb_btn = gr.Button(
                            "🗑️ Clear Knowledge Base",
                            variant="secondary",
                            size="lg",
                            scale=1
                        )
                    
                    kb_status = gr.Markdown(label="Knowledge Base Status")
                
                # Chat Interface
                gr.Markdown("### 💭 Ask Questions")
                
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=500,
                    show_copy_button=True,
                    type="messages",  # Add this line
                    avatar_images=(
                        "https://api.dicebear.com/7.x/avataaars/svg?seed=user",
                        "https://api.dicebear.com/7.x/bottts/svg?seed=assistant"
                    )
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your documents...",
                        scale=5,
                        show_label=False
                    )
                    submit_btn = gr.Button("Send 🚀", scale=1, variant="primary")
                
                # Chat
                submit_btn.click(
                    fn=chat_with_documents,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot]
                )

                msg_input.submit(
                    fn=chat_with_documents,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot]
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        clear_chat_btn = gr.Button("🔄 Clear Chat")
                    with gr.Column(scale=3):
                        gr.Markdown("")  # Empty space
                
                
                # Examples
                gr.Examples(
                    examples=[
                        ["What are the main topics discussed in the documents?"],
                        ["Can you summarize the key findings?"],
                        ["What are the recommendations mentioned?"],
                        ["Explain the methodology used."],
                        ["What are the limitations discussed?"]
                    ],
                    inputs=msg_input,
                    label="💡 Example Questions"
                )
            
            # Tab 3: System Info
            with gr.TabItem("ℹ️ System Information", id=2):
                gr.Markdown("""
                ## About This System
                
                This is a production-ready RAG (Retrieval-Augmented Generation) system with:
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### 🎯 Key Features
                        
                        **Document Processing:**
                        - Multi-format support (PDF, TXT, MD)
                        - Intelligent text chunking
                        - FAISS vector indexing
                        
                        **RAG Pipeline:**
                        - Retrieval: Top-k similarity search
                        - Guardrail: Relevance filtering
                        - Generation: GPT-4 powered answers
                        - Evaluation: Automatic quality scoring
                        
                        **Self-Correction:**
                        - Automatically regenerates low-quality answers
                        - Up to 3 correction attempts
                        - Ensures factual consistency
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### ⚙️ Configuration
                        
                        **Models:**
                        - Summarization: GPT-4o-mini
                        - Answer Generation: GPT-4o
                        - Embeddings: all-MiniLM-L6-v2
                        
                        **Performance:**
                        - Parallel guardrail checks
                        - Vector store caching
                        - Optimized chunking
                        
                        **Quality Assurance:**
                        - Min acceptable score: 3/5
                        - Max correction attempts: 3
                        - Comprehensive metrics tracking
                        """)
                
                stats_display = gr.Markdown(label="Statistics")
                refresh_stats_btn = gr.Button("🔄 Refresh Statistics")
                
                gr.Markdown("""
                ---
                ### 📚 Documentation
                
                - **README.md** - Complete user guide
                - **PROJECT_STRUCTURE.md** - Architecture details
                - **QUICK_REFERENCE.md** - Command reference
                
                ### 🔒 Security
                
                - Input validation and sanitization
                - API key protection
                - No data persistence (privacy-first)
                
                ### 📊 Metrics
                
                All interactions are tracked for quality monitoring:
                - Query latency
                - Quality scores
                - Success rates
                - Token usage
                """)
        
        # Event handlers
        
        # Initialize on load
        app.load(
            fn=initialize_system,
            outputs=[status_display],
        )
        
        # Summarization
        summarize_btn.click(
            fn=process_documents_for_summary,
            inputs=[summary_files],
            outputs=[summary_output, summary_status]
        )
        
        # RAG setup
        create_kb_btn.click(
            fn=create_knowledge_base,
            inputs=[rag_files, kb_name_input],
            outputs=[kb_status]
        )
        
        clear_kb_btn.click(
            fn=clear_knowledge_base,
            outputs=[kb_status]
        )
        
        # Chat
        submit_btn.click(
            fn=chat_with_documents,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        msg_input.submit(
            fn=chat_with_documents,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        clear_chat_btn.click(
            fn=lambda: [],
            outputs=[chatbot]
        )
        
        # Statistics
        stats_btn.click(
            fn=get_system_stats,
            outputs=[stats_display]
        )
        
        refresh_stats_btn.click(
            fn=get_system_stats,
            outputs=[stats_display]
        )
    
    return app


# Main entry point
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Starting RAG System Web Interface")
    print("="*80)
    
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  WARNING: OPENAI_API_KEY not found!")
        print("Please set your API key in .env file or environment variable\n")
    
    # Create and launch interface
    app = create_interface()
    
    print("\n✅ Interface ready!")
    print("📱 Opening in browser...\n")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True,
        quiet=False
    )
