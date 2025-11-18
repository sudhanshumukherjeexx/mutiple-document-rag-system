# RAG System - Document Summarization and Self-Corrected Question Answering

A production-ready Retrieval-Augmented Generation (RAG) system with document summarization capabilities and true self-correction mechanisms.

use the app here: [live](https://huggingface.co/spaces/sudhanshumukherjeexx/rag-chatbot)

## Features

### Document Summarization
- ✅ **MapReduce Strategy**: Efficiently summarizes multi-page documents
- ✅ **Multiple Formats**: Supports PDF, TXT, and Markdown files
- ✅ **Configurable Chunking**: Customizable chunk sizes and overlap
- ✅ **Progress Tracking**: Detailed logging and timing metrics

### Self-Corrected RAG Pipeline
- ✅ **4-Stage Pipeline**: Retrieval → Guardrail → Generation → Evaluation
- ✅ **True Self-Correction**: Automatically regenerates low-quality answers
- ✅ **Guardrail Filtering**: Removes irrelevant context before generation
- ✅ **Quality Evaluation**: Scores answers for factual consistency
- ✅ **Parallel Processing**: Concurrent guardrail checks for speed
- ✅ **Metrics Tracking**: Comprehensive performance monitoring

### Production-Ready Features
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Configuration Management**: YAML-based configuration
- ✅ **Structured Logging**: Rotating log files with multiple levels
- ✅ **Input Validation**: Security-focused input sanitization
- ✅ **Error Handling**: Graceful degradation and informative errors
- ✅ **Caching**: Vector store and embedding caching
- ✅ **Metrics Collection**: Performance and quality metrics
- ✅ **CLI Interface**: Easy-to-use command-line interface
- ✅ **Interactive Mode**: Real-time question answering

## Installation

### Prerequisites
- Python 3.11 or higher
- OpenAI API key

### Setup

1. **Clone or download the repository**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up your OpenAI API key**

   **Option 1: Using .env file (Recommended)**
   ```bash
   # Copy the template
   cp .env.template .env
   
   # Edit .env and add your API key
   # OPENAI_API_KEY=sk-your-actual-key-here
   ```
   
   **Option 2: Export in terminal**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Option 3: Add to shell profile (Permanent)**
   ```bash
   echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

4. **Configure the system** (optional)
   
   Edit `config.yaml` to customize:
   - Model selections (GPT-4, GPT-4o-mini, etc.)
   - Chunk sizes and parameters
   - RAG pipeline settings
   - Logging and metrics options

## Quick Start

### Option 1: Web Interface (Recommended) 🌐

Launch the beautiful web interface:

```bash
./launch_web.sh
# Or manually:
python app.py
```

Then open your browser to `http://localhost:7860`

**Features:**
- 📝 Document summarization with drag-and-drop
- 💬 Interactive Q&A chat interface
- 📊 Real-time statistics dashboard
- 🎨 Beautiful, intuitive UI

See [WEB_INTERFACE_GUIDE.md](WEB_INTERFACE_GUIDE.md) for detailed instructions.

### Option 2: Command Line Interface

For command-line usage:

### 1. Summarize Documents

```bash
python scripts/main.py summarize document1.pdf document2.pdf --output summary.txt
```

### 2. Create Knowledge Base

```bash
python scripts/main.py create-kb document1.pdf document2.pdf document3.pdf --save ./knowledge_base
```

### 3. Query the Knowledge Base

```bash
python scripts/main.py query "What are the main findings?" --kb ./knowledge_base
```

### 4. Interactive Mode

```bash
python scripts/main.py interactive --kb ./knowledge_base
```

In interactive mode:
- Type your questions and press Enter
- Type `context` to toggle context display
- Type `details` to toggle detailed metrics
- Type `exit` or `quit` to stop

### 5. View Metrics

```bash
python scripts/main.py metrics
```

## Usage Examples

### Example 1: Research Paper Summarization
```bash
# Summarize multiple research papers
python scripts/main.py summarize \
    paper1.pdf paper2.pdf paper3.pdf \
    --output research_summary.txt \
    --verbose
```

### Example 2: Company Knowledge Base
```bash
# Create knowledge base from company documents
python scripts/main.py create-kb \
    policies/*.pdf \
    handbooks/*.pdf \
    reports/*.pdf \
    --save ./company_kb

# Query the knowledge base
python scripts/main.py query \
    "What is our vacation policy?" \
    --kb ./company_kb \
    --context
```

### Example 3: Technical Documentation
```bash
# Create KB from documentation
python scripts/main.py create-kb \
    docs/api.md \
    docs/tutorial.md \
    docs/reference.pdf \
    --save ./tech_docs_kb

# Interactive technical support
python scripts/main.py interactive --kb ./tech_docs_kb
```

## Architecture

### Module Structure

```
rag_system/
├── src/                    # All source code
│   ├── loaders/           # Document loading
│   ├── processing/        # Summarizer, vector store
│   ├── agents/            # RAG agents
│   ├── pipeline/          # RAG pipeline
├── scripts/               # Executable scripts
|   ├── main.py             # CLI entry point
├── tests/                 # Unit tests
├── config/                # Config files
├── docs/                  # Documentation
└── app.py                 # Gradio app
└── [root files]           # requirements.txt, .gitignore, etc.
```



### Pipeline Flow

```
User Question
    ↓
[1. RETRIEVAL]
    ↓ (Vector similarity search)
Retrieved Documents
    ↓
[2. GUARDRAIL FILTERING]
    ↓ (Relevance checking)
Filtered Documents
    ↓
[3. ANSWER GENERATION]
    ↓ (Context-aware generation)
Generated Answer
    ↓
[4. EVALUATION]
    ↓ (Factual consistency scoring)
Score < Threshold? → YES → [Back to Step 3] (Max 3 attempts)
    ↓ NO
Final Answer
```

### Self-Correction Mechanism

The system implements **true self-correction** by:
1. Evaluating each generated answer for factual consistency (score 1-5)
2. If score < threshold (default: 3), regenerate the answer
3. Repeat up to max attempts (default: 3)
4. Return the best answer achieved

## Configuration

### Key Configuration Options

```yaml
# Models
models:
  summarize:
    name: "gpt-4o-mini"  # Cost-effective for summarization
  generate:
    name: "gpt-4o"       # High quality for answers
  
# RAG Settings
rag:
  max_correction_attempts: 3    # Number of regeneration attempts
  min_acceptable_score: 3       # Minimum quality threshold
  guardrail_enabled: true       # Enable filtering
  parallel_guardrail_checks: true  # Speed up filtering
  
# Vector Store
vector_store:
  top_k: 5  # Number of documents to retrieve
  
# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
```

## Advanced Usage

### Custom Configuration

```bash
python scripts/main.py --config custom_config.yaml summarize document.pdf
```

### Programmatic Usage

```python
from src.config_loader import load_config
from src.loaders.document_loader import DocumentLoader
from src.processing.summarizer import DocumentSummarizer
from src.processing.vector_store import VectorStoreManager
from src.pipeline.rag_pipeline import SelfCorrectedRAGPipeline
import asyncio

# Load configuration
load_config("config/config.yaml")

# Load and summarize documents
loader = DocumentLoader()
docs = loader.load_documents(["doc1.pdf", "doc2.pdf"])

summarizer = DocumentSummarizer()
summary = summarizer.summarize(docs)
print(summary)

# Create RAG pipeline
vector_store = VectorStoreManager()
retriever = vector_store.create_vector_store(docs)

pipeline = SelfCorrectedRAGPipeline(retriever)
result = asyncio.run(pipeline.run("What is the main topic?"))
print(result.answer)
```

### Custom Agents

```python
from src.agents.agents import GuardrailAgent, GenerationAgent, EvaluationAgent

# Use agents individually
guardrail = GuardrailAgent()
generator = GenerationAgent()
evaluator = EvaluationAgent()

# Filter documents
question = "What is AI?"
relevant_docs = await guardrail.filter_documents(question, documents)

# Generate answer
answer = await generator.generate_answer(question, context)

# Evaluate quality
evaluation = await evaluator.evaluate_answer(answer, context)
```

## Metrics and Monitoring

The system tracks:
- **Performance**: Latency per pipeline stage
- **Quality**: Score distribution and average scores
- **Efficiency**: Token usage, filter rejection rates
- **Reliability**: Success/failure rates

Metrics are automatically saved to `logs/metrics.json`.

## Troubleshooting

### Common Issues

**Issue**: "OpenAI API key not found"
- **Solution**: Set the `OPENAI_API_KEY` environment variable

**Issue**: "No documents loaded"
- **Solution**: Check file paths and formats (PDF, TXT, MD only)

**Issue**: "Vector store not found"
- **Solution**: Create a knowledge base first with `create-kb`

**Issue**: Slow performance
- **Solution**: Enable parallel guardrail checks in config.yaml

**Issue**: Low quality scores
- **Solution**: Adjust `min_acceptable_score` or use better models

### Debug Mode

Enable debug logging for detailed information:

```yaml
# config.yaml
logging:
  level: "DEBUG"
```

Or set temporarily:
```bash
export LOG_LEVEL=DEBUG
python scripts/main.py query "question" --kb ./kb
```

## Performance Optimization

### Tips for Better Performance

1. **Use caching**: Enable vector store caching
2. **Parallel processing**: Enable parallel guardrail checks
3. **Optimize chunk sizes**: Adjust based on document types
4. **Model selection**: Use GPT-4o-mini for faster/cheaper operations
5. **Batch queries**: Process multiple questions in one session

### Typical Performance

- **Summarization**: ~5-10 seconds per page
- **Vector store creation**: ~2-5 seconds per document
- **Single query**: ~3-8 seconds (depending on correction attempts)
- **With caching**: 30-50% faster on subsequent runs

## Security Considerations

The system includes:
- Input validation and sanitization
- Path traversal protection
- Suspicious pattern detection
- Configurable rate limiting (optional)
- No storage of sensitive data

## Limitations

- Maximum query length: 1000 characters (configurable)
- Context window: 8000 characters (configurable)
- Supported formats: PDF, TXT, MD only
- Requires internet connection for OpenAI API

## Contributing

This is a self-contained system. To extend:
1. Add new agents in `agents.py`
2. Implement custom loaders in `document_loader.py`
3. Create new pipeline strategies in `rag_pipeline.py`
4. Add metrics in `metrics.py`

## License

This code is provided as-is for educational under MIT License.

## Credits

Built with <3 by Sudhanshu and:
- LangChain for LLM orchestration
- OpenAI for language models
- FAISS for vector search
- Sentence Transformers for embeddings


