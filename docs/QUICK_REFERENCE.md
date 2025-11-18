# RAG System - Quick Reference Card

## Installation (2 minutes)
```bash
cd rag_system
./setup.sh
export OPENAI_API_KEY='your-key-here'
```

## Common Commands

### Summarize Documents
```bash
python main.py summarize document.pdf --output summary.txt
python main.py summarize doc1.pdf doc2.pdf doc3.pdf --verbose
```

### Create Knowledge Base
```bash
python main.py create-kb documents/*.pdf --save ./my_kb
python main.py create-kb file1.pdf file2.txt file3.md
```

### Query Knowledge Base
```bash
python main.py query "Your question?" --kb ./my_kb
python main.py query "Question?" --kb ./my_kb --context
python main.py query "Question?" --kb ./my_kb --details
```

### Interactive Mode
```bash
python main.py interactive --kb ./my_kb
# Commands in interactive mode:
#   - Type your question
#   - 'context' - toggle context display
#   - 'details' - toggle detailed metrics
#   - 'exit' or 'quit' - quit
```

### View Metrics
```bash
python main.py metrics
```

## Configuration Quick Edit

Edit `config.yaml`:

```yaml
# Use cheaper model for testing
models:
  generate:
    name: "gpt-4o-mini"  # Change from gpt-4o

# Adjust self-correction
rag:
  max_correction_attempts: 2  # Reduce for speed
  min_acceptable_score: 3     # Lower threshold
  
# Speed up retrieval
vector_store:
  top_k: 3  # Retrieve fewer documents
```

## File Structure
```
rag_system/
├── main.py              # Run this
├── config.yaml          # Edit configuration
├── requirements.txt     # Dependencies
└── README.md            # Full documentation
```

## Environment Variables
```bash
export OPENAI_API_KEY='sk-...'
export LOG_LEVEL='DEBUG'  # For troubleshooting
```

## Troubleshooting

### Error: "OpenAI API key not found"
```bash
export OPENAI_API_KEY='your-key-here'
```

### Error: "No module named 'langchain'"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Slow Performance
```yaml
# In config.yaml
rag:
  parallel_guardrail_checks: true
cache:
  enabled: true
```

### View Logs
```bash
tail -f logs/rag_system.log
cat logs/metrics.json
```

## Python API Quick Start

```python
from main import RAGApplication
import asyncio

# Initialize
app = RAGApplication()

# Create knowledge base
app.create_knowledge_base(['doc1.pdf', 'doc2.pdf'])

# Query
result = asyncio.run(app.query("What is AI?"))
print(result['answer'])

# Show metrics
app.show_metrics()
```

## Advanced Usage

### Custom Configuration
```bash
python main.py --config custom_config.yaml summarize doc.pdf
```

### Save/Load Vector Store
```bash
# Save
python main.py create-kb docs/*.pdf --save ./my_store

# Load and use
python main.py query "Question?" --kb ./my_store
```

### Batch Processing
```bash
# Process multiple questions
for q in "Q1" "Q2" "Q3"; do
  python main.py query "$q" --kb ./kb
done
```

## Help Commands
```bash
python main.py --help
python main.py summarize --help
python main.py create-kb --help
python main.py query --help
python main.py interactive --help
```

## Key Features

✅ Multi-page document summarization  
✅ Self-correcting RAG with quality evaluation  
✅ Guardrail filtering for relevance  
✅ Parallel processing for speed  
✅ Comprehensive metrics tracking  
✅ Interactive Q&A mode  
✅ Caching for performance  
✅ Production-ready error handling  

## Quick Tips

1. **Start small**: Test with 1-2 documents first
2. **Use caching**: Significantly speeds up repeated queries
3. **Check metrics**: Monitor performance with `metrics` command
4. **Enable DEBUG**: Set LOG_LEVEL=DEBUG for troubleshooting
5. **Adjust thresholds**: Lower min_score if regeneration too frequent

## Performance

| Operation | Typical Time |
|-----------|--------------|
| Summarize 1 page | ~5-10 seconds |
| Create vector store | ~2-5 sec/doc |
| Single query | ~3-8 seconds |
| With cache | 30-50% faster |

## Support

📖 Full docs: `README.md`  
🏗️ Architecture: `PROJECT_STRUCTURE.md`  
💡 Examples: `example_usage.py`  
🧪 Tests: `test_rag_system.py`  

---

**Quick Start**: `./setup.sh` → Set API key → `python main.py --help`
