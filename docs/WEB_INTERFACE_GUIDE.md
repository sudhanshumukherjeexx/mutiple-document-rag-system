# RAG System Web Interface - User Guide

## 🌐 Beautiful Web Interface for Document Analysis

A modern, user-friendly web interface built with Gradio for document summarization and intelligent Q&A.

---

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
cd rag_system
pip install -r requirements.txt
```

This will install Gradio and all other required packages.

### Step 2: Set API Key
Make sure your `.env` file has your OpenAI API key:
```env
OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Launch the Interface
```bash
python app.py
```

### Step 4: Open in Browser
The interface will automatically open in your default browser at:
```
http://localhost:7860
```

---

## 📋 Features Overview

### 1️⃣ Document Summarization Tab

**Purpose:** Generate AI-powered summaries from single or multiple documents

**How to Use:**
1. Click "Upload Documents" button
2. Select one or multiple files (PDF, TXT, or MD)
3. Click "✨ Generate Summary"
4. Wait for processing (5-30 seconds depending on document length)
5. View your summary in the output box
6. Use the copy button to copy the summary

**Features:**
- ✅ Upload multiple documents at once
- ✅ Automatic document chunking
- ✅ MapReduce summarization strategy
- ✅ Shows processing statistics
- ✅ Copy summary to clipboard

**Best Practices:**
- Upload clear, well-formatted documents
- Multiple related documents work great together
- Longer documents may take more time
- Check the status panel for progress

---

### 2️⃣ RAG Question Answering Tab

**Purpose:** Ask questions about your documents and get accurate, sourced answers

#### Part A: Knowledge Base Setup

**How to Create a Knowledge Base:**
1. Open the "🔧 RAG Setup" accordion
2. Upload your documents (PDF, TXT, MD)
3. (Optional) Give your knowledge base a name
4. Click "📚 Create Knowledge Base"
5. Wait for indexing to complete
6. You'll see a success message when ready

**Knowledge Base Info:**
- Indexed using FAISS vector store
- Uses sentence-transformers embeddings
- Supports multiple document formats
- Can be cleared and recreated anytime

#### Part B: Chat Interface

**How to Ask Questions:**
1. Make sure knowledge base is created (see Part A)
2. Type your question in the text box
3. Click "Send 🚀" or press Enter
4. View the AI-generated answer
5. Check the quality score and statistics

**Chat Features:**
- ✅ Self-correcting answers (automatically improves low scores)
- ✅ Quality scoring (1-5 stars)
- ✅ Shows number of documents used
- ✅ Displays correction attempts
- ✅ Chat history maintained
- ✅ Copy answers to clipboard

**Example Questions:**
- "What are the main topics discussed?"
- "Can you summarize the key findings?"
- "What are the recommendations?"
- "Explain the methodology used"
- "What are the limitations mentioned?"

**Understanding the Response:**

Each answer includes:
```
[Your Answer Text]

---
📊 Query Statistics:
- Quality Score: 5/5 ⭐        # Higher is better (3+ is good)
- Documents Retrieved: 5        # Total docs found
- Documents Used: 3             # After filtering
- Correction Attempts: 1        # Times regenerated
- Knowledge Base: my_docs       # Which KB was used
```

---

### 3️⃣ System Information Tab

**Purpose:** View system stats, configuration, and documentation

**Features:**
- 📊 Real-time system statistics
- ⚙️ Configuration details
- 📚 Documentation links
- 🔄 Refresh statistics button

**Statistics Shown:**
- Total queries processed
- Success rate
- Average latency
- Average quality score
- Performance metrics

---

## 🎨 Interface Guide

### Main Components

#### Header
- Shows system name and status
- Real-time system status indicator
- Quick access to statistics

#### Navigation Tabs
1. **Document Summarization** - For creating summaries
2. **RAG Question Answering** - For Q&A with documents
3. **System Information** - For stats and info

#### Status Indicators
- ✅ Green checkmark = Success
- ❌ Red X = Error
- 🔄 Loading spinner = Processing

---

## 💡 Tips for Best Results

### Document Summarization

1. **Multiple Documents:**
   - Upload related documents for comprehensive summaries
   - System automatically combines information

2. **Document Quality:**
   - Use clear, well-formatted PDFs
   - Text-heavy documents work best
   - Scanned documents may have lower quality

3. **Processing Time:**
   - Single page: ~5-10 seconds
   - Multiple pages: ~30-60 seconds
   - Be patient with large documents

### RAG Q&A

1. **Knowledge Base Creation:**
   - Upload all relevant documents at once
   - Give meaningful names to your knowledge bases
   - Recreate if you add new documents

2. **Asking Questions:**
   - Be specific in your questions
   - Reference concepts from your documents
   - Try rephrasing if answer isn't satisfactory

3. **Quality Scores:**
   - 5/5 ⭐⭐⭐⭐⭐ Perfect answer
   - 4/5 ⭐⭐⭐⭐ Excellent
   - 3/5 ⭐⭐⭐ Good (acceptable)
   - 2/5 ⭐⭐ Fair (may retry)
   - 1/5 ⭐ Poor (will retry)

4. **Self-Correction:**
   - System automatically regenerates answers scored < 3/5
   - Up to 3 attempts to improve quality
   - Best answer is always returned

---

## 🔧 Configuration

### Default Settings

```yaml
Models:
- Summarization: gpt-4o-mini (fast & cost-effective)
- Q&A Generation: gpt-4o (high quality)
- Embeddings: all-MiniLM-L6-v2 (local, fast)

Performance:
- Parallel processing: Enabled
- Caching: Enabled
- Max correction attempts: 3
- Min acceptable score: 3/5

Retrieval:
- Top-k documents: 5
- Chunk size: 1000 characters
- Chunk overlap: 100 characters
```

### Customization

To change settings, edit `config.yaml`:
```yaml
models:
  generate:
    name: "gpt-4o"  # Change model
    temperature: 0   # Adjust creativity

rag:
  max_correction_attempts: 2  # Reduce attempts
  min_acceptable_score: 4     # Higher threshold
```

---

## 🚨 Troubleshooting

### Interface Won't Start

**Problem:** Error when running `python app.py`

**Solutions:**
```bash
# 1. Check dependencies installed
pip install -r requirements.txt

# 2. Check API key
cat .env
# Should show: OPENAI_API_KEY=sk-...

# 3. Check Python version
python --version
# Should be 3.8 or higher
```

### "API Key Not Found" Error

**Problem:** System can't find OpenAI API key

**Solutions:**
```bash
# 1. Create .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 2. Or export temporarily
export OPENAI_API_KEY='sk-your-key-here'

# 3. Restart the interface
python app.py
```

### Upload Fails

**Problem:** Document upload doesn't work

**Solutions:**
- Check file format (PDF, TXT, MD only)
- Ensure file isn't corrupted
- Try smaller files first
- Check file permissions

### Slow Performance

**Problem:** Interface is slow

**Solutions:**
```yaml
# In config.yaml, enable optimizations:
rag:
  parallel_guardrail_checks: true
cache:
  enabled: true
vector_store:
  top_k: 3  # Retrieve fewer documents
```

### Chat Not Working

**Problem:** Questions don't get answered

**Solutions:**
1. **Check Knowledge Base:**
   - Make sure you created a KB first
   - Look for success message
   - Try recreating if needed

2. **Check Question:**
   - Be specific
   - Reference document content
   - Try simpler questions first

3. **Check Logs:**
   ```bash
   tail -f logs/rag_system.log
   ```

---

## 📊 Monitoring & Statistics

### View Real-Time Stats

Click "📊 View Statistics" button to see:
- Total queries processed
- Success/failure rates
- Average response time
- Quality score distribution
- System performance metrics

### Access Detailed Logs

```bash
# Application logs
tail -f logs/rag_system.log

# Metrics data
cat logs/metrics.json
```

---

## 🌐 Advanced Usage

### Public Access

To share your interface publicly:

```python
# In app.py, change:
app.launch(
    share=True  # Creates public link
)
```

⚠️ **Warning:** This creates a public URL. Don't share sensitive documents!

### Custom Port

```python
# In app.py, change:
app.launch(
    server_port=8080  # Use different port
)
```

### Authentication

Add password protection:

```python
app.launch(
    auth=("username", "password")
)
```

---

## 🎯 Use Cases

### 1. Research Paper Analysis
```
1. Upload multiple research papers
2. Ask: "What are the common findings?"
3. Ask: "Compare the methodologies used"
4. Ask: "What are the key limitations?"
```

### 2. Business Document Q&A
```
1. Upload company policies, reports, manuals
2. Ask: "What is our vacation policy?"
3. Ask: "What were Q3 revenue numbers?"
4. Ask: "Summarize the compliance requirements"
```

### 3. Educational Materials
```
1. Upload textbooks, lecture notes
2. Ask: "Explain concept X in simple terms"
3. Ask: "What are the main formulas?"
4. Ask: "Provide examples of Y"
```

### 4. Legal Document Review
```
1. Upload contracts, agreements
2. Ask: "What are the key terms?"
3. Ask: "What are the obligations?"
4. Ask: "Identify potential risks"
```

---

## 🔐 Security & Privacy

### Data Handling
- ✅ Documents processed in memory
- ✅ No permanent storage
- ✅ Vector store cleared on restart
- ✅ Chat history not saved

### API Key Security
- ✅ Loaded from .env file
- ✅ Not displayed in interface
- ✅ Not logged to files

### Best Practices
- Don't upload sensitive documents on public deployments
- Use authentication for production
- Keep API keys secure
- Monitor usage and costs

---

## 📱 Keyboard Shortcuts

- **Enter** - Submit question (in chat)
- **Ctrl+C** - Stop server (in terminal)
- **F5** - Refresh page

---

## 🆘 Getting Help

### Documentation
- **README.md** - Complete system guide
- **PROJECT_STRUCTURE.md** - Architecture details
- **QUICK_REFERENCE.md** - CLI commands

### Logs
- **logs/rag_system.log** - Application logs
- **logs/metrics.json** - Performance metrics

### Support
- Check troubleshooting section above
- Review error messages in terminal
- Enable DEBUG logging in config.yaml

---

## ✨ What Makes This Interface Special

### Beautiful Design
- 🎨 Modern, professional UI
- 📱 Responsive layout
- 🎯 Intuitive navigation
- ✨ Smooth animations

### Smart Features
- 🤖 Self-correcting AI
- 📊 Real-time statistics
- 🔍 Relevance filtering
- ⭐ Quality scoring

### Production-Ready
- 🔒 Secure by design
- 📈 Performance optimized
- 🛡️ Error handling
- 📝 Comprehensive logging

---

## 🚀 Getting Started Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Set API key in `.env` file
- [ ] Launch interface: `python app.py`
- [ ] Open browser to http://localhost:7860
- [ ] Try document summarization
- [ ] Create a knowledge base
- [ ] Ask some questions
- [ ] Check system statistics
- [ ] Explore all features!

---

**Enjoy your intelligent document analysis system!** 🎉
