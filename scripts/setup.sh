#!/bin/bash

# Setup script for RAG System
# Run this script to set up the environment

set -e  # Exit on error

echo "=================================="
echo "RAG System Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.8+
required_version="3.11"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "ERROR: Python 3.11 or higher is required"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p .cache
mkdir -p data
echo "✓ Directories created"
echo ""

# Check for API key
echo "Checking for OpenAI API key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: OPENAI_API_KEY not set"
    echo ""
    echo "Please set your OpenAI API key:"
    echo "  export OPENAI_API_KEY='your-key-here'"
    echo ""
    echo "Or add it to a .env file:"
    echo "  echo 'OPENAI_API_KEY=your-key-here' > .env"
    echo ""
else
    echo "✓ OPENAI_API_KEY is set"
fi
echo ""

# Run tests (optional)
read -p "Run tests? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing test dependencies..."
    pip install pytest pytest-asyncio --quiet
    echo "Running tests..."
    pytest test_rag_system.py -v
    echo ""
fi

echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python main.py --help"
echo ""
echo "Quick start:"
echo "  python main.py summarize document.pdf"
echo "  python main.py create-kb document.pdf --save ./kb"
echo "  python main.py interactive --kb ./kb"
echo ""
