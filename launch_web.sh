#!/bin/bash

# Web Interface Launcher Script
# Quick start for RAG System Web Interface

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        RAG System - Web Interface Launcher                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if in correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found!"
    echo "Please run this script from the rag_system directory."
    exit 1
fi

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $python_version"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "⚠️  Virtual environment not found."
    read -p "Would you like to create one? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
        echo "✅ Virtual environment created"
    fi
fi

# Activate venv if it exists
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check/Install dependencies
echo ""
echo "📦 Checking dependencies..."
if ! pip show gradio > /dev/null 2>&1; then
    echo "⚠️  Gradio not installed."
    read -p "Install dependencies now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📥 Installing dependencies..."
        pip install -r requirements.txt --quiet
        echo "✅ Dependencies installed"
    else
        echo "❌ Cannot proceed without dependencies"
        exit 1
    fi
else
    echo "✅ Dependencies OK"
fi

# Check API key
echo ""
echo "🔑 Checking API key..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo ""
    echo "To use this system, you need an OpenAI API key."
    echo ""
    read -p "Would you like to create .env file now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Please enter your OpenAI API key:"
        read -r api_key
        echo "OPENAI_API_KEY=$api_key" > .env
        echo "✅ .env file created"
    else
        echo "⚠️  You can create it manually:"
        echo "   echo 'OPENAI_API_KEY=sk-your-key' > .env"
        echo ""
    fi
elif ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo "⚠️  API key not found in .env file"
    echo "   Please add: OPENAI_API_KEY=sk-your-key"
    echo ""
else
    echo "✅ API key found"
fi

# Check logs directory
if [ ! -d "logs" ]; then
    echo ""
    echo "📁 Creating logs directory..."
    mkdir -p logs
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  🚀 Launching Web Interface                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📱 Interface will open at: http://localhost:7860"
echo "🛑 Press Ctrl+C to stop the server"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Launch the app
python app.py
