#!/bin/bash
# Setup script for Analogy Testing Platform

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        🔍 Analogy Testing Platform - Setup                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found! Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python OK"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ Pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."
echo ""
pip install -r requirements.txt
echo ""
echo "✅ Dependencies installed"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 To test a single analogy, run:"
echo "   python test_cli.py man woman king queen"
echo ""
echo "🚀 To run batch testing, run:"
echo "   python batch_test.py input.csv --model word2vec"
echo ""
echo "📖 For more information, see README.md"
echo ""


