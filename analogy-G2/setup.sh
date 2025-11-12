#!/bin/bash
# Setup script for Analogy Testing Platform
# Supports both conda and venv

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        🔍 Analogy Testing Platform - Setup                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    USE_CONDA=true
    echo "✅ Conda found - will use conda environment"
else
    USE_CONDA=false
    echo "⚠️  Conda not found - will use venv instead"
fi

echo ""

# Check Python version
echo "🔍 Checking Python version..."
if [ "$USE_CONDA" = true ]; then
    python --version 2>/dev/null || python3 --version
else
    python3 --version
fi

if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found! Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python OK"
echo ""

# Setup environment
if [ "$USE_CONDA" = true ]; then
    echo "📦 Setting up conda environment..."
    ENV_NAME="analogy"
    
    # Check if environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "⚠️  Conda environment '${ENV_NAME}' already exists."
        echo "   Activate it with: conda activate ${ENV_NAME}"
        echo ""
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  Removing existing environment..."
            conda env remove -n ${ENV_NAME} -y
            echo "📦 Creating new conda environment..."
            conda create -n ${ENV_NAME} python=3.9 -y
        else
            echo "📦 Using existing environment..."
        fi
    else
        echo "📦 Creating conda environment '${ENV_NAME}'..."
        conda create -n ${ENV_NAME} python=3.9 -y
    fi
    
    echo "✅ Conda environment ready"
    echo ""
    echo "🔄 To activate the environment, run:"
    echo "   conda activate ${ENV_NAME}"
    echo ""
    
else
    # Use venv
    echo "📦 Creating virtual environment..."
    if [ -d "venv" ]; then
        echo "⚠️  Virtual environment already exists. Skipping..."
    else
        python3 -m venv venv
        echo "✅ Virtual environment created"
    fi
    echo ""
    
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    echo ""
fi

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

if [ "$USE_CONDA" = true ]; then
    echo "🔄 Activate the environment:"
    echo "   conda activate analogy"
    echo ""
else
    echo "🔄 Activate the environment:"
    echo "   source venv/bin/activate"
    echo ""
fi

echo "🚀 To test a single analogy, run:"
echo "   python test_cli.py man woman king queen"
echo ""
echo "🚀 To run batch testing, run:"
echo "   python batch_test.py explore_analogies.csv --model word2vec"
echo ""
echo "📖 For more information, see README.md"
echo ""
