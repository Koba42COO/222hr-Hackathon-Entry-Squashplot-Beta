#!/bin/bash
# Development Environment Setup Script

echo "🚀 Setting up development environment..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

# Activate and install dependencies
source .venv/bin/activate
pip install -r dev-requirements.txt

echo "✅ Setup complete!"
echo "Run 'source .venv/bin/activate' to activate the environment"
