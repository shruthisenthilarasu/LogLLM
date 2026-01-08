#!/bin/bash
# Setup script for LogLLM project

echo "🔥 LogLLM Setup Script"
echo "======================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Label logs: python training/label_logs.py"
echo "3. Preprocess data: python training/preprocess.py"
echo "4. Train model: python training/train.py"
echo "5. Evaluate: python training/evaluate.py"
echo "6. Classify logs: python inference/classify_log.py \"Your log message\""

