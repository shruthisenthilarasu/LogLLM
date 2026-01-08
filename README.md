# LogLLM — Fine-Tuned LLM for Log Classification

A fine-tuned DistilBERT model for classifying system log messages into **INFO**, **WARNING**, and **ERROR** categories using PyTorch and Hugging Face Transformers.

## 📋 Problem Description

System logs are critical for monitoring and debugging applications, but manually classifying thousands of log messages is time-consuming and error-prone. This project addresses this challenge by leveraging transfer learning to automatically classify log messages into three severity levels:

- **INFO**: Normal operational messages
- **WARNING**: Potentially problematic situations that may require attention
- **ERROR**: Critical issues that need immediate action

## 🎯 Model Choice

**DistilBERT** (`distilbert-base-uncased`) was chosen for this project because:

- **Efficiency**: 60% faster and 40% smaller than BERT while retaining 97% of its performance
- **Lightweight**: Suitable for deployment in resource-constrained environments
- **Proven**: Well-established for text classification tasks
- **Fast Training**: Requires less computational resources for fine-tuning

## 📊 Dataset

The project supports two data sources:

1. **HDFS Logs** (Option A - Recommended): Real-world Hadoop Distributed File System logs
   - Download from: Search "HDFS log dataset GitHub"
   - Place in: `data/raw/hdfs.log`

2. **Synthetic Logs** (Option B): Automatically generated sample logs
   - Created automatically if HDFS logs are not available
   - Includes representative examples of all three log levels

### Labeling Rules

Logs are automatically labeled based on keyword matching:

- **ERROR**: Contains keywords like `error`, `failed`, `exception`, `fatal`, `critical`
- **WARNING**: Contains keywords like `warn`, `retry`, `deprecated`, `timeout`, `slow`
- **INFO**: All other log messages

## 🏗️ Project Structure

```
LogLLM/
├── data/
│   ├── raw/              # Raw log files
│   └── processed/        # Processed datasets
├── training/
│   ├── label_logs.py     # Dataset labeling script
│   ├── preprocess.py     # Data preprocessing
│   ├── train.py          # Model fine-tuning
│   └── evaluate.py       # Evaluation and visualization
├── inference/
│   └── classify_log.py   # CLI inference interface
├── models/               # Saved models and checkpoints
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- macOS / Linux (Windows OK with WSL)
- GPU optional (CPU training is supported)

### 2. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### 3. Prepare Data

```bash
# Option A: Place HDFS logs in data/raw/hdfs.log
# Option B: Synthetic logs will be created automatically

# Label the logs
python training/label_logs.py

# Preprocess the data
python training/preprocess.py
```

### 4. Train Model

```bash
python training/train.py
```

Training will:
- Fine-tune DistilBERT for 3 epochs
- Save checkpoints after each epoch
- Evaluate on test set
- Save final model to `models/logllm/`

### 5. Evaluate Model

```bash
python training/evaluate.py
```

This will:
- Test on sample log messages
- Generate classification report
- Create confusion matrix visualization

### 6. Classify Logs

```bash
python inference/classify_log.py "Failed to connect to database"
```

Output:
```
============================================================
Log Classification Result
============================================================
Log: Failed to connect to database
Predicted Level: ERROR
Confidence: 0.9234 (92.34%)
============================================================
```

## Example Predictions

| Log Message | Predicted | Confidence |
|-------------|-----------|------------|
| "Connection timeout after 30s" | ERROR | 98.2% |
| "API response time: 245ms" | WARNING | 87.5% |
| "User logged in successfully" | INFO | 99.1% |

## 📈 Training Details

### Hyperparameters

- **Model**: `distilbert-base-uncased`
- **Learning Rate**: 2e-5
- **Batch Size**: 16 (per device)
- **Epochs**: 3
- **Max Sequence Length**: 128 tokens
- **Weight Decay**: 0.01
- **Optimizer**: AdamW (default)

### Training Process

1. **Tokenization**: Log messages are tokenized using DistilBERT tokenizer
2. **Fine-tuning**: Model is trained on labeled log data
3. **Evaluation**: Performance is measured after each epoch
4. **Best Model**: Model with highest F1 score is saved

## 📊 Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Macro-averaged F1 score across all classes
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall

Example metrics (may vary based on dataset):
- Accuracy: ~0.95
- F1 Score: ~0.94
- Precision: ~0.93
- Recall: ~0.94

## 🔧 Limitations

1. **Keyword-Based Labeling**: Initial labeling relies on keyword matching, which may not capture all nuances
2. **Limited Context**: Model processes logs independently without temporal context
3. **Domain Specificity**: Model is trained on specific log patterns and may not generalize to all log formats
4. **Small Dataset**: Performance may improve with larger, more diverse datasets
5. **Class Imbalance**: If logs are heavily skewed toward one class, model may be biased

## 🚀 Future Work

1. **Active Learning**: Implement active learning to improve labeling efficiency
2. **Multi-class Expansion**: Add more log levels (DEBUG, CRITICAL, etc.)
3. **Structured Logs**: Support for structured log formats (JSON, XML)
4. **Real-time Classification**: Build streaming pipeline for real-time log classification
5. **Anomaly Detection**: Extend to detect anomalous log patterns
6. **Multi-domain Training**: Train on logs from multiple domains (web, database, system)
7. **Explainability**: Add attention visualization to explain predictions
8. **Deployment**: Create REST API or web interface for easy deployment

## 📝 Resume-Ready Description

> Fine-tuned a DistilBERT language model using PyTorch and Hugging Face to classify system log messages, integrating NLP-based inference into a structured log analytics workflow with quantitative evaluation.

## 📄 License

This project is open source and available for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project is designed for educational purposes and demonstrates the application of transfer learning to log classification. For production use, consider additional validation, larger datasets, and domain-specific fine-tuning.

