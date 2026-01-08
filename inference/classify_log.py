#!/usr/bin/env python3
"""
Log Classification Inference Script

Usage:
    python inference/classify_log.py "Your log message here"
"""

import sys
import os
from transformers import pipeline

def main():
    model_path = "models/logllm"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run training/train.py first to train the model.")
        sys.exit(1)
    
    # Get log message from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python inference/classify_log.py \"Your log message here\"")
        sys.exit(1)
    
    log = " ".join(sys.argv[1:])
    
    # Load classifier
    try:
        classifier = pipeline("text-classification", model=model_path, return_all_scores=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Classify log
    result = classifier(log)[0]
    label = result["label"]
    score = result["score"]
    
    # Map label to readable format
    label_map = {"LABEL_0": "INFO", "LABEL_1": "WARNING", "LABEL_2": "ERROR"}
    readable_label = label_map.get(label, label)
    
    # Print results
    print("="*60)
    print("Log Classification Result")
    print("="*60)
    print(f"Log: {log}")
    print(f"Predicted Level: {readable_label}")
    print(f"Confidence: {score:.4f} ({score*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()

