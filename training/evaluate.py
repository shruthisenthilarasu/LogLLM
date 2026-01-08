from transformers import pipeline
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    model_path = "models/logllm"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training/train.py first.")
        return
    
    print("Loading model...")
    classifier = pipeline("text-classification", model=model_path, return_all_scores=False)
    
    # Test samples
    samples = [
        "Connection timed out after 30 seconds",
        "Disk write failed due to permission error",
        "Service started successfully",
        "WARNING: High CPU usage detected",
        "ERROR: Database connection failed",
        "INFO: User authentication successful",
        "Retry attempt 3 after network failure",
        "Fatal exception in thread pool"
    ]
    
    print("\n" + "="*60)
    print("Sample Predictions:")
    print("="*60)
    
    for sample in samples:
        result = classifier(sample)[0]
        label = result["label"]
        score = result["score"]
        
        # Map label back to readable format
        label_map = {"LABEL_0": "INFO", "LABEL_1": "WARNING", "LABEL_2": "ERROR"}
        readable_label = label_map.get(label, label)
        
        print(f"\nLog: {sample}")
        print(f"Predicted Level: {readable_label} (confidence: {score:.4f})")
    
    print("\n" + "="*60)
    
    # Optional: Load test set and create confusion matrix
    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        
        dataset = load_from_disk("data/processed/dataset")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Get predictions for test set
        test_texts = [dataset["test"][i]["text"] for i in range(len(dataset["test"]))]
        test_labels = [dataset["test"][i]["label"] for i in range(len(dataset["test"]))]
        
        print(f"\nEvaluating on {len(test_texts)} test samples...")
        predictions = []
        for text in test_texts:
            result = classifier(text)[0]
            label = result["label"]
            # Convert LABEL_0, LABEL_1, LABEL_2 to 0, 1, 2
            pred_label = int(label.split("_")[1])
            predictions.append(pred_label)
        
        # Classification report
        label_names = ["INFO", "WARNING", "ERROR"]
        print("\n" + "="*60)
        print("Classification Report:")
        print("="*60)
        print(classification_report(test_labels, predictions, target_names=label_names))
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=label_names, yticklabels=label_names)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        os.makedirs("models", exist_ok=True)
        plt.savefig("models/confusion_matrix.png", dpi=150, bbox_inches="tight")
        print("\nConfusion matrix saved to models/confusion_matrix.png")
        plt.close()
        
    except Exception as e:
        print(f"\nNote: Could not generate confusion matrix: {e}")
        print("This is optional and requires the test dataset to be available.")

if __name__ == "__main__":
    main()

