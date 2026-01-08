from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

def tokenize(batch):
    """Tokenize log messages."""
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

def compute_metrics(pred):
    """Compute evaluation metrics."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    # Load preprocessed dataset
    dataset_path = "data/processed/dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Please run training/preprocess.py first.")
        return
    
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)
    
    # Load tokenizer
    print("Loading tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    
    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )
    
    # Training arguments
    output_dir = "models/checkpoints"
    os.makedirs(output_dir, exist_ok=True)
    
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="models/logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        logging_steps=10
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    model_output_dir = "models/logllm"
    os.makedirs(model_output_dir, exist_ok=True)
    
    print(f"Saving model to {model_output_dir}...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate()
    print("\nTest Set Metrics:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTraining complete! Model saved to {model_output_dir}")

if __name__ == "__main__":
    main()

