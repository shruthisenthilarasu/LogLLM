from datasets import Dataset
import pandas as pd
import os

def main():
    # Load labeled data
    csv_path = "data/processed/logs.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run training/label_logs.py first.")
        return
    
    df = pd.read_csv(csv_path)
    
    # Encode labels
    label_map = {"INFO": 0, "WARNING": 1, "ERROR": 2}
    df["label"] = df["label"].map(label_map)
    
    # Check for any NaN labels
    if df["label"].isna().any():
        print("Warning: Some labels could not be mapped. Removing rows with invalid labels.")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
    
    # Create dataset
    dataset = Dataset.from_pandas(df)
    
    # Split into train and test sets
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Save dataset
    output_dir = "data/processed/dataset"
    dataset.save_to_disk(output_dir)
    
    print(f"Preprocessed {len(df)} log entries")
    print(f"Train set: {len(dataset['train'])} samples")
    print(f"Test set: {len(dataset['test'])} samples")
    print(f"\nLabel distribution in train set:")
    train_labels = [dataset["train"][i]["label"] for i in range(len(dataset["train"]))]
    print(pd.Series(train_labels).value_counts().sort_index())
    print(f"\nLabel distribution in test set:")
    test_labels = [dataset["test"][i]["label"] for i in range(len(dataset["test"]))]
    print(pd.Series(test_labels).value_counts().sort_index())
    print(f"\nSaved preprocessed dataset to {output_dir}")

if __name__ == "__main__":
    main()

