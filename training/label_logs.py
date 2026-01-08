import pandas as pd
import re
import os

def label_log(line):
    """Label log messages based on keywords."""
    if re.search(r'error|failed|exception|fatal|critical', line, re.IGNORECASE):
        return "ERROR"
    elif re.search(r'warn|retry|deprecated|timeout|slow', line, re.IGNORECASE):
        return "WARNING"
    return "INFO"

def main():
    # Check if raw log file exists
    raw_log_path = "data/raw/hdfs.log"
    
    if not os.path.exists(raw_log_path):
        print(f"Warning: {raw_log_path} not found. Creating synthetic log data...")
        # Create synthetic log data
        synthetic_logs = [
            "INFO Connection established to server",
            "INFO Service started successfully",
            "INFO User login successful",
            "INFO Database connection pool initialized",
            "INFO Cache cleared successfully",
            "WARNING Retry attempt 2 after timeout",
            "WARNING Deprecated API endpoint used",
            "WARNING High memory usage detected",
            "WARNING Slow query detected: 5.2s",
            "WARNING Connection pool nearly exhausted",
            "ERROR Failed to write to disk",
            "ERROR Database connection failed",
            "ERROR Exception occurred in thread pool",
            "ERROR Fatal error: Out of memory",
            "ERROR Critical: Service unavailable"
        ]
        
        os.makedirs("data/raw", exist_ok=True)
        with open(raw_log_path, "w") as f:
            for log in synthetic_logs:
                f.write(log + "\n")
        print(f"Created synthetic log file at {raw_log_path}")
    
    logs = []
    with open(raw_log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                logs.append({
                    "text": line,
                    "label": label_log(line)
                })
    
    df = pd.DataFrame(logs)
    
    # Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save to CSV
    output_path = "data/processed/logs.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Labeled {len(df)} log entries")
    print(f"Label distribution:")
    print(df["label"].value_counts())
    print(f"\nSaved labeled data to {output_path}")

if __name__ == "__main__":
    main()

