"""
FastAPI Web Demo for LogLLM - Hugging Face Spaces Version
This version loads the model from Hugging Face Hub
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import pipeline
import os

# Initialize FastAPI app
app = FastAPI(
    title="LogLLM - Log Classification Demo",
    description="Fine-tuned DistilBERT for classifying log messages into INFO, WARNING, and ERROR",
    version="1.0.0"
)

# Global classifier (loaded once at startup)
classifier = None

# Hugging Face Hub model path (update this with your model path)
HF_MODEL_PATH = os.getenv("HF_MODEL_PATH", "YOUR_USERNAME/logllm")

def load_model():
    """Load the trained model from Hugging Face Hub."""
    global classifier
    
    try:
        # Try loading from Hugging Face Hub first
        print(f"Attempting to load model from Hugging Face Hub: {HF_MODEL_PATH}")
        classifier = pipeline(
            "text-classification",
            model=HF_MODEL_PATH,
            return_all_scores=False
        )
        print(f"Model loaded successfully from Hugging Face Hub")
        return classifier
    except Exception as e:
        print(f"Error loading from Hub: {e}")
        
        # Fallback: Try local path
        model_path = "models/logllm"
        if os.path.exists(model_path):
            try:
                classifier = pipeline(
                    "text-classification",
                    model=model_path,
                    return_all_scores=False
                )
                print(f"Model loaded from local path: {model_path}")
                return classifier
            except Exception as e2:
                print(f"Error loading local model: {e2}")
        
        print("Warning: Model not loaded. Using placeholder.")
        return None

# Load model at startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Label mapping
LABEL_MAP = {
    "LABEL_0": "INFO",
    "LABEL_1": "WARNING",
    "LABEL_2": "ERROR"
}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LogLLM - Log Classification Demo</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            
            .form-group {
                margin-bottom: 25px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 600;
                font-size: 1em;
            }
            
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 1em;
                font-family: 'Courier New', monospace;
                resize: vertical;
                transition: border-color 0.3s;
            }
            
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 10px;
                display: none;
            }
            
            .result.show {
                display: block;
                animation: fadeIn 0.3s;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .result.info {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
            }
            
            .result.warning {
                background: #fff3e0;
                border-left: 4px solid #ff9800;
            }
            
            .result.error {
                background: #ffebee;
                border-left: 4px solid #f44336;
            }
            
            .result-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .label {
                font-size: 1.5em;
                font-weight: 700;
                text-transform: uppercase;
            }
            
            .label.info { color: #2196f3; }
            .label.warning { color: #ff9800; }
            .label.error { color: #f44336; }
            
            .confidence {
                background: white;
                padding: 8px 15px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9em;
            }
            
            .log-text {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-top: 10px;
                font-family: 'Courier New', monospace;
                color: #333;
                word-break: break-word;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
                color: #666;
            }
            
            .loading.show {
                display: block;
            }
            
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .examples {
                margin-top: 30px;
                padding-top: 30px;
                border-top: 1px solid #e0e0e0;
            }
            
            .examples h3 {
                color: #333;
                margin-bottom: 15px;
            }
            
            .example-btn {
                display: inline-block;
                background: #f5f5f5;
                color: #333;
                padding: 8px 15px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 0.9em;
                transition: background 0.2s;
                border: 1px solid #e0e0e0;
            }
            
            .example-btn:hover {
                background: #e0e0e0;
            }
            
            .footer {
                margin-top: 30px;
                text-align: center;
                color: #999;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔥 LogLLM</h1>
            <p class="subtitle">Fine-tuned DistilBERT for Log Classification</p>
            
            <form id="classifyForm">
                <div class="form-group">
                    <label for="logMessage">Enter Log Message:</label>
                    <textarea 
                        id="logMessage" 
                        name="log_message" 
                        rows="4" 
                        placeholder="e.g., Connection timeout after 30 seconds"
                        required
                    ></textarea>
                </div>
                
                <button type="submit" id="submitBtn">Classify Log</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Classifying log message...</p>
            </div>
            
            <div class="result" id="result">
                <div class="result-header">
                    <span class="label" id="label">INFO</span>
                    <span class="confidence" id="confidence">95.0%</span>
                </div>
                <div class="log-text" id="logText"></div>
            </div>
            
            <div class="examples">
                <h3>Try these examples:</h3>
                <button class="example-btn" onclick="setExample('Connection timeout after 30 seconds')">Timeout Example</button>
                <button class="example-btn" onclick="setExample('Disk write failed due to permission error')">Error Example</button>
                <button class="example-btn" onclick="setExample('Service started successfully')">Info Example</button>
                <button class="example-btn" onclick="setExample('WARNING: High CPU usage detected')">Warning Example</button>
                <button class="example-btn" onclick="setExample('User authentication successful')">Success Example</button>
            </div>
            
            <div class="footer">
                <p>Powered by DistilBERT | Fine-tuned for log classification</p>
            </div>
        </div>
        
        <script>
            function setExample(text) {
                document.getElementById('logMessage').value = text;
            }
            
            document.getElementById('classifyForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const logMessage = document.getElementById('logMessage').value;
                const submitBtn = document.getElementById('submitBtn');
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                // Show loading, hide result
                loading.classList.add('show');
                result.classList.remove('show');
                submitBtn.disabled = true;
                
                try {
                    const response = await fetch('/classify', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `log_message=${encodeURIComponent(logMessage)}`
                    });
                    
                    const data = await response.json();
                    
                    // Hide loading, show result
                    loading.classList.remove('show');
                    
                    if (data.error) {
                        alert('Error: ' + data.error);
                        submitBtn.disabled = false;
                        return;
                    }
                    
                    // Update result
                    const label = data.label.toLowerCase();
                    result.className = 'result show ' + label;
                    document.getElementById('label').textContent = data.label;
                    document.getElementById('label').className = 'label ' + label;
                    document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(1) + '%';
                    document.getElementById('logText').textContent = data.log_message;
                    
                } catch (error) {
                    loading.classList.remove('show');
                    alert('Error: ' + error.message);
                } finally {
                    submitBtn.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/classify")
async def classify_log(log_message: str = Form(...)):
    """Classify a log message."""
    if classifier is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model not loaded. Please check model configuration.",
                "label": "ERROR",
                "confidence": 0.0,
                "log_message": log_message
            }
        )
    
    try:
        # Classify the log message
        result = classifier(log_message)[0]
        label = result["label"]
        score = result["score"]
        
        # Map label to readable format
        readable_label = LABEL_MAP.get(label, label)
        
        return {
            "label": readable_label,
            "confidence": score,
            "log_message": log_message
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "label": "ERROR",
                "confidence": 0.0,
                "log_message": log_message
            }
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

