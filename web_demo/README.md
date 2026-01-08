# LogLLM Web Demo

A beautiful web interface for the LogLLM log classification model, built with FastAPI.

## 🚀 Local Development

### 1. Install Dependencies

```bash
cd web_demo
pip install -r requirements.txt
```

### 2. Make sure the model is trained

The model should be in `../models/logllm/` (relative to web_demo directory).

If not, train it first:
```bash
cd ..
python training/label_logs.py
python training/preprocess.py
python training/train.py
```

### 3. Run the Web Server

```bash
python app.py
```

Or with uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### 4. Open in Browser

Navigate to: http://localhost:7860

## 🌐 Deploy to Hugging Face Spaces

### Option 1: Using Hugging Face CLI

1. Install Hugging Face CLI:
```bash
pip install huggingface_hub
```

2. Login to Hugging Face:
```bash
huggingface-cli login
```

3. Create a new Space:
   - Go to https://huggingface.co/new-space
   - Choose "FastAPI" as the SDK
   - Name it (e.g., "logllm-demo")

4. Clone and push:
```bash
cd web_demo
git clone https://huggingface.co/spaces/YOUR_USERNAME/logllm-demo
# Copy app.py and requirements.txt to the cloned directory
cd logllm-demo
git add .
git commit -m "Add LogLLM web demo"
git push
```

### Option 2: Using Git

1. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/new-space
   - Choose "FastAPI" as the SDK
   - Name it (e.g., "logllm-demo")

2. Add the Space as a remote:
```bash
cd web_demo
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/logllm-demo
git add app.py requirements.txt README.md
git commit -m "Initial commit"
git push
```

### Important Notes for Hugging Face Spaces:

1. **Model Path**: The app will look for the model in `models/logllm/`. You'll need to either:
   - Upload the trained model to the Space
   - Use a model from Hugging Face Hub (modify `app.py` to load from hub)
   - Train the model as part of the Space setup

2. **Model Upload**: To upload your trained model:
   ```bash
   huggingface-cli upload YOUR_USERNAME/logllm-demo models/logllm/ --repo-type space
   ```

3. **Alternative**: Load model from Hugging Face Hub:
   - First, upload your model to the Hub:
     ```bash
     huggingface-cli upload YOUR_USERNAME/logllm models/logllm/
     ```
   - Then modify `app.py` to load from hub:
     ```python
     classifier = pipeline(
         "text-classification",
         model="YOUR_USERNAME/logllm",
         return_all_scores=False
     )
     ```

## 📝 API Endpoints

### GET `/`
Returns the HTML interface.

### POST `/classify`
Classifies a log message.

**Request:**
```
Content-Type: application/x-www-form-urlencoded

log_message=Your log message here
```

**Response:**
```json
{
    "label": "ERROR",
    "confidence": 0.982,
    "log_message": "Connection timeout after 30 seconds"
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## 🎨 Features

- ✨ Beautiful, modern UI with gradient design
- 🚀 Fast classification using FastAPI
- 📊 Real-time confidence scores
- 🎯 Example log messages for quick testing
- 📱 Responsive design for mobile and desktop
- ⚡ Async support for better performance

## 🔧 Customization

You can customize the UI by modifying the CSS in the `read_root()` function in `app.py`.

