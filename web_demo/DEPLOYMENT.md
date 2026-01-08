# 🚀 Deploying LogLLM Web Demo to Hugging Face Spaces

This guide will help you deploy the LogLLM web demo to Hugging Face Spaces for free.

## Prerequisites

1. A Hugging Face account (sign up at https://huggingface.co/join)
2. Your trained model (from `models/logllm/`)
3. Git installed on your machine

## Step-by-Step Deployment

### Step 1: Upload Your Model to Hugging Face Hub

First, upload your trained model to the Hugging Face Hub so it can be accessed by the Space.

```bash
# Install Hugging Face CLI if not already installed
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Upload your model
cd /Users/shruthisenthilarasu/LogLLM
huggingface-cli upload YOUR_USERNAME/logllm models/logllm/
```

Replace `YOUR_USERNAME` with your Hugging Face username.

**Note**: Make the model public or use a token for private models.

### Step 2: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `logllm-demo` (or your preferred name)
   - **SDK**: Select **FastAPI**
   - **Visibility**: Public (recommended) or Private
3. Click **Create Space**

### Step 3: Clone the Space Repository

```bash
# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/logllm-demo
cd logllm-demo
```

### Step 4: Copy Files to Space

Copy the necessary files from the web_demo directory:

```bash
# From the LogLLM project root
cp web_demo/app_hf.py app.py
cp web_demo/requirements.txt .
```

### Step 5: Update Model Path

Edit `app.py` and update the `HF_MODEL_PATH` variable:

```python
HF_MODEL_PATH = os.getenv("HF_MODEL_PATH", "YOUR_USERNAME/logllm")
```

Replace `YOUR_USERNAME` with your Hugging Face username.

### Step 6: Create README.md for Space

Create a `README.md` file in the Space directory:

```markdown
---
title: LogLLM
emoji: 🔥
colorFrom: purple
colorTo: blue
sdk: fastapi
sdk_version: 0.104.0
app_file: app.py
pinned: false
---

# LogLLM - Log Classification Demo

Fine-tuned DistilBERT model for classifying system log messages into INFO, WARNING, and ERROR categories.

## How to Use

1. Enter a log message in the text area
2. Click "Classify Log"
3. View the predicted log level and confidence score

## Model

The model is available at: https://huggingface.co/YOUR_USERNAME/logllm
```

### Step 7: Commit and Push

```bash
git add app.py requirements.txt README.md
git commit -m "Add LogLLM web demo"
git push
```

### Step 8: Wait for Build

Hugging Face Spaces will automatically:
1. Install dependencies from `requirements.txt`
2. Start the FastAPI application
3. Make it available at `https://huggingface.co/spaces/YOUR_USERNAME/logllm-demo`

The build typically takes 2-5 minutes. You can monitor progress in the Space's "Logs" tab.

## Alternative: Using Environment Variables

If you want to keep the model path configurable, you can set it as an environment variable in the Space settings:

1. Go to your Space settings
2. Navigate to "Variables and secrets"
3. Add a new variable:
   - **Name**: `HF_MODEL_PATH`
   - **Value**: `YOUR_USERNAME/logllm`

Then the app will automatically use this value.

## Troubleshooting

### Model Not Loading

- Verify the model was uploaded correctly: `https://huggingface.co/YOUR_USERNAME/logllm`
- Check that the model path in `app.py` matches your username
- Ensure the model files are present (config.json, pytorch_model.bin, etc.)

### Build Fails

- Check the Space logs for error messages
- Verify `requirements.txt` has all necessary dependencies
- Ensure Python version compatibility (Spaces uses Python 3.10 by default)

### Slow Loading

- First load may be slow as the model downloads
- Consider using a smaller model or optimizing the model size

## Updating the Space

To update your Space after making changes:

```bash
cd logllm-demo
# Make your changes
git add .
git commit -m "Update description"
git push
```

The Space will automatically rebuild.

## Custom Domain (Optional)

Hugging Face Spaces provides a free subdomain. For custom domains, you'll need to use a different hosting service.

## Cost

Hugging Face Spaces is **free** for public Spaces with:
- CPU instances
- Standard hardware
- Automatic scaling

For private Spaces or GPU instances, check Hugging Face pricing.

## Next Steps

- Share your Space link with others
- Add it to your portfolio
- Pin it on your Hugging Face profile
- Integrate it into other projects via API

Enjoy your deployed LogLLM demo! 🎉

