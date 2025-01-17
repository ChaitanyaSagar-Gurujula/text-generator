import torch
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import PreTrainedModel, AutoConfig
from huggingface_hub import hf_hub_download
import tiktoken
from model import GPT, GPTConfig
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import tempfile

# Get the absolute path to the templates directory
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

MODEL_ID = "sagargurujula/text-generator"

# Initialize FastAPI
app = FastAPI(title="GPT Text Generator")

# Templates with absolute path
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use system's temporary directory
cache_dir = Path(tempfile.gettempdir()) / "model_cache"
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_HOME'] = str(cache_dir)

# Load model from Hugging Face Hub
def load_model():
    try:
        # Download the model file from HF Hub with authentication
        model_path = hf_hub_download(
            repo_id=MODEL_ID,
            filename="best_model.pth",
            cache_dir=cache_dir,
            token=os.environ.get('HF_TOKEN')  # Get token from environment variable
        )
        
        # Initialize our custom GPT model
        model = GPT(GPTConfig())
        
        # Load the state dict
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(device)
        model.eval()
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load the model
model = load_model()

# Define the request body
class TextInput(BaseModel):
    text: str

@app.post("/generate/")
async def generate_text(input: TextInput):
    # Prepare input tensor
    enc = tiktoken.get_encoding('gpt2')
    input_ids = torch.tensor([enc.encode(input.text)]).to(device)
    
    # Generate multiple tokens
    generated_tokens = []
    num_tokens_to_generate = 50  # Generate 20 new tokens
    
    with torch.no_grad():
        current_ids = input_ids
        
        for _ in range(num_tokens_to_generate):
            # Get model predictions
            logits, _ = model(current_ids)
            next_token = logits[0, -1, :].argmax().item()
            generated_tokens.append(next_token)
            
            # Add the new token to our current sequence
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]]).to(device)], dim=1)
    
    # Decode all generated tokens
    generated_text = enc.decode(generated_tokens)
    
    # Return both input and generated text
    return {
        "input_text": input.text,
        "generated_text": generated_text
    }

# Modify the root route to serve the template
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "title": "GPT Text Generator"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)

# To run the app, use the command: uvicorn app:app --reload