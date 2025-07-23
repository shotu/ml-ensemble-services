import os
import sys
import traceback
from sentence_transformers import CrossEncoder

# Constants
MODEL_DIR = "/app/model_cache"
MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

# Set up model cache directory
os.makedirs(MODEL_DIR, exist_ok=True)

# Set environment variables for model caching
os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR
os.environ["HF_HOME"] = MODEL_DIR

try:
    print(f"Loading model: {MODEL_NAME}")
    
    # Initialize the CrossEncoder with the model name
    model = CrossEncoder(MODEL_NAME)
    print("Model loaded successfully")
    
    # Test the model
    model.predict([("test input", "test output")])
    print("Model test successful")
    
except Exception as e:
    print(f"Failed to load model: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 