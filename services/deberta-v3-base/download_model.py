import os
import sys
import traceback

# Constants
MODEL_DIR = "/app/model_cache"
MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"

# Set up model cache directory
os.makedirs(MODEL_DIR, exist_ok=True)

# Set environment variables for model caching
os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR
os.environ["HF_HOME"] = MODEL_DIR

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    
    # Load model and tokenizer
    model_obj = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    model = pipeline("text-classification", model=model_obj, tokenizer=tokenizer)
    
    # Test the model
    test_result = model("Hello, how are you?")
    
    print("Model downloaded and cached successfully")
    
except Exception as e:
    print(f"Failed to download model: {str(e)}")
    traceback.print_exc()
    sys.exit(1) 