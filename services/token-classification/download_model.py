import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

MODEL_NAME = "iiiorg/piiranha-v1-detect-personal-information"
MODEL_CACHE_DIR = "/app/model_cache"

def download_full_model():
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    print(f"Downloading model and tokenizer for {MODEL_NAME} to {MODEL_CACHE_DIR}...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(MODEL_CACHE_DIR)
    model.save_pretrained(MODEL_CACHE_DIR)
    print("Download and save complete.")

if __name__ == "__main__":
    download_full_model() 