import os
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir = "/app/model_cache"
os.makedirs(model_dir, exist_ok=True)

model_name = "vennify/t5-base-grammar-correction"

try:
    print(f"Downloading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
    tokenizer.save_pretrained(model_dir)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_dir)
    model.save_pretrained(model_dir)

    print("Model and tokenizer saved successfully to:", model_dir)

except Exception as e:
    print(f"Failed to download model: {str(e)}")
    sys.exit(1) 