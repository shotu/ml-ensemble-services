# Run this at build time to cache the sentiment analysis model
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "/app/model_cache"
os.makedirs(model_dir, exist_ok=True)

model_name = "siebert/sentiment-roberta-large-english"
print(f"Downloading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_dir)

# Save tokenizer + model manually
tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)
print(f"Model cached at {model_dir}") 