import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "ProtectAI/distilroberta-base-rejection-v1"
model_dir = "/app/model_cache"

os.makedirs(model_dir, exist_ok=True)

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
tokenizer.save_pretrained(model_dir)

# Download and save model
model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_dir)
model.save_pretrained(model_dir) 