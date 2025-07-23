import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer

model_dir = "/app/model_cache"
os.makedirs(model_dir, exist_ok=True)

# Use standard BERT model for text similarity and relevance detection
model_name = "bert-base-uncased"  

# Download tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=model_dir)
tokenizer.save_pretrained(model_dir)

# Download model
model = BertForSequenceClassification.from_pretrained(
    model_name,
    cache_dir=model_dir,
    num_labels=2  # Binary classification
)
model.save_pretrained(model_dir)
