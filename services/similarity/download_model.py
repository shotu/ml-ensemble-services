# Run this at build time to cache the SBERT model in the Docker image
import os
from sentence_transformers import SentenceTransformer

# Create model directory
model_dir = "/app/model_cache"
os.makedirs(model_dir, exist_ok=True)

print("Downloading all-MiniLM-L6-v2…")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save(model_dir)
print(f"Model cached at {model_dir}")
