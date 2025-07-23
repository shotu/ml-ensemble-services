import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ["TRANSFORMERS_CACHE"] = "/tmp/cache"
os.makedirs("/tmp/cache", exist_ok=True)

print("Downloading Vectara hallucination evaluation model and FLAN-T5 tokenizer...")
AutoModelForSequenceClassification.from_pretrained(
    "vectara/hallucination_evaluation_model",
    trust_remote_code=True,
    cache_dir="/tmp/cache"
)
AutoTokenizer.from_pretrained("google/flan-t5-xxl", cache_dir="/tmp/cache")
print("Download complete.") 