#!/usr/bin/env python3
"""
Download RoBERTa-large-mnli model for faithfulness evaluation.
This model is specifically designed for Natural Language Inference (NLI) tasks.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_roberta_mnli():
    """Download RoBERTa-large-mnli for faithfulness evaluation"""
    model_name = "roberta-large-mnli"
    cache_dir = "/app/model_cache"
    
    print(f"Starting download of {model_name}...")
    print(f"Target directory: {cache_dir}")
    
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download model and tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Downloading model (this may take several minutes)...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Save to cache directory
        print("Saving tokenizer...")
        tokenizer.save_pretrained(cache_dir)
        
        print("Saving model...")
        model.save_pretrained(cache_dir)
        
        print(f"✅ Successfully downloaded and saved {model_name} to {cache_dir}")
        
        # Verify the download
        print("Verifying download...")
        test_tokenizer = AutoTokenizer.from_pretrained(cache_dir)
        test_model = AutoModelForSequenceClassification.from_pretrained(cache_dir)
        print(f"✅ Verification successful!")
        print(f"Model config: {test_model.config}")
        
    except Exception as e:
        print(f"❌ Error downloading model: {str(e)}")
        sys.exit(1)

def main():
    """Main function to download the model"""
    print("=" * 60)
    print("RoBERTa-large-mnli Model Download Script")
    print("For Faithfulness Evaluation in RAG Systems")
    print("=" * 60)
    
    download_roberta_mnli()
    
    print("=" * 60)
    print("Download completed successfully!")
    print("The service is ready to be started.")
    print("=" * 60)

if __name__ == "__main__":
    main() 