import os
import sys
import traceback

# Set up model cache directory
model_dir = "/app/model_cache"
os.makedirs(model_dir, exist_ok=True)

# Set environment variables for model caching
os.environ["TRANSFORMERS_CACHE"] = model_dir
os.environ["HF_HOME"] = model_dir

model_name = "sentence-transformers/all-mpnet-base-v2"

def download_spacy_model():
    """Download spaCy transformer model for entity recognition"""
    try:
        import spacy
        from spacy.cli import download
        
        print("Downloading spaCy transformer model...")
        download("en_core_web_trf")
        
        # Test the model
        nlp = spacy.load("en_core_web_trf")
        doc = nlp("Test entity recognition with OpenAI in San Francisco.")
        print(f"✅ spaCy model loaded successfully - found {len(doc.ents)} entities")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to download spaCy model: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        print("Note: spaCy model is optional - service will use regex fallback for entity recognition")
        return False

def test_sentence_transformer():
    """Test sentence-transformers model functionality"""
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading sentence-transformers model: {model_name}")
        model = SentenceTransformer(model_name, cache_folder=model_dir)
        
        # Test encoding to ensure model works correctly
        test_sentences = ["test query", "sample context"]
        embeddings = model.encode(test_sentences)
        
        print(f"✅ Model {model_name} loaded successfully")
        print(f"✅ Generated embeddings shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load sentence-transformers model: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting model download process...")
    
    # Download and test sentence-transformers model
    if not test_sentence_transformer():
        print("❌ Critical error: sentence-transformers model failed to load")
        sys.exit(1)
    
    # Download spaCy model (optional)
    spacy_success = download_spacy_model()
    
    print("\n📋 Download Summary:")
    print(f"✅ sentence-transformers model: SUCCESS")
    print(f"{'✅' if spacy_success else '⚠️'} spaCy model: {'SUCCESS' if spacy_success else 'FAILED (will use fallback)'}")
    
    print("\n🎉 Model download process completed!")
    
    if not spacy_success:
        print("\n⚠️  Note: Some RAG metrics may have reduced accuracy without spaCy transformer model")
        print("   The service will use regex-based entity recognition as fallback") 