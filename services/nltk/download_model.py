import os
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_nltk_resources():
    # Create directory for NLTK data
    nltk_data_dir = "/app/nltk_data"
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    # List of required NLTK packages
    required_packages = [
        'punkt',      # For sentence tokenization
        'stopwords',  # For stopword filtering
        'cmudict'     # For syllable counting
    ]

    logger.info("Downloading NLTK resources...")
    for package in required_packages:
        try:
            logger.info(f"Downloading {package}...")
            nltk.download(package, download_dir=nltk_data_dir, quiet=True)
            logger.info(f"Successfully downloaded {package}")
        except Exception as e:
            logger.error(f"Failed to download {package}: {str(e)}")
            raise

    logger.info(f"NLTK resources downloaded to {nltk_data_dir}")
    logger.info("NLTK resources ready.")

if __name__ == "__main__":
    download_nltk_resources() 