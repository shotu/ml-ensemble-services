# Longformer Service to detect coherence

A FastAPI service that computes coherence scores for text using the Longformer model (`lenguist/longformer-coherence-1`).

## API Endpoints

### Health Check
- **GET** `/health`
- Returns service health status

**Response:**
```json
{
  "status": "ok"
}
```

### Detect Coherence
- **POST** `/detect/coherence`
- Computes coherence score for the provided text

**Request Body:**
```json
{
  "text": "Your text to analyze for coherence"
}
```

**Response:**
```json
{
  "metric_name": "coherence_evaluation",
  "actual_value": 0.8542,
  "actual_value_type": "float",
  "others": {
    "raw_label": "LABEL_1",
    "raw_score": 0.8542,
    "inference_time_ms": 245.67
  }
}
```

## Technical Details

- **Model**: `lenguist/longformer-coherence-1` from Hugging Face
- **Input**: Text string
- **Output**: Coherence score (0-1) with metadata
- **Processing Time**: Typically 200-500ms depending on text length

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the model:
```bash
python download_model.py
```

3. Run the service:
```bash
python main.py
```

The service will be available at `http://localhost:8080`

## Model Details
- Model: `lenguist/longformer-coherence-1`
- Input: LLM output text
- Output: Coherence score between 0 and 1
- Threshold: Configurable via request payload

## Environment Variables
- `PORT`: Service port (default: 8080)
- `TRANSFORMERS_CACHE`: Model cache directory
- `HF_HOME`: Hugging Face cache directory 