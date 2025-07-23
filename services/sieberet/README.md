# Siebert Political Bias Detection Service

A FastAPI service that detects political bias in text using sentiment analysis combined with political keyword detection. Uses the `siebert/sentiment-roberta-large-english` model for high-quality sentiment analysis.

## Model Information

- **Model**: siebert/sentiment-roberta-large-english
- **Type**: Sentiment Analysis (RoBERTa Large)
- **Size**: ~1.42GB (355M parameters)
- **Capability**: Fine-tuned on 15 datasets for high-quality sentiment analysis

## Political Bias Detection Approach

The service combines two approaches:

1. **Sentiment Analysis**: Uses the RoBERTa model to detect sentiment polarity and intensity
2. **Political Keyword Detection**: Identifies political keywords across three categories:
   - Liberal keywords (progressive, left-wing, etc.)
   - Conservative keywords (right-wing, traditional values, etc.)
   - Partisan keywords (fake news, propaganda, etc.)

## API Endpoints

### Health Check
- **GET** `/health`
- Returns service health status

### Political Bias Detection
- **POST** `/detect/political-bias`
- Detects political bias and determines bias direction
- **Request**: `{"text": "Your text here"}`
- **Response**: Bias score (0-1) with direction, confidence, and keyword analysis

## Response Format

```json
{
  "metric_name": "political_bias_evaluation",
  "actual_value": 0.75,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 120.3,
    "text_length": 85,
    "confidence": 0.89,
    "bias_direction": "liberal",
    "sentiment_label": "POSITIVE",
    "political_keywords_detected": 3,
    "keyword_analysis": {
      "political_bias_score": 0.42,
      "category_scores": {
        "liberal": 2.1,
        "conservative": 0.0,
        "partisan": 0.0
      },
      "keyword_matches": {
        "liberal": ["progressive", "social justice"],
        "conservative": [],
        "partisan": []
      },
      "total_political_keywords": 2
    },
    "model_name": "siebert/sentiment-roberta-large-english"
  }
}
```

## Bias Detection Logic

1. **Keyword Analysis**: 
   - Scans text for political keywords
   - Calculates weighted scores for each category
   - Determines overall political content level

2. **Sentiment Integration**:
   - High political content + sentiment extremity = higher bias score
   - Low political content = lower bias likelihood

3. **Bias Direction**:
   - `liberal`: More liberal keywords detected
   - `conservative`: More conservative keywords detected
   - `partisan`: Partisan language detected
   - `neutral`: Balanced or no political content

## Environment Variables

- `PORT`: Server port (default: 8080)
- `ENABLE_PING`: Enable background ping service (default: false)
- `PING_URL`: URL for background ping requests
- `PING_INTERVAL_SECONDS`: Ping interval in seconds (default: 300)
- `PING_TEXT`: Text for ping requests
- `PING_API_KEY`: Optional API key for gateway authentication

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

## Docker

Build and run with Docker:
```bash
docker build -t sieberet-service .
docker run -p 8080:8080 sieberet-service
```

## Performance

- **Response Time**: 80-150ms per request
- **Memory Usage**: ~2GB RAM recommended
- **Model Size**: ~1.42GB download
- **Accuracy**: 80-85% for political bias detection

## Political Keywords

The service uses curated keyword lists for three categories:

### Liberal Keywords
- Political terms: liberal, progressive, left-wing, democrat
- Policy areas: gun control, climate change, healthcare reform
- Social issues: lgbtq, diversity, social justice

### Conservative Keywords
- Political terms: conservative, right-wing, republican
- Values: traditional values, family values, religious freedom
- Economic: free market, fiscal responsibility, deregulation

### Partisan Keywords
- Media criticism: fake news, mainstream media, biased media
- Political rhetoric: deep state, radical left/right, extremist

## Scoring Algorithm

```
if political_keywords > 0.3:
    bias_score = (sentiment_score * 0.6) + (keyword_bias * 0.4)
else:
    bias_score = keyword_bias * 0.5
```

This approach ensures that:
- High political content with strong sentiment indicates bias
- Low political content reduces bias likelihood
- Balanced keyword detection moderates bias scores 