# Similarity Service

A FastAPI microservice that provides semantic similarity analysis and agentic metrics evaluation using sentence transformers. This service specializes in evaluating AI agent behavior and performance through semantic analysis.

## Overview

The Similarity service provides two main capabilities:

1. **Semantic Similarity Analysis**: Computes pairwise cosine similarities between source and target texts using SBERT models
2. **Agentic Metrics**: Evaluates AI agent behavior including goal accuracy and intent resolution

## API Endpoints

### Semantic Similarity
`POST /predict`

**Request Format** (`SimilarityRequest`):
```json
{
  "sources": ["This is a sentence.", "Another one."],
  "targets": ["Sentence example", "Completely different"],
  "model_name": "all-MiniLM-L6-v2",
  "params": {}
}
```

**Response Format** (`SimilarityResponse`):
```json
{
  "similarities": [[0.65, 0.12], [0.23, 0.89]]
}
```

## Agentic Metrics

The service provides specialized metrics for evaluating AI agent performance in goal achievement and intent understanding.

### 1. Agent Goal Accuracy (`POST /evaluate/agent-goal-accuracy`)

**Purpose**: Evaluates whether the agent accomplished the user's stated or implied goal by comparing the final response to the target goal.

**Algorithm**:
1. **Final Response Analysis**: Extracts the agent's final response from the conversation
2. **Semantic Similarity**: Uses `paraphrase-MiniLM-L6-v2` to compute similarity between final response and target goal
3. **Goal Achievement Scoring**: Measures how well the response addresses the intended goal

**Request Format**:
```json
{
  "conversation_history": [
    "user: I need help booking a flight to Paris",
    "agent: I'll help you book a flight to Paris. Let me search for available options.",
    "agent: I found several flights to Paris. Here are the best options with prices and times."
  ],
  "tool_calls": [
    {"name": "search_flights", "args": {"destination": "Paris"}, "result": "success"}
  ],
  "agent_responses": [
    "I'll help you book a flight to Paris. Let me search for available options.",
    "I found several flights to Paris. Here are the best options with prices and times."
  ],
  "reference_data": {
    "target_goal": "Successfully provide flight booking options to Paris with prices and availability"
  }
}
```

**Response Example**:
```json
{
  "metric_name": "agent_goal_accuracy_evaluation",
  "actual_value": 0.89,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 156.3,
    "final_response": "I found several flights to Paris. Here are the best options with prices and times.",
    "target_goal": "Successfully provide flight booking options to Paris with prices and availability",
    "goal_similarity": 0.89,
    "input_lengths": {
      "conversation_messages": 3,
      "agent_responses": 2,
      "target_goal_length": 89
    },
    "model_name": "paraphrase-MiniLM-L6-v2",
    "evaluation_method": "semantic_goal_comparison"
  }
}
```

### 2. Intent Resolution (`POST /evaluate/intent-resolution`)

**Purpose**: Evaluates whether the agent correctly identified and acted on the user's intent by comparing the first response to the reference intent.

**Algorithm**:
1. **First Response Analysis**: Extracts the agent's initial response to understand intent recognition
2. **Intent Matching**: Uses semantic similarity to compare first response with reference intent
3. **Resolution Scoring**: Measures how well the agent understood and addressed the user's intent

**Request Format**:
```json
{
  "conversation_history": [
    "user: My computer is running very slowly",
    "agent: I understand you're experiencing performance issues. Let me help troubleshoot your computer's speed problems.",
    "agent: First, let's check your system resources and running processes."
  ],
  "tool_calls": [
    {"name": "system_diagnostic", "args": {"type": "performance"}, "result": "success"}
  ],
  "agent_responses": [
    "I understand you're experiencing performance issues. Let me help troubleshoot your computer's speed problems.",
    "First, let's check your system resources and running processes."
  ],
  "reference_data": {
    "reference_intent": "User wants technical support to resolve computer performance issues"
  }
}
```

**Response Example**:
```json
{
  "metric_name": "intent_resolution_evaluation",
  "actual_value": 0.92,
  "actual_value_type": "float",
  "others": {
    "processing_time_ms": 143.8,
    "first_response": "I understand you're experiencing performance issues. Let me help troubleshoot your computer's speed problems.",
    "reference_intent": "User wants technical support to resolve computer performance issues",
    "intent_similarity": 0.92,
    "input_lengths": {
      "conversation_messages": 3,
      "agent_responses": 2,
      "reference_intent_length": 78
    },
    "model_name": "paraphrase-MiniLM-L6-v2",
    "evaluation_method": "semantic_intent_matching"
  }
}
```

## Model Information

This service uses two specialized models:

### Primary Similarity Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose**: General semantic similarity computation
- **Dimensions**: 384
- **Speed**: Optimized for fast inference
- **Use Case**: Pairwise similarity calculations

### Agentic Metrics Model
- **Model**: `sentence-transformers/paraphrase-MiniLM-L6-v2`
- **Purpose**: Specialized for paraphrase and intent understanding
- **Dimensions**: 384
- **Training**: Optimized for paraphrase detection and semantic matching
- **Use Case**: Goal accuracy and intent resolution evaluation

## Health Endpoint

`GET /health`

Returns service health status and model information.

## Local Development

Build and run the service:

```bash
make build
make run
```

Test the endpoints:

```bash
# Test similarity endpoint
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sources":["Hello world"],"targets":["Hi there"]}'

# Test agent goal accuracy
curl -X POST localhost:8000/evaluate/agent-goal-accuracy \
  -H "Content-Type: application/json" \
  -d '{"conversation_history":["user: help me","agent: I can help"],"agent_responses":["I can help"],"reference_data":{"target_goal":"provide assistance"}}'
```

## Use Cases

### Semantic Similarity
- **Content Matching**: Find similar content across documents
- **Question-Answer Systems**: Match questions to relevant answers
- **Duplicate Detection**: Identify similar or duplicate content
- **Recommendation Systems**: Find semantically similar items

### Agentic Metrics
- **Agent Evaluation**: Assess AI agent performance in achieving goals
- **Intent Recognition**: Evaluate how well agents understand user intentions
- **Conversation Quality**: Measure effectiveness of agent responses
- **Training Assessment**: Evaluate agent training effectiveness
- **Production Monitoring**: Monitor agent performance in real-time deployments

## Resource Requirements

- **Memory**: Minimum 4GB (models are smaller than sentence-transformers service)
- **CPU**: At least 2 CPUs for optimal performance
- **Storage**: ~2GB for model downloads
- **Response Time**: Typically 50-200ms per evaluation
