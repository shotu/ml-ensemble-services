# Rule-Based Detection Service

A FastAPI-based microservice that provides rule-based detection for invisible text characters, insecure output patterns, and plugin execution risks. This service combines multiple security detection capabilities into a single, efficient service.

## Features

### 1. Invisible Text Detection
Detects invisible and control characters in text that could be used for:
- Hidden content injection
- Text manipulation attacks
- Zero-width character abuse
- Non-standard space characters

### 2. Insecure Output Detection
Identifies potentially insecure code patterns including:
- SQL injection vulnerabilities
- XSS (Cross-Site Scripting) patterns
- Insecure function calls (eval, exec, os.system)
- Dangerous library usage
- Insecure deserialization patterns

### 3. Plugin Execution Risk Detection
Analyzes code for dangerous execution patterns that may indicate security vulnerabilities:
- **Dynamic Code Execution**: eval, exec, compile patterns
- **System Command Execution**: os.system, subprocess calls
- **File System Operations**: file operations with user input
- **Network Operations**: urllib, requests with dynamic URLs
- **Import Manipulation**: dynamic imports and module loading
- **AST-Based Analysis**: Advanced syntax tree analysis for complex patterns

### 4. Narrative Flow Analysis
Analyzes narrative continuity and logical flow in text through:
- **Logical Connector Detection**: Identifies and categorizes discourse markers
- **Narrative Break Detection**: Detects temporal, perspective, and logical inconsistencies
- **Implicit Flow Analysis**: Evaluates topic consistency and pronoun usage
- **Connector Variety Scoring**: Measures diversity of logical connections

## API Endpoints

### Health Check
```
GET /health
```
Returns service health status.

### Invisible Text Detection
```
POST /detect/invisible_text
```

**Request Body:**
```json
{
  "text": "Your text to analyze"
}
```

**Response:**
```json
{
  "metric_name": "invisible_text_evaluation",
  "actual_value": 1.0,
  "explanation": "Found 1 invisible or control characters: U+200B",
  "detected_characters": [
    {
      "code": "U+200B",
      "category": "Cf",
      "position": 5,
      "name": "ZERO WIDTH SPACE"
    }
  ],
  "processing_time": 0.0001
}
```

### Insecure Output Detection
```
POST /detect/insecure_output
```

**Request Body:**
```json
{
  "text": "import os; os.system('rm -rf /')"
}
```

**Response:**
```json
{
  "metric_name": "insecure_output_evaluation",
  "actual_value": 1.0,
  "max_score": 1.0,
  "processing_time": 0.0002
}
```

### Plugin Execution Risk Detection
```
POST /detect/plugin-execution-risk
```

**Request Body:**
```json
{
  "text": "import subprocess; subprocess.call(['rm', '-rf', user_input])"
}
```

**Response:**
```json
{
  "metric_name": "plugin_execution_risk_evaluation",
  "actual_value": 0.85,
  "actual_value_type": "float",
  "others": {
    "risk_level": "high",
    "explanation": "High risk plugin execution patterns detected",
    "detected_patterns": [
      {
        "type": "subprocess_call",
        "severity": "high",
        "line": 1,
        "pattern": "subprocess.call with dynamic arguments"
      }
    ],
    "risk_factors": {
      "dynamic_execution": true,
      "system_commands": true,
      "file_operations": false,
      "network_operations": false,
      "import_manipulation": false
    },
    "processing_time_ms": 1.5
  }
}
```

### Narrative Flow Analysis
```
POST /analyze/narrative_flow
```

**Request Body:**
```json
{
  "text": "First, we need to understand the concepts. Therefore, I will explain each step. However, we must consider practical applications.",
  "sentences": [
    "First, we need to understand the concepts.",
    "Therefore, I will explain each step.", 
    "However, we must consider practical applications."
  ]
}
```

**Response:**
```json
{
  "logical_flow_score": 0.7142857142857143,
  "narrative_breaks": [],
  "break_types": {
    "temporal_inconsistency": false,
    "perspective_shift": false,
    "logical_contradiction": false
  },
  "connector_analysis": {
    "causal_connectors": 1,
    "contrast_connectors": 1,
    "additive_connectors": 0,
    "sequential_connectors": 1,
    "reason_connectors": 0,
    "exemplification_connectors": 0,
    "conclusive_connectors": 0,
    "variety_score": 0.42857142857142855,
    "total_connectors": 3
  },
  "processing_time_ms": 2.5
}
```

## Detection Categories

### Invisible Characters
- **Zero-width characters**: U+200B, U+200C, U+200D, U+2060, U+180E, U+FEFF, U+061C
- **Control characters**: Categories Cf, Cc, Co, Cn
- **Problematic spaces**: Non-breaking space, em space, en space, etc.

### Insecure Patterns
- **SQL Injection**: Dynamic query construction patterns
- **XSS**: innerHTML, outerHTML, document.write patterns
- **Insecure Functions**: eval, exec, os.system, subprocess calls
- **Dangerous Libraries**: pickle, yaml, marshal, shelve
- **Deserialization**: pickle.load, yaml.load patterns

### Plugin Execution Risk Patterns
- **Dynamic Code Execution**: 
  - eval(), exec(), compile() functions
  - Code generation and execution patterns
- **System Command Execution**:
  - os.system(), os.popen(), subprocess calls
  - Shell command injection patterns
- **File System Operations**:
  - open(), file operations with user input
  - Path traversal vulnerabilities
- **Network Operations**:
  - urllib, requests with dynamic URLs
  - SSRF (Server-Side Request Forgery) patterns
- **Import Manipulation**:
  - Dynamic imports, __import__() usage
  - Module loading with user input
- **AST Analysis Patterns**:
  - Complex syntax tree analysis
  - Advanced code injection detection

### Narrative Flow Patterns
- **Logical Connectors**: 7 categories of discourse markers
  - **Causal**: therefore, thus, consequently, as a result, hence
  - **Contrast**: however, but, nevertheless, nonetheless, although, yet
  - **Additive**: furthermore, moreover, additionally, also, besides, in addition
  - **Sequential**: first, second, third, finally, next, then, subsequently
  - **Reason**: because, since, due to, owing to, given that
  - **Exemplification**: for example, for instance, such as, namely
  - **Conclusive**: in conclusion, to summarize, overall, in summary

- **Narrative Breaks**: Inconsistencies that disrupt flow
  - **Temporal Inconsistency**: Contradictory time references
  - **Perspective Shift**: Abrupt changes in narrative viewpoint
  - **Logical Contradiction**: Conflicting statements or claims

- **Implicit Flow Indicators**: Subtle continuity markers
  - **Topic Consistency**: Repeated key terms and concepts
  - **Pronoun Usage**: 50+ pronoun types for reference tracking
  - **Semantic Coherence**: Contextual relationship maintenance

## Scoring

### Invisible Text
- **Score 0.0**: No invisible characters detected
- **Score 1.0**: One or more invisible characters found

### Insecure Output
- **Score 0**: No insecure patterns detected
- **Score > 0**: Weighted score based on pattern severity
  - SQL injection patterns: +10 points each
  - XSS patterns: +10 points each
  - Insecure functions: +5 points each
  - Dangerous libraries: +5 points each

### Plugin Execution Risk
- **Score 0.0-0.3**: Low risk - minimal or no dangerous patterns
- **Score 0.3-0.6**: Medium risk - some concerning patterns detected
- **Score 0.6-0.8**: High risk - multiple dangerous patterns
- **Score 0.8-1.0**: Critical risk - severe security vulnerabilities

**Risk Factors:**
- **Dynamic Execution**: Code that executes user input
- **System Commands**: Direct system command execution
- **File Operations**: Unsafe file system access
- **Network Operations**: Potential SSRF vulnerabilities
- **Import Manipulation**: Dynamic module loading

### Narrative Flow Analysis
The narrative flow analysis uses a sophisticated multi-component scoring algorithm:

#### **Core Algorithm (Final Optimized)**
```
Final Score = (Coherence × 35%) + (Similarity × 30%) + (Logical Flow × 25%) - (Breaks Penalty × 10%)
```

#### **Weight Distribution Changes**
**Updated from initial configuration:**
- **Coherence**: 37% → 35% (reduced emphasis, fixed inverted scoring bug)
- **Semantic Similarity**: 37% → 30% (reduced emphasis on similarity)
- **Logical Flow**: 18% → 25% (increased emphasis on connectors)
- **Narrative Breaks Penalty**: 8% → 10% (increased penalty for inconsistencies)

#### **Component Scoring**
1. **Logical Flow Score** (0.0 - 1.0) - 25% Weight:
   - **Base Score**: 0.4 (minimum for multi-sentence text)
   - **Connector Bonus**: +0.15 per connector type + 0.3 base
   - **Variety Bonus**: +0.2 × (unique connector types / 7)
   - **Implicit Flow Bonus**: Up to +0.25 for topic consistency and pronoun usage

2. **Coherence Score** (0.0 - 1.0) - 35% Weight:
   - **Fixed Implementation**: Proper label interpretation for coherence model
   - **NEGATIVE Label**: Score = 1.0 - model_confidence (invert for incoherent text)
   - **POSITIVE Label**: Score = model_confidence (direct for coherent text)
   - **Model**: lenguist/longformer-coherence-1

3. **Semantic Similarity** (0.0 - 1.0) - 30% Weight:
   - **Sentence Transformers**: Using all-MiniLM-L6-v2 model
   - **Pairwise Comparison**: Between consecutive sentences
   - **Topic Consistency**: Weighted by shared vocabulary

4. **Narrative Breaks Detection** - 10% Penalty:
   - **Temporal Inconsistency**: Contradictory time references
   - **Perspective Shift**: Abrupt viewpoint changes (I → you → he/she)
   - **Logical Contradiction**: Conflicting absolute/conditional statements
   - **Penalty**: 0.15 per break (max 0.8 total penalty)

#### **Production Thresholds (Optimized)**
- **Recommended Threshold**: 0.45 (lowered from 0.47)
- **Label Thresholds**: [0.25, 0.38, 0.55] → ["Poor", "Fair", "Good", "Excellent"]
- **Typical Score Range**: 0.24 - 0.52 for realistic content distribution

#### **Performance Results**
- **Pass Rate**: Improved from 12.5% to 37.5% for quality content
- **Score Distribution**: Realistic range with proper discrimination
- **Bug Fixes**: Coherence scoring now properly interprets model labels
- **Calibration**: Thresholds aligned with actual score distributions

#### **Algorithm Features**
- **Production-Grade Pronoun Detection**: 50+ pronoun types
- **Implicit Flow Analysis**: Topic consistency + reference tracking
- **Optimized Weighting**: Balanced emphasis across all components
- **Realistic Scoring**: Calibrated for real-world content quality
- **Bug-Free Implementation**: Fixed inverted coherence scoring

## Usage Examples

### Using curl

**Test invisible text detection:**
```bash
curl -X POST "http://localhost:8000/detect/invisible_text" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello\u200bworld"}'
```

**Test insecure output detection:**
```bash
curl -X POST "http://localhost:8000/detect/insecure_output" \
  -H "Content-Type: application/json" \
  -d '{"text":"import os; os.system(\"ls\")"}'
```

**Test narrative flow analysis:**
```bash
curl -X POST "http://localhost:8000/analyze/narrative_flow" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "First, we need to understand the concepts. Therefore, I will explain each step. However, we must consider practical applications.",
    "sentences": [
      "First, we need to understand the concepts.",
      "Therefore, I will explain each step.",
      "However, we must consider practical applications."
    ]
  }'
```

### Using Python

```python
import requests

# Invisible text detection
response = requests.post(
    "http://localhost:8000/detect/invisible_text",
    json={"text": "Hello\u200bworld"}
)
print(response.json())

# Insecure output detection
response = requests.post(
    "http://localhost:8000/detect/insecure_output",
    json={"text": "eval(user_input)"}
)
print(response.json())

# Narrative flow analysis
response = requests.post(
    "http://localhost:8000/analyze/narrative_flow",
    json={
        "text": "First, we need to understand the concepts. Therefore, I will explain each step. However, we must consider practical applications.",
        "sentences": [
            "First, we need to understand the concepts.",
            "Therefore, I will explain each step.",
            "However, we must consider practical applications."
        ]
    }
)
print(response.json())
```

## Development

### Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

### Docker
```bash
# Build image
docker build -t rule-based-service .

# Run container
docker run -p 8000:8000 rule-based-service
```

### Testing
```bash
# Run the test script
./test_script.sh
```

## Deployment

This service is deployed on Google Cloud Run and integrated with the RAIME ML Services API Gateway.

### Production URL
```
https://rule-based-production-drnc7zg5yq-uc.a.run.app
```

### API Gateway Endpoints
```
https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1/detect/invisible_text
https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1/detect/insecure_output
https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1/analyze/narrative_flow
```

## Performance

- **Response Time**: < 1ms for typical text inputs
- **Memory Usage**: ~50MB base memory
- **Concurrency**: Supports up to 100 concurrent requests
- **Scalability**: Auto-scales from 0 to 10 instances

## Security Considerations

This service analyzes text for security vulnerabilities but does not execute any code. All pattern matching is performed using regular expressions and Unicode character analysis.

## Contributing

1. Follow the existing code style
2. Add tests for new detection patterns
3. Update this README for any new features
4. Ensure all endpoints return consistent response formats 