#!/bin/bash

# Set variables
PROJECT_ID="atri-dev"
REGION="us-central1"
API_ID="raime-ml-api"
ENV=${1:-"prod"}  # Default to prod if not specified
TIMESTAMP=$(date +%Y%m%d-%H%M%S)  # Using hyphen instead of underscore
CONFIG_ID="raime-ml-config-${ENV}-${TIMESTAMP}"
GATEWAY_ID="raime-ml-gateway-${ENV}"

# Set environment-specific variables
if [ "$ENV" = "prod" ]; then
    BLEU_SCORE_URL="https://nltk-production-drnc7zg5yq-uc.a.run.app"
    ROUGE_SCORE_URL="https://nltk-production-drnc7zg5yq-uc.a.run.app"
    BERT_SCORE_URL="https://bert-score-production-drnc7zg5yq-uc.a.run.app"
    ROBERTA_URL="https://roberta-production-drnc7zg5yq-uc.a.run.app"
    ROBERTA_MNLI_URL="https://roberta-mnli-production-drnc7zg5yq-uc.a.run.app"
    COMPRESSION_SCORE_URL="https://nltk-production-drnc7zg5yq-uc.a.run.app"
    COSINE_SIMILARITY_URL="https://nltk-production-drnc7zg5yq-uc.a.run.app"
    SIMILARITY_URL="https://similarity-production-1018792104281.us-central1.run.app"
    LONGFORMER_URL="https://longformer-production-drnc7zg5yq-uc.a.run.app"
    BERT_DETOXIFY_URL="https://bert-detoxify-production-drnc7zg5yq-uc.a.run.app"
    DISTILBERT_NSFW_URL="https://distilbert-nsfw-text-classifier-production-drnc7zg5yq-uc.a.run.app"
    DISTILROBERTA_REJECTION_URL="https://distilroberta-base-rejection-v1-production-drnc7zg5yq-uc.a.run.app"
    BERT_BASE_UNCASED_URL="https://bert-base-uncased-production-drnc7zg5yq-uc.a.run.app"
    BERTWEET_BASE_URL="https://bertweet-base-production-drnc7zg5yq-uc.a.run.app"
    DEBERTA_V3_LARGE_URL="https://deberta-v3-large-production-drnc7zg5yq-uc.a.run.app"
    SENTENCE_TRANSFORMERS_URL="https://sentence-transformers-production-drnc7zg5yq-uc.a.run.app"
    FLAN_T5_URL="https://flan-t5-production-drnc7zg5yq-uc.a.run.app"
    TOKEN_CLASSIFICATION_URL="https://token-classification-production-drnc7zg5yq-uc.a.run.app"
    DEBERTA_V3_BASE_URL="https://deberta-v3-base-production-drnc7zg5yq-uc.a.run.app"
    T5_BASE_URL="https://t5-base-production-drnc7zg5yq-uc.a.run.app"
    NLTK_URL="https://nltk-production-drnc7zg5yq-uc.a.run.app"
    RULE_BASED_URL="https://rule-based-production-drnc7zg5yq-uc.a.run.app"
    MORITZLAUREL_URL="https://moritzlaurel-production-drnc7zg5yq-uc.a.run.app"
    API_CONFIG="api-gateway-config.yaml"
    API_CONFIG_ID="raime-ml-api-config"
    API_GATEWAY_ID="raime-ml-api-gateway"
else
    BLEU_SCORE_URL="https://nltk-staging-drnc7zg5yq-uc.a.run.app"
    ROUGE_SCORE_URL="https://nltk-staging-drnc7zg5yq-uc.a.run.app"
    BERT_SCORE_URL="https://bert-score-staging-drnc7zg5yq-uc.a.run.app"
    ROBERTA_URL="https://roberta-staging-drnc7zg5yq-uc.a.run.app"
    ROBERTA_MNLI_URL="https://roberta-mnli-staging-drnc7zg5yq-uc.a.run.app"
    COMPRESSION_SCORE_URL="https://nltk-staging-drnc7zg5yq-uc.a.run.app"
    COSINE_SIMILARITY_URL="https://nltk-staging-drnc7zg5yq-uc.a.run.app"
    SIMILARITY_URL="https://similarity-staging-1018792104281.us-central1.run.app"
    LONGFORMER_URL="https://longformer-staging-drnc7zg5yq-uc.a.run.app"
    BERT_DETOXIFY_URL="https://bert-detoxify-staging-drnc7zg5yq-uc.a.run.app"
    DISTILBERT_NSFW_URL="https://distilbert-nsfw-text-classifier-staging-drnc7zg5yq-uc.a.run.app"
    DISTILROBERTA_REJECTION_URL="https://distilroberta-base-rejection-v1-staging-drnc7zg5yq-uc.a.run.app"
    BERT_BASE_UNCASED_URL="https://bert-base-uncased-staging-drnc7zg5yq-uc.a.run.app"
    BERTWEET_BASE_URL="https://bertweet-base-staging-drnc7zg5yq-uc.a.run.app"
    DEBERTA_V3_LARGE_URL="https://deberta-v3-large-staging-drnc7zg5yq-uc.a.run.app"
    SENTENCE_TRANSFORMERS_URL="https://sentence-transformers-staging-drnc7zg5yq-uc.a.run.app"
    FLAN_T5_URL="https://flan-t5-staging-drnc7zg5yq-uc.a.run.app"
    TOKEN_CLASSIFICATION_URL="https://token-classification-staging-drnc7zg5yq-uc.a.run.app"
    DEBERTA_V3_BASE_URL="https://deberta-v3-base-staging-drnc7zg5yq-uc.a.run.app"
    T5_BASE_URL="https://t5-base-staging-drnc7zg5yq-uc.a.run.app"
    NLTK_URL="https://nltk-staging-drnc7zg5yq-uc.a.run.app"
    RULE_BASED_URL="https://rule-based-staging-drnc7zg5yq-uc.a.run.app"
    MORITZLAUREL_URL="https://moritzlaurel-staging-drnc7zg5yq-uc.a.run.app"
    API_CONFIG="api-gateway-config-staging.yaml"
    API_CONFIG_ID="raime-ml-api-config-staging"
    API_GATEWAY_ID="raime-ml-api-gateway-staging"
fi

# Update the API config file with the correct URLs
sed -i '' "s|https://bleu-score-production-drnc7zg5yq-uc.a.run.app|$BLEU_SCORE_URL|g" "$API_CONFIG"
sed -i '' "s|https://rouge-score-production-drnc7zg5yq-uc.a.run.app|$ROUGE_SCORE_URL|g" "$API_CONFIG"
sed -i '' "s|https://bert-score-production-drnc7zg5yq-uc.a.run.app|$BERT_SCORE_URL|g" "$API_CONFIG"
sed -i '' "s|https://roberta-production-drnc7zg5yq-uc.a.run.app|$ROBERTA_URL|g" "$API_CONFIG"
sed -i '' "s|https://roberta-mnli-production-drnc7zg5yq-uc.a.run.app|$ROBERTA_MNLI_URL|g" "$API_CONFIG"
sed -i '' "s|https://compression-score-production-drnc7zg5yq-uc.a.run.app|$COMPRESSION_SCORE_URL|g" "$API_CONFIG"
sed -i '' "s|https://cosine-similarity-production-drnc7zg5yq-uc.a.run.app|$COSINE_SIMILARITY_URL|g" "$API_CONFIG"
sed -i '' "s|https://similarity-production-1018792104281.us-central1.run.app|$SIMILARITY_URL|g" "$API_CONFIG"
sed -i '' "s|https://longformer-production-drnc7zg5yq-uc.a.run.app|$LONGFORMER_URL|g" "$API_CONFIG"
sed -i '' "s|https://bert-detoxify-production-drnc7zg5yq-uc.a.run.app|$BERT_DETOXIFY_URL|g" "$API_CONFIG"
sed -i '' "s|https://distilbert-nsfw-text-classifier-production-drnc7zg5yq-uc.a.run.app|$DISTILBERT_NSFW_URL|g" "$API_CONFIG"
sed -i '' "s|https://distilroberta-base-rejection-v1-production-drnc7zg5yq-uc.a.run.app|$DISTILROBERTA_REJECTION_URL|g" "$API_CONFIG"
sed -i '' "s|https://bert-base-uncased-production-drnc7zg5yq-uc.a.run.app|$BERT_BASE_UNCASED_URL|g" "$API_CONFIG"
sed -i '' "s|https://bertweet-base-production-drnc7zg5yq-uc.a.run.app|$BERTWEET_BASE_URL|g" "$API_CONFIG"
sed -i '' "s|https://answer-relevance-production-drnc7zg5yq-uc.a.run.app|$DEBERTA_V3_LARGE_URL|g" "$API_CONFIG"
sed -i '' "s|https://sentence-transformers-production-drnc7zg5yq-uc.a.run.app|$SENTENCE_TRANSFORMERS_URL|g" "$API_CONFIG"
sed -i '' "s|https://flan-t5-production-drnc7zg5yq-uc.a.run.app|$FLAN_T5_URL|g" "$API_CONFIG"
sed -i '' "s|https://token-classification-production-drnc7zg5yq-uc.a.run.app|$TOKEN_CLASSIFICATION_URL|g" "$API_CONFIG"
sed -i '' "s|https://deberta-v3-base-production-drnc7zg5yq-uc.a.run.app|$DEBERTA_V3_BASE_URL|g" "$API_CONFIG"
sed -i '' "s|https://t5-base-production-drnc7zg5yq-uc.a.run.app|$T5_BASE_URL|g" "$API_CONFIG"
sed -i '' "s|https://nltk-production-drnc7zg5yq-uc.a.run.app|$NLTK_URL|g" "$API_CONFIG"
sed -i '' "s|https://rule-based-production-drnc7zg5yq-uc.a.run.app|$RULE_BASED_URL|g" "$API_CONFIG"
sed -i '' "s|https://moritzlaurel-production-drnc7zg5yq-uc.a.run.app|$MORITZLAUREL_URL|g" "$API_CONFIG"

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Function to backup current config
backup_current_config() {
    local current_config
    current_config=$(gcloud api-gateway api-configs list \
        --api=$API_ID \
        --project=$PROJECT_ID \
        --format="value(id)" \
        --limit=1) || handle_error "Failed to get current config"
    
    if [ ! -z "$current_config" ]; then
        echo "Backing up current config: $current_config"
        gcloud api-gateway api-configs describe $current_config \
            --api=$API_ID \
            --project=$PROJECT_ID \
            --format=json > "backup_config_${TIMESTAMP}.json" || handle_error "Failed to backup config"
    fi
}

# Function to rollback
rollback() {
    local backup_file=$1
    if [ -f "$backup_file" ]; then
        echo "Rolling back to previous configuration..."
        gcloud api-gateway api-configs create "rollback-${TIMESTAMP}" \
            --api=$API_ID \
            --project=$PROJECT_ID \
            --openapi-spec="$backup_file" \
            --display-name="Rollback Config" || handle_error "Failed to create rollback config"
        
        gcloud api-gateway gateways update $GATEWAY_ID \
            --api=$API_ID \
            --api-config="rollback-${TIMESTAMP}" \
            --project=$PROJECT_ID \
            --location=$REGION || handle_error "Failed to update gateway with rollback config"
        
        echo "Rollback completed successfully"
    else
        echo "No backup file found for rollback"
    fi
}

# Install required components
echo "Installing required components..."
gcloud components install alpha --quiet || handle_error "Failed to install alpha components"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable apigateway.googleapis.com --project=$PROJECT_ID || handle_error "Failed to enable API Gateway API"
gcloud services enable firebase.googleapis.com --project=$PROJECT_ID || handle_error "Failed to enable Firebase API"
gcloud services enable monitoring.googleapis.com --project=$PROJECT_ID || handle_error "Failed to enable Monitoring API"

# Create or get API
echo "Setting up API..."
if ! gcloud api-gateway apis describe $API_ID --project=$PROJECT_ID &>/dev/null; then
    gcloud api-gateway apis create $API_ID \
        --project=$PROJECT_ID \
        --display-name="RAIME ML Services API" || handle_error "Failed to create API"
fi

# Backup current config
backup_current_config

# Create API Config
echo "Creating API Config..."
gcloud api-gateway api-configs create $CONFIG_ID \
    --api=$API_ID \
    --project=$PROJECT_ID \
    --openapi-spec=$API_CONFIG \
    --display-name="RAIME ML Services API Config ${ENV} ${TIMESTAMP}" || handle_error "Failed to create API config"

# Create or update Gateway
echo "Setting up Gateway..."
if gcloud api-gateway gateways describe $GATEWAY_ID --project=$PROJECT_ID --location=$REGION &>/dev/null; then
    gcloud api-gateway gateways update $GATEWAY_ID \
        --api=$API_ID \
        --api-config=$CONFIG_ID \
        --project=$PROJECT_ID \
        --location=$REGION || handle_error "Failed to update gateway"
else
    gcloud api-gateway gateways create $GATEWAY_ID \
        --api=$API_ID \
        --api-config=$CONFIG_ID \
        --project=$PROJECT_ID \
        --location=$REGION \
        --display-name="RAIME ML Services Gateway ${ENV}" || handle_error "Failed to create gateway"
fi

# Get Gateway URL
GATEWAY_URL=$(gcloud api-gateway gateways describe $GATEWAY_ID \
    --project=$PROJECT_ID \
    --location=$REGION \
    --format="value(defaultHostname)") || handle_error "Failed to get gateway URL"

# Set up monitoring
echo "Setting up monitoring..."

# Create uptime check for similarity service
gcloud monitoring uptime create http similarity-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/text-similarity" \
    --project=$PROJECT_ID \
    --display-name="Similarity API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"sources":["test"],"targets":["test"]}' || echo "Warning: Failed to create similarity uptime check"

# Create uptime check for roberta bias service
gcloud monitoring uptime create http roberta-bias-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/bias" \
    --project=$PROJECT_ID \
    --display-name="RoBERTa Bias API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create roberta bias uptime check"

# Create uptime check for BLEU score service
gcloud monitoring uptime create http bleu-score-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/bleu-score" \
    --project=$PROJECT_ID \
    --display-name="BLEU Score API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"references":["test"],"predictions":["test"]}' || echo "Warning: Failed to create BLEU score uptime check"

# Create uptime check for longformer service
gcloud monitoring uptime create http longformer-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/coherence" \
    --project=$PROJECT_ID \
    --display-name="Longformer API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create longformer uptime check"

# Create uptime check for compression score service
gcloud monitoring uptime create http compression-score-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/compression-score" \
    --project=$PROJECT_ID \
    --display-name="Compression Score API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"references":["test"],"predictions":["test"]}' || echo "Warning: Failed to create compression score uptime check"

# Create uptime check for cosine similarity service
gcloud monitoring uptime create http cosine-similarity-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/cosine-similarity" \
    --project=$PROJECT_ID \
    --display-name="Cosine Similarity API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"references":["test"],"predictions":["test"]}' || echo "Warning: Failed to create cosine similarity uptime check"

# Create uptime check for bert-detoxify service
gcloud monitoring uptime create http bert-detoxify-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/toxicity" \
    --project=$PROJECT_ID \
    --display-name="BERT Detoxify API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create bert-detoxify uptime check"

# Create uptime check for distilbert-nsfw-text-classifier service
gcloud monitoring uptime create http distilbert-nsfw-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/nsfw" \
    --project=$PROJECT_ID \
    --display-name="DistilBERT NSFW API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create distilbert-nsfw uptime check"

# Create uptime check for distilroberta-rejection service
gcloud monitoring uptime create http distilroberta-rejection-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/refusal" \
    --project=$PROJECT_ID \
    --display-name="DistilRoBERTa Rejection API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create distilroberta-rejection uptime check"

# Create uptime check for bert-base-uncased service
gcloud monitoring uptime create http bert-base-uncased-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/creativity" \
    --project=$PROJECT_ID \
    --display-name="BERT Base Uncased API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"context":"test","response":"test"}' || echo "Warning: Failed to create bert-base-uncased uptime check"

# Create uptime check for deberta-v3-large service
gcloud monitoring uptime create http deberta-v3-large-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/answer_relevance" \
    --project=$PROJECT_ID \
    --display-name="DeBERTa v3 Large API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"input_text":"test input","output_text":"test output"}' || echo "Warning: Failed to create deberta-v3-large uptime check"

# Create uptime check for sentence transformers service
gcloud monitoring uptime create http sentence-transformers-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/similarity" \
    --project=$PROJECT_ID \
    --display-name="Sentence Transformers API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text1":"test","text2":"test"}' || echo "Warning: Failed to create sentence transformers uptime check"

# Create uptime check for data leakage service
gcloud monitoring uptime create http data-leakage-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/data_leakage" \
    --project=$PROJECT_ID \
    --display-name="Data Leakage API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create data leakage uptime check"

# Create notification channel for alerts
NOTIFICATION_CHANNEL=$(gcloud beta monitoring channels create \
    --display-name="API Team Alerts" \
    --type=email \
    --channel-labels=email_address=api-team@atri.ai \
    --project=$PROJECT_ID \
    --format="value(name)") || echo "Warning: Failed to create notification channel"

# Create alert policy for 5xx errors
gcloud alpha monitoring policies create \
    --project=$PROJECT_ID \
    --display-name="API Gateway 5xx Errors" \
    --condition-display-name="API Gateway 5xx Errors" \
    --condition-filter='resource.type="api_gateway" AND metric.type="apigateway.googleapis.com/request_count" AND metric.labels.response_code_class="5xx"' \
    --condition-threshold-value=1 \
    --condition-threshold-duration=300s \
    --notification-channels="$NOTIFICATION_CHANNEL" || echo "Warning: Failed to create alert policy"

# Create alert policy for high latency
gcloud alpha monitoring policies create \
    --project=$PROJECT_ID \
    --display-name="API Gateway High Latency" \
    --condition-display-name="API Gateway High Latency" \
    --condition-filter='resource.type="api_gateway" AND metric.type="apigateway.googleapis.com/request_latencies"' \
    --condition-threshold-value=5000 \
    --condition-threshold-duration=300s \
    --notification-channels="$NOTIFICATION_CHANNEL" || echo "Warning: Failed to create high latency alert policy"

# Create uptime check for NLTK diversity service
gcloud monitoring uptime create http nltk-diversity-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/diversity" \
    --project=$PROJECT_ID \
    --display-name="NLTK Diversity API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create NLTK diversity uptime check"

# Create uptime check for NLTK readability service
gcloud monitoring uptime create http nltk-readability-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/readability" \
    --project=$PROJECT_ID \
    --display-name="NLTK Readability API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create NLTK readability uptime check"

# Create uptime check for NLTK clarity service
gcloud monitoring uptime create http nltk-clarity-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/clarity" \
    --project=$PROJECT_ID \
    --display-name="NLTK Clarity API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create NLTK clarity uptime check"

# Create uptime check for rule-based invisible text service
gcloud monitoring uptime create http rule-based-invisible-text-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/invisible_text" \
    --project=$PROJECT_ID \
    --display-name="Rule-Based Invisible Text API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create rule-based invisible text uptime check"

# Create uptime check for rule-based insecure output service
gcloud monitoring uptime create http rule-based-insecure-output-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/insecure_output" \
    --project=$PROJECT_ID \
    --display-name="Rule-Based Insecure Output API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create rule-based insecure output uptime check"

# Create uptime check for deberta-v3-base prompt injection service
gcloud monitoring uptime create http deberta-v3-base-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/prompt_injection" \
    --project=$PROJECT_ID \
    --display-name="DeBERTa v3 Base API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create deberta-v3-base uptime check"

# Create uptime check for fuzzy score service
gcloud monitoring uptime create http fuzzy-score-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/fuzzy-score" \
    --project=$PROJECT_ID \
    --display-name="Fuzzy Score API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"references":["test"],"predictions":["test"]}' || echo "Warning: Failed to create fuzzy score uptime check"

# Create uptime check for ROUGE score service
gcloud monitoring uptime create http rouge-score-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/rouge-score" \
    --project=$PROJECT_ID \
    --display-name="ROUGE Score API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"references":["test"],"predictions":["test"]}' || echo "Warning: Failed to create ROUGE score uptime check"

# Create uptime check for METEOR score service
gcloud monitoring uptime create http meteor-score-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/meteor-score" \
    --project=$PROJECT_ID \
    --display-name="METEOR Score API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"references":["test"],"predictions":["test"]}' || echo "Warning: Failed to create METEOR score uptime check"

# Create uptime check for sentence tokenization service
gcloud monitoring uptime create http sentence-tokenize-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/tokenize/sentences" \
    --project=$PROJECT_ID \
    --display-name="Sentence Tokenization API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"This is a test sentence."}' || echo "Warning: Failed to create sentence tokenization uptime check"

# Create uptime check for narrative flow analysis service
gcloud monitoring uptime create http narrative-flow-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/analyze/narrative_flow" \
    --project=$PROJECT_ID \
    --display-name="Narrative Flow Analysis API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"This is a test sentence. However, this is another sentence.","sentences":["This is a test sentence.","However, this is another sentence."]}' || echo "Warning: Failed to create narrative flow analysis uptime check"

# Create uptime check for faithfulness service
gcloud monitoring uptime create http faithfulness-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/faithfulness" \
    --project=$PROJECT_ID \
    --display-name="Faithfulness API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"llm_input_query":"test query","llm_input_context":"test context","llm_output":"test output"}' || echo "Warning: Failed to create faithfulness uptime check"
# Create uptime check for agentic metrics endpoints (rule-based)
gcloud monitoring uptime create http rule-based-tool-call-accuracy-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/tool-call-accuracy" \
    --project=$PROJECT_ID \
    --display-name="Rule-Based Tool Call Accuracy API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"conversation_history":["user: test"],"tool_calls":[],"agent_responses":["test"],"reference_data":{"expected_tool_calls":[]}}' || echo "Warning: Failed to create rule-based tool-call-accuracy uptime check"

gcloud monitoring uptime create http rule-based-plan-coherence-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/plan-coherence" \
    --project=$PROJECT_ID \
    --display-name="Rule-Based Plan Coherence API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"conversation_history":["user: test"],"tool_calls":[],"agent_responses":["test"],"reference_data":{}}' || echo "Warning: Failed to create rule-based plan-coherence uptime check"

gcloud monitoring uptime create http rule-based-plan-optimality-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/plan-optimality" \
    --project=$PROJECT_ID \
    --display-name="Rule-Based Plan Optimality API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"conversation_history":["user: test"],"tool_calls":[],"agent_responses":["test"],"reference_data":{"ideal_plan_length":1}}' || echo "Warning: Failed to create rule-based plan-optimality uptime check"

gcloud monitoring uptime create http rule-based-tool-failure-rate-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/tool-failure-rate" \
    --project=$PROJECT_ID \
    --display-name="Rule-Based Tool Failure Rate API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"conversation_history":["user: test"],"tool_calls":[],"agent_responses":["test"],"reference_data":{}}' || echo "Warning: Failed to create rule-based tool-failure-rate uptime check"

gcloud monitoring uptime create http rule-based-fallback-rate-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/fallback-rate" \
    --project=$PROJECT_ID \
    --display-name="Rule-Based Fallback Rate API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"conversation_history":["user: test"],"tool_calls":[],"agent_responses":["test"],"reference_data":{}}' || echo "Warning: Failed to create rule-based fallback-rate uptime check"

# Create uptime check for agentic metrics endpoints (similarity)
gcloud monitoring uptime create http similarity-agent-goal-accuracy-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/agent-goal-accuracy" \
    --project=$PROJECT_ID \
    --display-name="Similarity Agent Goal Accuracy API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"conversation_history":["user: test"],"tool_calls":[],"agent_responses":["test"],"reference_data":{"target_goal":"test goal"}}' || echo "Warning: Failed to create similarity agent-goal-accuracy uptime check"

gcloud monitoring uptime create http similarity-intent-resolution-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/intent-resolution" \
    --project=$PROJECT_ID \
    --display-name="Similarity Intent Resolution API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"conversation_history":["user: test"],"tool_calls":[],"agent_responses":["test"],"reference_data":{"reference_intent":"test intent"}}' || echo "Warning: Failed to create similarity intent-resolution uptime check"

# Create uptime check for agentic metrics endpoint (sentence-transformers)
gcloud monitoring uptime create http sentence-transformers-topic-adherence-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/topic-adherence" \
    --project=$PROJECT_ID \
    --display-name="Sentence Transformers Topic Adherence API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"conversation_history":["user: test"],"tool_calls":[],"agent_responses":["test"],"reference_data":{"expected_topics":["test topic"]}}' || echo "Warning: Failed to create sentence-transformers topic-adherence uptime check"

# Moritzlaurel Bias Detection API Health Checks
gcloud monitoring uptime create http moritzlaurel-gender-bias-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/gender-bias" \
    --project=$PROJECT_ID \
    --display-name="Moritzlaurel Gender Bias API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"Women are naturally better at nurturing children than men."}' || echo "Warning: Failed to create moritzlaurel gender-bias uptime check"

# Create uptime check for token bloat dos service
gcloud monitoring uptime create http token-bloat-dos-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/token-bloat-dos" \
    --project=$PROJECT_ID \
    --display-name="Token Bloat DoS API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create token bloat dos uptime check"

# Create uptime check for supply chain risk service
gcloud monitoring uptime create http supply-chain-risk-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/supply-chain-risk" \
    --project=$PROJECT_ID \
    --display-name="Supply Chain Risk API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"model_name":"test"}' || echo "Warning: Failed to create supply chain risk uptime check"

# Create uptime check for membership inference risk service
gcloud monitoring uptime create http membership-inference-risk-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/membership-inference-risk" \
    --project=$PROJECT_ID \
    --display-name="Membership Inference Risk API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create membership inference risk uptime check"

# Create uptime check for model leakage service
gcloud monitoring uptime create http model-leakage-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/model-leakage" \
    --project=$PROJECT_ID \
    --display-name="Model Leakage API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create model leakage uptime check"

# Create uptime check for plugin execution risk service
gcloud monitoring uptime create http plugin-execution-risk-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/plugin-execution-risk" \
    --project=$PROJECT_ID \
    --display-name="Plugin Execution Risk API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"text":"test"}' || echo "Warning: Failed to create plugin execution risk uptime check"

# Create uptime check for autonomy risk service
gcloud monitoring uptime create http autonomy-risk-api-health \
    --uri="https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/autonomy-risk" \
    --project=$PROJECT_ID \
    --display-name="Autonomy Risk API Health Check" \
    --check-interval=60 \
    --timeout=5 \
    --content-type=application/json \
    --request-method=POST \
    --body='{"llm_output":"test"}' || echo "Warning: Failed to create autonomy risk uptime check"


echo "API Gateway deployed successfully!"
echo "Environment: $ENV"
echo "Gateway URL: https://$GATEWAY_URL"
echo "Test endpoints:"
echo "- Similarity: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/text-similarity"
echo "- RoBERTa Bias: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/bias"
echo "- Moritzlaurel Gender Bias: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/gender-bias"
echo "- Moritzlaurel Racial Bias: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/racial-bias"
echo "- Moritzlaurel Intersectionality: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/intersectionality"
echo "- BLEU Score: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/bleu-score"
echo "- Compression Score: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/compression-score"
echo "- Cosine Similarity: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/cosine-similarity"
echo "- Fuzzy Score: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/fuzzy-score"
echo "- ROUGE Score: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/rouge-score"
echo "- METEOR Score: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/meteor-score"
echo "- Longformer: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/coherence"
echo "- BERT Detoxify: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/toxicity"
echo "- BERT Detoxify Hate Speech: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/hate-speech"
echo "- BERT Detoxify Sexual Content: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/sexual-content"
echo "- BERT Detoxify Terrorism: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/terrorism"
echo "- BERT Detoxify Violence: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/violence"
echo "- BERT Detoxify Self-Harm: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/self-harm"
echo "- DistilBERT NSFW: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/nsfw"
echo "- DistilRoBERTa Refusal: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/refusal"
echo "- BERT Base Uncased: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/creativity"
echo "- DeBERTa v3 Large: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/answer_relevance"
echo "- Sentence Transformers: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/compute/similarity"
echo "- Flan T5: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/factual_consistency"
echo "- Token Classification: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/data_leakage"
echo "- DeBERTa v3 Base: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/prompt_injection"
echo "- T5 Base: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/grammar"
echo "- Diversity: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/diversity"
echo "- Readability: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/readability"
echo "- Clarity: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/clarity"
echo "- Invisible Text: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/invisible_text"
echo "- Insecure Output: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/insecure_output"
echo "- BERTweet Base: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/response_tone"
echo "- Context Precision: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/context-precision"
echo "- Context Recall: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/context-recall"
echo "- Context Entities Recall: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/context-entities-recall"
echo "- Noise Sensitivity: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/noise-sensitivity"
echo "- Response Relevancy: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/response-relevancy"
echo "- Context Relevance: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/context-relevance"
echo "- Faithfulness: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/faithfulness"
echo "- Tool Call Accuracy: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/tool-call-accuracy"
echo "- Plan Coherence: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/plan-coherence"
echo "- Plan Optimality: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/plan-optimality"
echo "- Tool Failure Rate: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/tool-failure-rate"
echo "- Fallback Rate: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/fallback-rate"
echo "- Agent Goal Accuracy: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/agent-goal-accuracy"
echo "- Intent Resolution: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/intent-resolution"
echo "- Topic Adherence: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/evaluate/topic-adherence"
echo "- Token Bloat DoS: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/token-bloat-dos"
echo "- Supply Chain Risk: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/supply-chain-risk"
echo "- Membership Inference Risk: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/membership-inference-risk"
echo "- Model Leakage: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/model-leakage"
echo "- Plugin Execution Risk: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/plugin-execution-risk"
echo "- Autonomy Risk: https://$GATEWAY_URL/atri_raime_ml_services/api/v1/detect/autonomy-risk"
echo "Backup config saved as: backup_config_${TIMESTAMP}.json"

# Instructions for rollback
echo -e "\nTo rollback to previous configuration:"
echo "./scripts/deploy_api_gateway.sh rollback backup_config_${TIMESTAMP}.json"

# Define service URLs
BLEU_SCORE_URL_PROD="https://nltk-production-drnc7zg5yq-uc.a.run.app/compute/bleu-score"
BLEU_SCORE_URL_STAGING="https://nltk-staging-drnc7zg5yq-uc.a.run.app/compute/bleu-score"

COMPRESSION_SCORE_URL_PROD="https://nltk-production-drnc7zg5yq-uc.a.run.app/compute/compression-score"
COMPRESSION_SCORE_URL_STAGING="https://nltk-staging-drnc7zg5yq-uc.a.run.app/compute/compression-score"

COSINE_SIMILARITY_URL_PROD="https://nltk-production-drnc7zg5yq-uc.a.run.app/compute/cosine-similarity"
COSINE_SIMILARITY_URL_STAGING="https://nltk-staging-drnc7zg5yq-uc.a.run.app/compute/cosine-similarity"

FUZZY_SCORE_URL_PROD="https://nltk-production-drnc7zg5yq-uc.a.run.app/compute/fuzzy-score"
FUZZY_SCORE_URL_STAGING="https://nltk-staging-drnc7zg5yq-uc.a.run.app/compute/fuzzy-score"

ROUGE_SCORE_URL_PROD="https://nltk-production-drnc7zg5yq-uc.a.run.app/compute/rouge-score"
ROUGE_SCORE_URL_STAGING="https://nltk-staging-drnc7zg5yq-uc.a.run.app/compute/rouge-score" 
