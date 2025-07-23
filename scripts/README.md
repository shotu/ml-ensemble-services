# API Gateway Deployment

This directory contains scripts and configurations for deploying and managing the RAIME ML Services API Gateway.

## Files

### 1. deploy_api_gateway.sh
Main deployment script for the API Gateway. This script:
- Enables required Google Cloud APIs
- Creates/updates the API Gateway configuration
- Sets up monitoring and alerting
- Creates backups of configurations
- Supports rollback functionality

#### Usage:
```bash
# Deploy to production
./deploy_api_gateway.sh

# Deploy to staging
./deploy_api_gateway.sh staging

# Rollback to previous configuration
./deploy_api_gateway.sh rollback <backup_file>
```

#### Features:
- Environment handling (prod/staging)
- Automatic backup creation
- Rollback support
- Monitoring setup
- Error handling
- API key management

### 2. api-gateway-config.yaml
OpenAPI specification for the API Gateway. This file defines:
- API endpoints and their paths
- Request/response schemas
- Security definitions
- Rate limiting
- Backend service configurations

#### Current Endpoints:
1. Health Check:
   ```
   GET /atri_raime_ml_services/api/v1/health
   ```

2. Text Similarity:
   ```
   POST /atri_raime_ml_services/api/v1/compute/text-similarity
   ```

#### Security:
- API Key authentication via `x-api-key` header
- Rate limiting: 100 requests per minute
- JWT authentication support (Firebase)

#### Configuration Sections:
1. **Info**: API metadata and version
2. **Backend**: Cloud Run service configuration
3. **Security**: Authentication and authorization
4. **Definitions**: Request/response schemas
5. **Paths**: API endpoints and their configurations

## Deployment Process

1. **Prerequisites**:
   - Google Cloud SDK installed
   - Project configured (`atri-dev`)
   - Required APIs enabled

2. **Deployment Steps**:
   ```bash
   # 1. Update api-gateway-config.yaml if needed
   # 2. Run deployment script
   ./deploy_api_gateway.sh
   # 3. Verify deployment
   curl -X GET "https://raime-ml-gateway-prod-d00yiiqh.uc.gateway.dev/atri_raime_ml_services/api/v1/health" \
   --header "x-api-key: YOUR_API_KEY"
   ```

3. **Rollback**:
   ```bash
   ./deploy_api_gateway.sh rollback backup_config_TIMESTAMP.json
   ```

## Adding New Services

To add a new service:
1. Update `api-gateway-config.yaml`:
   - Add new path definition
   - Define request/response schemas
   - Configure backend service
2. Run deployment script
3. Test new endpoint

## Monitoring

The deployment script sets up:
- Uptime checks
- Error rate monitoring
- Latency monitoring
- Alert policies

## API Key Management

### Storing API Key in Secret Manager

1. **Create a secret**:
```bash
gcloud secrets create raime-ml-api-key \
    --project=atri-dev \
    --replication-policy="automatic"
```

2. **Store the API key**:
```bash
echo -n "YOUR_API_KEY" | gcloud secrets versions add raime-ml-api-key \
    --project=atri-dev \
    --data-file=-
```

### Accessing the API Key

1. **View the secret** in Google Cloud Console:
   - Go to: https://console.cloud.google.com/security/secret-manager
   - Select project: `atri-dev`
   - Look for secret: `raime-ml-api-key`

2. **Access the secret** in your applications:
```bash
gcloud secrets versions access latest --secret="raime-ml-api-key" --project=atri-dev
```

3. **Grant access** to other services or users:
```bash
gcloud secrets add-iam-policy-binding raime-ml-api-key \
    --project=atri-dev \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"
```

### API Key Security Best Practices

1. **Keep API keys secure**:
   - Never commit API keys to version control
   - Use Secret Manager for storage
   - Rotate keys periodically

2. **Restrict API key usage**:
   - Limit to specific APIs
   - Set IP address restrictions
   - Configure usage quotas

3. **Monitor API key usage**:
   - Track API calls
   - Set up alerts for unusual activity
   - Review access logs regularly

## Troubleshooting

1. **Deployment Issues**:
   - Check Google Cloud Console for API Gateway status
   - Verify API key permissions
   - Check Cloud Run service status

2. **API Issues**:
   - Verify API key is correct
   - Check rate limits
   - Validate request format

3. **Monitoring Issues**:
   - Check alert policies in Google Cloud Console
   - Verify uptime checks
   - Review error logs 