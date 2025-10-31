# Ada Cloud Infrastructure

This directory contains the cloud infrastructure components for Ada's Modal-hosted serverless backend with Wasabi storage integration.

## Architecture Overview

```
Local Mac Client → FastAPI Gateway → Modal Functions → Wasabi Storage
```

The cloud infrastructure provides:
- **Modal Functions**: Serverless compute for inference, training, optimization, and missions
- **Wasabi Storage**: S3-compatible storage for models, embeddings, and logs
- **FastAPI Gateway**: HTTP API with authentication, rate limiting, and error handling

## Components

- `modal_app.py`: Modal application entrypoint and function registry
- `api_gateway.py`: FastAPI HTTP gateway for client requests
- `inference_service.py`: High-performance inference for Ada modules
- `mission_service.py`: Mission orchestration and execution
- `optimizer_service.py`: Parameter optimization and model evolution
- `storage_service.py`: Wasabi S3-compatible storage service
- `config_cloud.yaml`: Cloud-specific configuration
- `requirements_cloud.txt`: Cloud deployment dependencies

## Deployment

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Wasabi Account**: Sign up at [wasabi.com](https://wasabi.com)
3. **API Keys**: Set environment variables:
   ```bash
   export ADA_API_KEY="your-ada-api-key"
   export WASABI_KEY_ID="your-wasabi-key-id"
   export WASABI_SECRET="your-wasabi-secret"
   export WASABI_ENDPOINT="https://s3.wasabisys.com"
   ```

### Setup

1. Install cloud dependencies:
   ```bash
   make setup-cloud
   ```

2. Deploy to Modal:
   ```bash
   make deploy-cloud
   ```

3. Test the deployment:
   ```bash
   make test-cloud
   ```

### Testing Cloud Connection

1. Test adapter connection:
   ```bash
   python -m interfaces.remote_client --action test
   ```

2. Check infrastructure status:
   ```bash
   make status-cloud
   ```

## Usage

### From Local Client

```python
from interfaces.remote_client import AdaCloudClient

async def main():
    client = AdaCloudClient()
    
    # Run inference on cloud
    result = await client.infer(
        module="core.reasoning",
        prompt="Analyze this data and provide insights.",
        parameters={"temperature": 0.7}
    )
    
    print(result["response"])

# Run the async main function
asyncio.run(main())
```

### Direct API Calls

```bash
# Inference
curl -X POST https://ada-cloud.modal.run/infer \
  -H "Authorization: Bearer $ADA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "module": "core.reasoning",
    "parameters": {"max_tokens": 100}
  }'

# Mission execution
curl -X POST https://ada-cloud.modal.run/mission \
  -H "Authorization: Bearer $ADA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": "Analyze system logs for errors",
    "context": {"priority": "high"},
    "priority": "high"
  }'

# Optimization
curl -X POST https://ada-cloud.modal.run/optimize \
  -H "Authorization: Bearer $ADA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "target_module": "core.reasoning",
    "type": "parameter_tuning",
    "budget": 1000
  }'
```

## Storage Management

### List models in Wasabi
```bash
python -m cloud.storage_service list --bucket ada-models --prefix models/
```

### Upload model checkpoint
```bash
python -m cloud.storage_service upload \
  --file storage/checkpoints/model.pt \
  --key models/model.pt \
  --bucket ada-models
```

### Sync memory database
```bash
make sync-storage
```

## Configuration

### Cloud Settings (config/settings.yaml)
```yaml
cloud:
  enabled: true
  provider: "modal"
  endpoint: "https://ada-cloud.modal.run"
  api_key: "${ADA_API_KEY}"
  gpu_enabled: true
  auto_scale_zero: true
```

### Wasabi Settings
```yaml
storage:
  provider: "wasabi"
  endpoint: "https://s3.wasabisys.com"
  region: "us-east-1"
  bucket: "ada-models"
  access_key_id: "${WASABI_KEY_ID}"
  secret_access_key: "${WASABI_SECRET}"
  min_storage_days: 90
  sync_on_startup: true
```

## Monitoring

### Check System Status
```bash
curl https://ada-cloud.modal.run/status \
  -H "Authorization: Bearer $ADA_API_KEY"
```

### View Metrics
```bash
curl https://ada-cloud.modal.run/metrics \
  -H "Authorization: Bearer $ADA_API_KEY"
```

### Client Metrics
```python
from interfaces.remote_client import AdaCloudClient

client = AdaCloudClient()
# After running some requests...
metrics = client.get_client_metrics()
print(metrics)
```

## Development

### Local Development

1. Start the API gateway locally:
   ```bash
   cd cloud
   python api_gateway.py --host 0.0.0.0 --port 8000 --reload
   ```

2. Test local instance:
   ```bash
   ADA_CLOUD_ENDPOINT=http://localhost:8000 python -m interfaces.remote_client --action test
   ```

### Adding New Functions

1. Implement function in appropriate service module
2. Add Modal function wrapper in `modal_app.py`
3. Add API endpoint in `api_gateway.py`
4. Update client if needed

### Testing

Run all cloud tests:
```bash
make test-cloud
```

Test individual components:
```bash
# Modal functions
modal run cloud.modal_app::health_check

# Storage service
python -m cloud.storage_service list

# API gateway
python -m cloud.api_gateway --action test
```

## Troubleshooting

### Common Issues

1. **Authentication Error**: Check ADA_API_KEY environment variable
2. **Modal Deployment Error**: Verify Modal account and credentials
3. **Wasabi Connection Error**: Check WASABI_KEY_ID and WASABI_SECRET
4. **GPU Not Available**: Adjust GPU settings in modal_app.py

### Debug Tips

1. Enable verbose logging:
   ```
   export ADA_CLOUD_LOG_LEVEL=DEBUG
   ```

2. Check Modal logs:
   ```bash
   modal app logs AdaCloud
   ```

3. Test connection locally:
   ```bash
   python -m interfaces.remote_client --action test
   ```

## Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Local Client  │────│  FastAPI Gateway │────│   Modal Functions│
│                 │    │   (API/Rate)     │    │                 │
│ - CLI Interface │    │ - Authentication │    │ - Inference     │
│ - Voice Interface│   │ - Rate Limiting  │    │ - Training      │
│ - Web Interface │    │ - Error Handling │    │ - Optimization  │
└─────────────────┘    └──────────────────┘    │ - Missions      │
                                                └─────────────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  Wasabi Storage │
                                                │                 │
                                                │ - Models        │
                                                │ - Embeddings    │
                                                │ - Checkpoints   │
                                                │ - Logs          │
                                                │ - Memory        │
                                                └─────────────────┘
```

## Performance Notes

- **Auto-scaling**: Functions scale to zero when idle (cost $0)
- **GPU Acceleration**: A10G GPUs for inference and training
- **Optimization**: Batch requests for better throughput
- **Caching**: Model loading optimization with caching
- **Compression**: Automatic compression for storage

## Security

- API key authentication for all requests
- Rate limiting to prevent abuse
- Input validation and sanitization
- Secure TLS connections
- No sensitive data in logs
