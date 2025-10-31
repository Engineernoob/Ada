# Ada Cloud Deployment Guide

## Prerequisites

1. **Modal Account**: Set up your Modal account
2. **Wasabi Account**: Set up your Wasabi storage account

## Setup Steps

### 1. Configure Modal Authentication

```bash
# Authenticate with Modal (opens browser)
modal token new
```

### 2. Set Environment Variables

Create or update your `.env` file in the Ada root directory:

```bash
touch .env
```

Add these variables to `.env`:

```env
# Modal Cloud Configuration
ADA_CLOUD_ENDPOINT="https://your-username-ada-cloud.modal.run"
ADA_API_KEY="your-generated-api-key"  # We'll generate this after first deployment

# Wasabi Storage Configuration  
WASABI_KEY_ID="your-wasabi-access-key"
WASABI_SECRET="your-wasabi-secret-key"
WASABI_ENDPOINT="https://s3.wasabisys.com"
```

### 3. Install Cloud Dependencies

```bash
make setup-cloud
```

### 4. Deploy to Modal

```bash
# Deploy the cloud infrastructure
make deploy-cloud
```

After deployment, Modal will provide your actual endpoint URL which you should update in your `.env` file.

### 5. Generate API Key

After deployment, Modal might provide an API key or you may need to generate one:

```bash
# If Modal provides a key after deployment, add it to .env
# Or you might need to use Modal dashboard to generate one
```

### 6. Test Local Connection

```bash
# Test the cloud client connection
make test-cloud
```

### 7. Verify Full Setup

```bash
# Check infrastructure status
make status-cloud
```

## Environment Variable Details

### Modal Variables
- `ADA_CLOUD_ENDPOINT`: Your Modal app endpoint (after deployment)
- `ADA_API_KEY`: API key for authentication

### Wasabi Variables
- `WASABI_KEY_ID`: Wasabi access key ID
- `WASABI_SECRET`: Wasabi secret access key
- `WASABI_ENDPOINT`: Wasabi S3 endpoint URL

## Troubleshooting

### Modal Authentication Issues
```bash
# Reset Modal authentication
modal token new
```

### Wasabi Connection Issues
- Verify your Wasabi account is active
- Check bucket naming rules (must be globally unique)
- Ensure you have the correct region settings

### API Key Missing
If you don't have an API key yet, the system will be configured to work without it for initial testing.

### First Deployment
For the very first deployment, you might need to modify some settings to allow the system to work without an API key initially.
