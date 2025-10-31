import modal
import os
from typing import Dict, Any, Optional
from cloud.inference_service import ada_infer as core_infer
from cloud.optimizer_service import cloud_optimize as core_optimize
from cloud.mission_service import cloud_run_mission as core_mission

# Create Modal App (use AdaCloudFinal to avoid conflicts)
app = modal.App("AdaCloudFinal")

# Define shared Modal image
AdaCloudImage = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("cloud/requirements_cloud.txt")
    .env({
        "TORCH_HOME": "/root/.cache/torch",
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface"
    })
)

# Create storage volume (exists)
storage_volume = modal.Volume.from_name("ada-cloud-storage", create_if_missing=True)

# Create secrets for Wasabi storage
ada_secrets = modal.Secret.from_dict({
    "WASABI_KEY_ID": os.environ.get("WASABI_KEY_ID", "4OIHFFRH7L9I49TZ2UQD"),
    "WASABI_SECRET": os.environ.get("WASABI_SECRET", "h68fdXXztPem0E0yCUCb8nYpmooQKtIAiUfctXGn"),
    "WASABI_ENDPOINT": os.environ.get("WASABI_ENDPOINT", "https://s3.wasabisys.com"),
    "ADA_CLOUD_ENDPOINT": os.environ.get("ADA_CLOUD_ENDPOINT", "https://engineernoob--adacloudapi-api.modal.run"),
    "ADA_API_KEY": os.environ.get("ADA_API_KEY", "placeholder-key-for-deployment"),
    "ADA_CLOUD_TIMEOUT": os.environ.get("ADA_CLOUD_TIMEOUT", "300"),
    "ADA_CLOUD_MAX_RETRIES": os.environ.get("ADA_CLOUD_MAX_RETRIES", "3"),
    "ADA_CLOUD_RETRY_DELAY": os.environ.get("ADA_CLOUD_RETRY_DELAY", "1.0"),
    "ADA_CLOUD_ENABLE_STREAMING": os.environ.get("ADA_CLOUD_ENABLE_STREAMING", "true"),
})

# Core Inference Function
@app.function(
    image=AdaCloudImage, 
    gpu="A10G",
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
def ada_infer(data):
    import json
    if isinstance(data, str):
        data = json.loads(data)
    return core_infer(data)

# Optimizer Function  
@app.function(
    image=AdaCloudImage, 
    gpu="A10G",
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_optimize(params):
    import json
    if isinstance(params, str):
        params = json.loads(params)
    return await core_optimize(params)

# Mission Daemon Function
@app.function(
    image=AdaCloudImage,
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_mission(goal):
    # goal is expected to be a string param directly
    return await core_mission(goal)

# Storage Functions
@app.function(
    image=AdaCloudImage,
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_upload_file(file_path, key=None):
    """
    Upload a file to Wasabi S3 storage.
    
    Args:
        file_path: Path to file to upload
        key: Storage key (uses filename if not provided)
        
    Returns:
        Upload result information
    """
    from cloud.storage_service import get_storage_service
    storage = get_storage_service()
    return await storage.upload_file(file_path, key)

@app.function(
    image=AdaCloudImage,
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_upload_json(key, data):
    """
    Upload JSON data to Wasabi S3 storage.
    
    Args:
        key: Storage key for the JSON data
        data: JSON data to upload
        
    Returns:
        Upload result information
    """
    from cloud.storage_service import get_storage_service
    storage = get_storage_service()
    return await storage.upload_json(key, data)

@app.function(
    image=AdaCloudImage,
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_download_file(key, destination):
    """
    Download a file from Wasabi S3 storage.
    
    Args:
        key: Storage key to download
        destination: Local destination path
        
    Returns:
        Download result information
    """
    from cloud.storage_service import get_storage_service
    storage = get_storage_service()
    return await storage.download_file(key, destination)

@app.function(
    image=AdaCloudImage,
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_download_json(key):
    """
    Download JSON data from Wasabi S3 storage.
    
    Args:
        key: Storage key for the JSON data
        
    Returns:
        Downloaded JSON data or error info
    """
    from cloud.storage_service import get_storage_service
    storage = get_storage_service()
    return await storage.download_json(key)

@app.function(
    image=AdaCloudImage,
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_list_files(prefix="", limit=1000):
    """
    List files in Wasabi S3 storage.
    
    Args:
        prefix: Key prefix to filter
        limit: Maximum number of files to return
        
    Returns:
        List of files and metadata
    """
    from cloud.storage_service import get_storage_service
    storage = get_storage_service()
    return await storage.list_files(prefix, limit, include_metadata=True)

@app.function(
    image=AdaCloudImage,
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_delete_file(key):
    """
    Delete a file from Wasabi S3 storage.
    
    Args:
        key: Storage key to delete
        
    Returns:
        Deletion result information
    """
    from cloud.storage_service import get_storage_service
    storage = get_storage_service()
    return await storage.delete_file(key)

@app.function(
    image=AdaCloudImage,
    volumes={"/root/ada/storage": storage_volume},
    secrets=[ada_secrets]
)
async def ada_sync_directory(local_dir, storage_prefix=""):
    """
    Sync a local directory to Wasabi S3 storage.
    
    Args:
        local_dir: Local directory to sync
        storage_prefix: Storage key prefix
        
    Returns:
        Sync operation results
    """
    from cloud.storage_service import get_storage_service
    storage = get_storage_service()
    return await storage.sync_directory(local_dir, storage_prefix)
