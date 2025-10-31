"""Storage service for Ada Cloud infrastructure.

This module provides S3-compatible storage integration with Wasabi for
persisting models, logs, checkpoints, and mission data.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union, BinaryIO
from pathlib import Path
import json
import hashlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import boto3, handle missing dependency
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - storage functions will be simulated")


@dataclass
class StorageConfig:
    """Storage configuration."""
    endpoint_url: str = "https://s3.wasabisys.com"
    region_name: str = "us-east-1"
    bucket_name: str = "ada-models"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create config from environment variables."""
        return cls(
            access_key_id=os.getenv("WASABI_KEY_ID"),
            secret_access_key=os.getenv("WASABI_SECRET"),
            bucket_name=os.getenv("WASABI_BUCKET", "ada-models"),
            endpoint_url=os.getenv("WASABI_ENDPOINT", "https://s3.wasabisys.com"),
            region_name=os.getenv("WASABI_REGION", "us-east-1"),
        )


class StorageService:
    """S3-compatible storage service for Ada Cloud."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """Initialize storage service.
        
        Args:
            config: Storage configuration, uses environment if None
        """
        self.config = config or StorageConfig.from_env()
        self.client = None
        self._initialized = False
        
        if BOTO3_AVAILABLE and self.config.access_key_id and self.config.secret_access_key:
            self._initialize_client()
        else:
            logger.warning("Storage service initialized without S3 backend - operations will be simulated")
        
        # Create local cache directory
        self.cache_dir = Path("/tmp/ada_storage_cache")
        self.cache_dir.mkdir(exist_ok=True)
    
    def _initialize_client(self):
        """Initialize S3 client."""
        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name=self.config.region_name,
            )
            
            # Test connection
            self.client.list_buckets()
            self._initialized = True
            logger.info(f"Storage service connected to {self.config.endpoint_url}")
            
            # Ensure bucket exists
            self._ensure_bucket_exists()
            
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {e}")
            self.client = None
    
    def _ensure_bucket_exists(self):
        """Ensure the target bucket exists."""
        if not self.client:
            return
        
        try:
            self.client.head_bucket(Bucket=self.config.bucket_name)
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    self.client.create_bucket(Bucket=self.config.bucket_name)
                    logger.info(f"Created bucket: {self.config.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
            else:
                logger.error(f"Bucket access error: {e}")
    
    async def upload_file(
        self,
        file_path: Union[str, Path],
        key: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Upload a file to storage.
        
        Args:
            file_path: Path to file to upload
            key: Storage key (uses filename if not provided)
            metadata: Optional metadata
            
        Returns:
            Upload result information
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        key = key or file_path.name
        
        if self._initialized and self.client:
            try:
                # Prepare upload parameters
                upload_args = {
                    "Key": key,
                    "Filename": str(file_path),
                }
                
                if metadata:
                    upload_args["ExtraArgs"] = {"Metadata": metadata}
                
                # Upload file
                self.client.upload_file(**upload_args)
                
                # Get file info
                file_stats = file_path.stat()
                
                result = {
                    "success": True,
                    "key": key,
                    "bucket": self.config.bucket_name,
                    "size": file_stats.st_size,
                    "last_modified": file_stats.st_mtime,
                    "etag": self._calculate_file_etag(file_path),
                    "upload_time": time.time(),
                }
                
                logger.info(f"Uploaded {key} ({file_stats.st_size} bytes)")
                return result
                
            except Exception as e:
                logger.error(f"Failed to upload {key}: {e}")
                return {
                    "success": False,
                    "key": key,
                    "error": str(e),
                }
        else:
            # Simulated upload - copy to cache
            cache_path = self.cache_dir / key
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(file_path, cache_path)
            
            file_stats = file_path.stat()
            
            return {
                "success": True,
                "key": key,
                "bucket": f"simulated-{self.config.bucket_name}",
                "size": file_stats.st_size,
                "last_modified": file_stats.st_mtime,
                "upload_time": time.time(),
                "simulated": True,
            }
    
    async def upload_json(
        self,
        key: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Upload JSON data to storage.
        
        Args:
            key: Storage key for the JSON data
            data: JSON data to upload
            metadata: Optional metadata
            
        Returns:
            Upload result information
        """
        json_str = json.dumps(data, indent=2)
        
        if self._initialized and self.client:
            try:
                # Upload JSON as object
                upload_args = {
                    "Bucket": self.config.bucket_name,
                    "Key": key,
                    "Body": json_str,
                    "ContentType": "application/json",
                }
                
                if metadata:
                    upload_args["Metadata"] = metadata
                
                self.client.put_object(**upload_args)
                
                # Calculate ETag
                etag = hashlib.md5(json_str.encode()).hexdigest()
                
                result = {
                    "success": True,
                    "key": key,
                    "bucket": self.config.bucket_name,
                    "size": len(json_str),
                    "etag": etag,
                    "upload_time": time.time(),
                }
                
                logger.info(f"Uploaded JSON {key} ({len(json_str)} bytes)")
                return result
                
            except Exception as e:
                logger.error(f"Failed to upload JSON {key}: {e}")
                return {
                    "success": False,
                    "key": key,
                    "error": str(e),
                }
        else:
            # Simulated upload - save to cache
            cache_path = self.cache_dir / key
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            
            etag = hashlib.md5(json_str.encode()).hexdigest()
            
            return {
                "success": True,
                "key": key,
                "bucket": f"simulated-{self.config.bucket_name}",
                "size": len(json_str),
                "etag": etag,
                "upload_time": time.time(),
                "simulated": True,
            }
    
    async def download_file(
        self,
        key: str,
        destination: Union[str, Path],
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Download a file from storage.
        
        Args:
            key: Storage key to download
            destination: Local destination path
            overwrite: Whether to overwrite existing files
            
        Returns:
            Download result information
        """
        destination = Path(destination)
        
        if destination.exists() and not overwrite:
            return {
                "success": False,
                "key": key,
                "error": "File already exists and overwrite=False",
            }
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if self._initialized and self.client:
            try:
                # Download file
                self.client.download_file(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Filename=str(destination),
                )
                
                # Get file info
                file_stats = destination.stat()
                
                result = {
                    "success": True,
                    "key": key,
                    "path": str(destination),
                    "size": file_stats.st_size,
                    "download_time": time.time(),
                }
                
                logger.info(f"Downloaded {key} to {destination}")
                return result
                
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")
                return {
                    "success": False,
                    "key": key,
                    "error": str(e),
                }
        else:
            # Simulated download - copy from cache
            cache_path = self.cache_dir / key
            
            if not cache_path.exists():
                return {
                    "success": False,
                    "key": key,
                    "error": "File not found in simulated storage",
                }
            
            import shutil
            shutil.copy2(cache_path, destination)
            
            file_stats = destination.stat()
            
            return {
                "success": True,
                "key": key,
                "path": str(destination),
                "size": file_stats.st_size,
                "download_time": time.time(),
                "simulated": True,
            }
    
    async def download_json(self, key: str) -> Dict[str, Any]:
        """Download JSON data from storage.
        
        Args:
            key: Storage key for the JSON data
            
        Returns:
            Downloaded JSON data or error info
        """
        if self._initialized and self.client:
            try:
                response = self.client.get_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                )
                
                json_data = json.loads(response["Body"].read().decode())
                
                logger.info(f"Downloaded JSON {key}")
                return {
                    "success": True,
                    "key": key,
                    "data": json_data,
                    "download_time": time.time(),
                }
                
            except Exception as e:
                logger.error(f"Failed to download JSON {key}: {e}")
                return {
                    "success": False,
                    "key": key,
                    "error": str(e),
                }
        else:
            # Simulated download - read from cache
            cache_path = self.cache_dir / key
            
            if not cache_path.exists():
                return {
                    "success": False,
                    "key": key,
                    "error": "JSON file not found in simulated storage",
                }
            
            with open(cache_path, "r") as f:
                json_data = json.load(f)
            
            return {
                "success": True,
                "key": key,
                "data": json_data,
                "download_time": time.time(),
                "simulated": True,
            }
    
    async def list_files(
        self,
        prefix: str = "",
        limit: int = 1000,
        include_metadata: bool = False,
    ) -> Dict[str, Any]:
        """List files in storage.
        
        Args:
            prefix: Key prefix to filter
            limit: Maximum number of files to return
            include_metadata: Whether to include file metadata
            
        Returns:
            List of files and metadata
        """
        if self._initialized and self.client:
            try:
                paginator = self.client.get_paginator("list_objects_v2")
                pages = paginator.paginate(
                    Bucket=self.config.bucket_name,
                    Prefix=prefix,
                )
                
                files = []
                file_count = 0
                
                for page in pages:
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            if file_count >= limit:
                                break
                            
                            file_info = {
                                "key": obj["Key"],
                                "size": obj["Size"],
                                "last_modified": obj["LastModified"].timestamp(),
                                "etag": obj["ETag"].strip('"'),
                            }
                            
                            if include_metadata:
                                try:
                                    metadata_resp = self.client.head_object(
                                        Bucket=self.config.bucket_name,
                                        Key=obj["Key"],
                                    )
                                    file_info["metadata"] = metadata_resp.get("Metadata", {})
                                except:
                                    file_info["metadata"] = {}
                            
                            files.append(file_info)
                            file_count += 1
                        
                        if file_count >= limit:
                            break
                
                return {
                    "success": True,
                    "files": files,
                    "total_count": len(files),
                    "prefix": prefix,
                }
                
            except Exception as e:
                logger.error(f"Failed to list files: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }
        else:
            # Simulated listing - list from cache
            files = []
            
            try:
                cache_prefix = self.cache_dir / prefix if prefix else self.cache_dir
                
                if cache_prefix.is_dir():
                    for file_path in cache_prefix.rglob("*"):
                        if file_path.is_file() and len(files) < limit:
                            relative_path = file_path.relative_to(self.cache_dir)
                            
                            file_info = {
                                "key": str(relative_path),
                                "size": file_path.stat().st_size,
                                "last_modified": file_path.stat().st_mtime,
                                "simulated": True,
                            }
                            
                            files.append(file_info)
                
                return {
                    "success": True,
                    "files": files,
                    "total_count": len(files),
                    "prefix": prefix,
                    "simulated": True,
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                }
    
    async def delete_file(self, key: str) -> Dict[str, Any]:
        """Delete a file from storage.
        
        Args:
            key: Storage key to delete
            
        Returns:
            Deletion result information
        """
        if self._initialized and self.client:
            try:
                self.client.delete_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                )
                
                logger.info(f"Deleted {key}")
                return {
                    "success": True,
                    "key": key,
                    "deleted_at": time.time(),
                }
                
            except Exception as e:
                logger.error(f"Failed to delete {key}: {e}")
                return {
                    "success": False,
                    "key": key,
                    "error": str(e),
                }
        else:
            # Simulated deletion - remove from cache
            cache_path = self.cache_dir / key
            
            if cache_path.exists():
                cache_path.unlink()
                
                return {
                    "success": True,
                    "key": key,
                    "deleted_at": time.time(),
                    "simulated": True,
                }
            else:
                return {
                    "success": False,
                    "key": key,
                    "error": "File not found in simulated storage",
                    "simulated": True,
                }
    
    def _calculate_file_etag(self, file_path: Path) -> str:
        """Calculate ETag for a file (MD5 hash)."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    async def sync_directory(
        self,
        local_dir: Union[str, Path],
        storage_prefix: str = "",
        pattern: str = "*",
        delete_extra: bool = False,
    ) -> Dict[str, Any]:
        """Sync a local directory to storage.
        
        Args:
            local_dir: Local directory to sync
            storage_prefix: Storage key prefix
            pattern: File pattern to match
            delete_extra: Whether to delete extra files in storage
            
        Returns:
            Sync operation results
        """
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            return {
                "success": False,
                "error": f"Directory not found: {local_dir}",
            }
        
        # Get local files
        local_files = {}
        for file_path in local_dir.rglob(pattern):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                local_files[str(relative_path)] = file_path
        
        # Get storage files
        storage_result = await self.list_files(storage_prefix, limit=10000)
        if not storage_result["success"]:
            return {
                "success": False,
                "error": f"Failed to list storage files: {storage_result['error']}",
            }
        
        storage_files = {}
        for file_info in storage_result["files"]:
            key = file_info["key"]
            if key.startswith(storage_prefix):
                relative_key = key[len(storage_prefix):].lstrip("/")
                storage_files[relative_key] = file_info
        
        # Upload new and modified files
        uploaded = []
        updated = []
        
        for relative_key, local_path in local_files.items():
            storage_key = f"{storage_prefix}/{relative_key}" if storage_prefix else relative_key
            
            if relative_key not in storage_files:
                # New file - upload
                result = await self.upload_file(local_path, storage_key)
                if result["success"]:
                    uploaded.append(storage_key)
            else:
                # Check if file needs updating
                storage_info = storage_files[relative_key]
                local_mtime = local_path.stat().st_mtime
                
                if local_mtime > storage_info["last_modified"]:
                    result = await self.upload_file(local_path, storage_key, overwrite=True)
                    if result["success"]:
                        updated.append(storage_key)
        
        # Delete extra files if requested
        deleted = []
        if delete_extra:
            for relative_key, storage_info in storage_files.items():
                if relative_key not in local_files:
                    result = await self.delete_file(storage_info["key"])
                    if result["success"]:
                        deleted.append(storage_info["key"])
        
        return {
            "success": True,
            "uploaded": uploaded,
            "updated": updated,
            "deleted": deleted,
            "total_local_files": len(local_files),
            "total_storage_files": len(storage_files),
        }


# Global storage service instance
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get global storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service


# Modal wrapper functions
async def cloud_upload_checkpoint(file_path: str, key: Optional[str] = None) -> Dict[str, Any]:
    """Modal checkpoint upload function.
    
    Args:
        file_path: Path to checkpoint file
        key: Storage key (optional)
        
    Returns:
        Upload result
    """
    try:
        service = get_storage_service()
        return await service.upload_file(file_path, key)
    except Exception as e:
        logger.error(f"Checkpoint upload failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


async def cloud_download_model(model_name: str, destination: str) -> Dict[str, Any]:
    """Modal model download function.
    
    Args:
        model_name: Model name/key in storage
        destination: Local destination path
        
    Returns:
        Download result
    """
    try:
        service = get_storage_service()
        return await service.download_file(model_name, destination)
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "model_name": model_name,
        }
