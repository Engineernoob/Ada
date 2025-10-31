"""Wasabi Hot Cloud Storage integration for Ada's persistent data.

This module provides S3-compatible storage operations for models, embeddings,
checkpoints, and logs using Wasabi's cloud storage service.
"""

from __future__ import annotations

import os
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class WasabiStorageService:
    """Wasabi S3-compatible storage service for Ada's cloud infrastructure."""
    
    def __init__(
        self,
        bucket_name: str = "ada-models",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
        endpoint_url: str = "https://s3.wasabisys.com",
    ):
        """Initialize Wasabi storage service.
        
        Args:
            bucket_name: Name of the Wasabi bucket
            access_key_id: Wasabi access key ID (from env if not provided)
            secret_access_key: Wasabi secret access key (from env if not provided)
            region_name: AWS region (wasabi uses us-east-1)
            endpoint_url: Wasabi S3 endpoint URL
        """
        self.bucket_name = bucket_name
        self.access_key_id = access_key_id or os.getenv("WASABI_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("WASABI_SECRET")
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        
        if not all([self.access_key_id, self.secret_access_key]):
            raise ValueError("Wasabi credentials must be provided or set as environment variables")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket '{self.bucket_name}' exists and is accessible")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Created bucket '{self.bucket_name}'")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    raise
            else:
                logger.error(f"Error accessing bucket: {e}")
                raise
    
    def upload_file(
        self,
        file_path: Union[str, Path],
        object_key: str,
        compress: bool = True,
        metadata: Optional[Dict[str, str]] = None,
    ) ->bool:
        """Upload a file to Wasabi storage.
        
        Args:
            file_path: Path to local file
            object_key: S3 object key (path in bucket)
            compress: Whether to compress the file with gzip
            metadata: Optional metadata to attach
            
        Returns:
            True if upload successful
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if compress:
                # Read file and compress
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                compressed_data = gzip.compress(file_data)
                
                # Upload with compression
                extra_args = {
                    'ContentEncoding': 'gzip',
                    'Metadata': metadata or {},
                }
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=compressed_data,
                    **extra_args
                )
            else:
                # Regular upload
                self.s3_client.upload_file(
                    str(file_path),
                    self.bucket_name,
                    object_key,
                    ExtraArgs={'Metadata': metadata or {}} if metadata else None
                )
            
            logger.info(f"Uploaded {file_path} to s3://{self.bucket_name}/{object_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            return False
    
    def download_file(
        self,
        object_key: str,
        file_path: Union[str, Path],
        decompress: bool = True,
    ) -> bool:
        """Download a file from Wasabi storage.
        
        Args:
            object_key: S3 object key
            file_path: Local path to save file
            decompress: Whether to decompress if compressed
            
        Returns:
            True if download successful
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if decompress:
                # Download and decompress
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=object_key
                )
                
                compressed_data = response['Body'].read()
                if response.get('ContentEncoding') == 'gzip':
                    file_data = gzip.decompress(compressed_data)
                else:
                    file_data = compressed_data
                
                with open(file_path, 'wb') as f:
                    f.write(file_data)
            else:
                # Regular download
                self.s3_client.download_file(
                    self.bucket_name,
                    object_key,
                    str(file_path)
                )
            
            logger.info(f"Downloaded s3://{self.bucket_name}/{object_key} to {file_path}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"Object not found: s3://{self.bucket_name}/{object_key}")
            else:
                logger.error(f"Failed to download {object_key}: {e}")
            return False
    
    def upload_pickle(
        self,
        obj: Any,
        object_key: str,
        compress: bool = True,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Upload a Python object as pickle file.
        
        Args:
            obj: Python object to serialize
            object_key: S3 object key
            compress: Whether to compress
            metadata: Optional metadata
            
        Returns:
            True if upload successful
        """
        try:
            data = pickle.dumps(obj)
            if compress:
                data = gzip.compress(data)
            
            extra_args = {
                'ContentEncoding': 'gzip' if compress else None,
                'Metadata': metadata or {},
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data,
                **{k: v for k, v in extra_args.items() if v is not None}
            )
            
            logger.info(f"Uploaded pickle object to s3://{self.bucket_name}/{object_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload pickle object: {e}")
            return False
    
    def download_pickle(
        self,
        object_key: str,
        decompress: bool = True,
    ) -> Optional[Any]:
        """Download and deserialize a Python object.
        
        Args:
            object_key: S3 object key
            decompress: Whether to decompress
            
        Returns:
            Deserialized Python object or None if failed
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            data = response['Body'].read()
            if decompress and response.get('ContentEncoding') == 'gzip':
                data = gzip.decompress(data)
            
            obj = pickle.loads(data)
            logger.info(f"Downloaded pickle object from s3://{self.bucket_name}/{object_key}")
            return obj
            
        except Exception as e:
            logger.error(f"Failed to download pickle object: {e}")
            return None
    
    def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List objects in bucket with optional prefix filter.
        
        Args:
            prefix: Prefix filter
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object information
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = response.get('Contents', [])
            return [
                {
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag'].strip('"'),
                }
                for obj in objects
            ]
            
        except ClientError as e:
            logger.error(f"Failed to list objects: {e}")
            return []
    
    def delete_object(self, object_key: str) -> bool:
        """Delete an object from storage.
        
        Args:
            object_key: S3 object key
            
        Returns:
            True if deletion successful
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.info(f"Deleted s3://{self.bucket_name}/{object_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete {object_key}: {e}")
            return False
    
    def check_expiry(self, object_key: str, min_days: int = 90) -> bool:
        """Check if object can be deleted (wasabi has 90-day minimum).
        
        Args:
            object_key: S3 object key
            min_days: Minimum storage days (wasabi policy)
            
        Returns:
            True if object is past minimum storage period
        """
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
            last_modified = response['LastModified']
            
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            age_days = (now - last_modified).days
            
            return age_days >= min_days
            
        except ClientError as e:
            logger.error(f"Failed to check object expiry: {e}")
            return False


# Convenience functions for common operations
def upload_checkpoint(file_path: str, bucket: str = "ada-models") -> bool:
    """Upload model checkpoint to Wasabi."""
    storage = WasabiStorageService(bucket_name=bucket)
    object_key = f"checkpoints/{Path(file_path).name}"
    return storage.upload_file(file_path, object_key)


def download_model(model_name: str, bucket: str = "ada-models", local_path: str = "./models") -> bool:
    """Download model from Wasabi."""
    storage = WasabiStorageService(bucket_name=bucket)
    object_key = f"models/{model_name}"
    local_file = Path(local_path) / model_name
    return storage.download_file(object_key, local_file)


def list_models(bucket: str = "ada-models") -> List[str]:
    """List available models in Wasabi."""
    storage = WasabiStorageService(bucket_name=bucket)
    objects = storage.list_objects(prefix="models/")
    return [obj['key'].replace('models/', '') for obj in objects]


def sync_memory(bucket: str = "ada-models") -> bool:
    """Synchronize memory database with cloud storage."""
    storage = WasabiStorageService(bucket_name=bucket)
    
    # Upload conversations database
    db_path = Path("storage/conversations.db")
    if db_path.exists():
        return storage.upload_file(db_path, "memory/conversations.db")
    
    return False


# Command line interface for manual operations
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wasabi Storage Service CLI")
    parser.add_argument("command", choices=["upload", "download", "list", "sync", "delete"])
    parser.add_argument("--key", help="Object key")
    parser.add_argument("--file", help="Local file path")
    parser.add_argument("--bucket", default="ada-models", help="Bucket name")
    parser.add_argument("--prefix", default="", help="Prefix filter for list")
    
    args = parser.parse_args()
    
    storage = WasabiStorageService(bucket_name=args.bucket)
    
    if args.command == "upload":
        if not args.file or not args.key:
            print("Error: --file and --key required for upload")
        else:
            success = storage.upload_file(args.file, args.key)
            print(f"Upload {'successful' if success else 'failed'}")
    
    elif args.command == "download":
        if not args.key or not args.file:
            print("Error: --key and --file required for download")
        else:
            success = storage.download_file(args.key, args.file)
            print(f"Download {'successful' if success else 'failed'}")
    
    elif args.command == "list":
        objects = storage.list_objects(prefix=args.prefix)
        for obj in objects:
            print(f"{obj['key']} ({obj['size']} bytes, {obj['last_modified']})")
    
    elif args.command == "sync":
        success = sync_memory(args.bucket)
        print(f"Sync {'successful' if success else 'failed'}")
    
    elif args.command == "delete":
        if not args.key:
            print("Error: --key required for delete")
        else:
            success = storage.delete_object(args.key)
            print(f"Delete {'successful' if success else 'failed'}")
