#!/usr/bin/env python3
"""
Test script for Wasabi S3 storage setup.
"""

import os
import asyncio
import json
from pathlib import Path
from cloud.storage_service import StorageService, get_storage_service

async def test_wasabi_connection():
    """Test basic Wasabi S3 connection and operations."""
    print("🌩️  Testing Wasabi S3 connection...")
    
    # Get storage service (will use environment variables)
    storage = get_storage_service()
    
    if not storage._initialized:
        print("❌ Storage service not initialized - check Wasabi credentials")
        print(f"   Endpoint: {storage.config.endpoint_url}")
        print(f"   Bucket: {storage.config.bucket_name}")
        print(f"   Access Key: {'SET' if storage.config.access_key_id else 'MISSING'}")
        return False
    
    print("✅ Storage service initialized successfully")
    
    # Test listing buckets
    try:
        buckets = storage.client.list_buckets()
        bucket_names = [b['Name'] for b in buckets['Buckets']]
        print(f"📦 Available buckets: {bucket_names}")
        
        # Check if our bucket exists
        if storage.config.bucket_name in bucket_names:
            print(f"✅ Target bucket '{storage.config.bucket_name}' exists")
        else:
            print(f"⚠️  Target bucket '{storage.config.bucket_name}' not found, will be created")
    except Exception as e:
        print(f"❌ Failed to list buckets: {e}")
        return False
    
    return True

async def test_storage_operations():
    """Test basic storage operations."""
    print("\n🧪 Testing storage operations...")
    
    storage = get_storage_service()
    
    # Test 1: Upload JSON
    test_data = {
        "timestamp": asyncio.get_event_loop().time(),
        "message": "Ada Cloud Wasabi test",
        "config": {
            "version": "1.0",
            "test": True
        }
    }
    
    print("📤 Uploading test JSON...")
    result = await storage.upload_json("test/wasabi_test.json", test_data)
    
    if result["success"]:
        print(f"✅ JSON uploaded to: {result['key']}")
        print(f"   Size: {result['size']} bytes")
        print(f"   ETag: {result.get('etag', 'N/A')}")
    else:
        print(f"❌ JSON upload failed: {result.get('error')}")
        return False
    
    # Test 2: Download JSON
    print("\n📥 Downloading test JSON...")
    download_result = await storage.download_json("test/wasabi_test.json")
    
    if download_result["success"]:
        print("✅ JSON downloaded successfully")
        downloaded_data = download_result["data"]
        if downloaded_data["message"] == test_data["message"]:
            print("✅ Downloaded data matches uploaded data")
        else:
            print("❌ Downloaded data mismatch")
            return False
    else:
        print(f"❌ JSON download failed: {download_result.get('error')}")
        return False
    
    # Test 3: List files
    print("\n📋 Listing files...")
    list_result = await storage.list_files("test/", include_metadata=True)
    
    if list_result["success"]:
        files = list_result["files"]
        print(f"✅ Found {len(files)} files in 'test/' prefix:")
        for file_info in files:
            print(f"   - {file_info['key']} ({file_info['size']} bytes)")
    else:
        print(f"❌ List files failed: {list_result.get('error')}")
        return False
    
    # Test 4: Upload a file
    test_file = Path("/tmp/ada_wasabi_test.txt")
    test_file.write_text("Ada Cloud Wasabi file test\nThis is a test file for S3 upload.\n")
    
    print(f"\n📤 Uploading test file: {test_file}")
    file_result = await storage.upload_file(test_file, "test/file_upload.txt")
    
    if file_result["success"]:
        print(f"✅ File uploaded to: {file_result['key']}")
        print(f"   Size: {file_result['size']} bytes")
    else:
        print(f"❌ File upload failed: {file_result.get('error')}")
        return False
    
    # Clean up test file
    if test_file.exists():
        test_file.unlink()
    
    # Test 5: Delete files
    print("\n🗑️  Cleaning up test files...")
    
    delete_json = await storage.delete_file("test/wasabi_test.json")
    if delete_json["success"]:
        print("✅ Test JSON deleted")
    else:
        print(f"⚠️  Failed to delete test JSON: {delete_json.get('error')}")
    
    delete_file = await storage.delete_file("test/file_upload.txt")
    if delete_file["success"]:
        print("✅ Test file deleted")
    else:
        print(f"⚠️  Failed to delete test file: {delete_file.get('error')}")
    
    return True

async def test_bucket_creation():
    """Test bucket creation functionality."""
    print("\n🪣 Testing bucket operations...")
    
    storage = get_storage_service()
    
    # The _ensure_bucket_exists() is called during initialization
    # Let's verify we can write to the bucket
    test_result = await storage.upload_json("bucket_test/verify.json", {"verified": True})
    
    if test_result["success"]:
        print(f"✅ Successfully wrote to bucket '{storage.config.bucket_name}'")
        
        # Clean up
        await storage.delete_file("bucket_test/verify.json")
    else:
        print(f"❌ Failed to write to bucket: {test_result.get('error')}")
        return False
    
    return True

async def main():
    """Run all tests."""
    print("🧪 Ada Cloud Wasabi S3 Connection Test\n")
    
    # Check environment variables
    required_vars = ["WASABI_KEY_ID", "WASABI_SECRET", "WASABI_ENDPOINT"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * (len(value) - 4) + value[-4:] if len(value) > 4 else 'SET'}")
        else:
            print(f"❌ {var}: MISSING")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file or environment")
        return 1
    
    print(f"\n🔧 Configuration:")
    config = StorageService().config
    print(f"   Endpoint: {config.endpoint_url}")
    print(f"   Bucket: {config.bucket_name}")
    print(f"   Region: {config.region_name}")
    
    # Run tests
    tests = [
        ("Connection Test", test_wasabi_connection),
        ("Storage Operations", test_storage_operations),
        ("Bucket Operations", test_bucket_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        try:
            if await test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! Wasabi S3 is properly configured.")
        return 0
    else:
        print("❌ Some tests failed. Check the configuration above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
