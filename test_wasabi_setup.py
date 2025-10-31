#!/usr/bin/env python3
"""Test script to verify Wasabi S3 setup."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cloud.storage_service import StorageService, StorageConfig

def test_wasabi_connection():
    """Test Wasabi connection and bucket creation."""
    print("🔍 Testing Wasabi S3 connection...")
    
    # Check environment variables
    required_vars = ["WASABI_KEY_ID", "WASABI_SECRET"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these environment variables:")
        for var in missing_vars:
            print(f"  export {var}='your-value'")
        return False
    
    try:
        # Create storage service
        print("📡 Initializing storage service...")
        config = StorageConfig.from_env()
        service = StorageService(config)
        
        if service._initialized:
            print(f"✅ Successfully connected to Wasabi at {config.endpoint_url}")
            print(f"📦 Bucket: {config.bucket_name}")
            print(f"🌐 Region: {config.region_name}")
            
            # Test a simple upload/download
            test_content = b"Test file from Ada Cloud"
            test_file = Path("/tmp/ada_test_file.txt")
            test_file.write_bytes(test_content)
            
            print("📤 Testing file upload...")
            upload_result = await service.upload_file(test_file, "test/ada_test.txt")
            
            if upload_result.get("success"):
                print("✅ File upload successful")
                
                # Test download
                print("📥 Testing file download...")
                download_path = "/tmp/ada_downloaded_test.txt"
                download_result = await service.download_file("test/ada_test.txt", download_path)
                
                if download_result.get("success"):
                    print("✅ File download successful")
                    
                    # Verify content
                    if Path(download_path).exists():
                        downloaded_content = Path(download_path).read_bytes()
                        if downloaded_content == test_content:
                            print("✅ Content verification successful")
                        else:
                            print("❌ Content verification failed")
                    else:
                        print("❌ Downloaded file not found")
                else:
                    print(f"❌ Download failed: {download_result.get('error')}")
            else:
                print(f"❌ Upload failed: {upload_result.get('error')}")
            
            # Cleanup
            cleanup_result = await service.delete_file("test/ada_test.txt")
            if cleanup_result.get("success"):
                print("🧹 Test files cleaned up")
            
            return True
        else:
            print("❌ Storage service not initialized - check credentials")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    
    # Run the async test
    success = asyncio.run(test_wasabi_connection())
    
    if success:
        print("\n🎉 Wasabi S3 setup is working correctly!")
        print("You can now run 'make deploy-cloud' and 'make sync-storage'")
    else:
        print("\n❌ Wasabi setup failed. Please check your credentials and configuration.")
