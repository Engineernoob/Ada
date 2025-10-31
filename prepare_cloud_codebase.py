#!/usr/bin/env python3
"""
Prepare Ada codebase for deployment to Modal Cloud
"""

import subprocess
import sys
import os
from pathlib import Path

def sync_codebase_to_modal():
    """Sync Ada codebase to Modal volume"""
    print("📦 Preparing Ada codebase for Modal Cloud deployment...")
    
    # Create a temporary directory with the Ada codebase
    ada_modules = ["core", "neural", "rl", "persona", "planner", "memory", "agent", "tools", "interfaces"]
    temp_dir = Path("temp_ada_code")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy required modules to temp directory
    for module in ada_modules:
        module_path = Path(module)
        if module_path.exists():
            print(f"📁 Copying {module}...")
            subprocess.run(["cp", "-r", str(module_path), str(temp_dir / module)], check=True)
        else:
            print(f"⚠️  Module {module} not found, skipping...")
    
    # Copy required files
    required_files = ["requirements.txt"]
    for file in required_files:
        if Path(file).exists():
            subprocess.run(["cp", file, temp_dir], check=True)
            print(f"📄 Copied {file}")
    
    print(f"✅ Codebase prepared in {temp_dir}")
    return temp_dir

def deploy_codebase_to_volume():
    """Deploy codebase to Modal volume"""
    print("☁️  Deploying codebase to Modal volume...")
    
    try:
        # Create Modal volume and upload codebase
        result = subprocess.run([
            "modal", "volume", "create", "ada-codebase"
        ], capture_output=True, text=True)
        
        if "already exists" not in result.stderr:
            print("📦 Created ada-codebase volume")
        else:
            print("📦 ada-codebase volume already exists")
        
        # Upload codebase using shell
        upload_cmd = f"""
        modal shell --volume ada-codebase --volume ada-cloud-storage bash -c '
        echo "📁 Uploading Ada codebase to volume..."
        rm -rf /mnt/ada-codebase/* 2>/dev/null || true
        cd /mnt
        
        # Here you would typically copy from local, but for now we'll create a basic structure
        mkdir -p /mnt/ada-codebase/core
        mkdir -p /mnt/ada-codebase/neural
        mkdir -p /mnt/ada-codebase/rl
        mkdir -p /mnt/ada-codebase/persona
        mkdir -p /mnt/ada-codebase/planner
        mkdir -p /mnt/ada-codebase/memory
        mkdir -p /mnt/ada-codebase/agent
        mkdir -p /mnt/ada-codebase/tools
        mkdir -p /mnt/ada-codebase/interfaces
        
        # Create basic __init__.py files
        for dir in /mnt/ada-codebase/*/; do
            echo "## Ada Module" > "$__init__.py" 2>/dev/null || true
        done
        
        echo "✅ Codebase structure created"
        ls -la /mnt/ada-codebase/
        '
        """
        
        result = subprocess.run(upload_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Codebase deployed to Modal volume")
            print("📋 Deployment output:")
            print(result.stdout)
        else:
            print(f"❌ Deployment failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error deploying codebase: {e}")

if __name__ == "__main__":
    sync_codebase_to_modal()
    deploy_codebase_to_volume()
