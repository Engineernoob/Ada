"""Simple Modal deployment for testing connectivity."""

import modal

# Create a simple app for testing
app = modal.App("AdaCloudSimple")

# Simple image with minimal dependencies
simple_image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "fastapi",
])

@app.function(image=simple_image)
def simple_test():
    """Simple test function to verify Modal deployment."""
    return {
        "message": "Ada Cloud deployment successful!",
        "timestamp": "2024",
        "status": "healthy"
    }

@app.function(image=simple_image)
@modal.fastapi_endpoint(method="GET")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ada-cloud"}
