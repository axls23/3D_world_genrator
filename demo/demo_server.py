#!/usr/bin/env python3
"""
SHIM: Redirects execution to hypersplat.services.api.server
"""
import sys
import subprocess
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    print("Redirecting to hypersplat.services.api.server...")
    
    try:
        import uvicorn
        # Import the app directly. This benefits from the sys.path.insert above.
        from hypersplat.services.api.server import app
        
        # Run uvicorn programmatically
        print("\n\033[92mServer is live! Access it here: http://localhost:8081\033[0m\n")
        uvicorn.run(app, host="0.0.0.0", port=8081)
    except ImportError as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)
