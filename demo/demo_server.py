#!/usr/bin/env python3
"""
SHIM: Redirects execution to hypersplat.services.api.server
"""
import os
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
        port = int(os.environ.get("HYPERSPLAT_SERVER_PORT", 8081))
        print(f"\n\033[92mServer is live! Access it here: http://localhost:{port}\033[0m\n")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)
