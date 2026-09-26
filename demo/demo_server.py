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

    # Resolve the port before the import/run try block so a misconfigured
    # env var gets a clear error instead of an unhandled traceback.
    _raw_port = os.environ.get("HYPERSPLAT_SERVER_PORT", "8081")
    try:
        port = int(_raw_port)
    except ValueError:
        print(f"Invalid HYPERSPLAT_SERVER_PORT value: {_raw_port!r}; must be an integer.")
        sys.exit(1)

    try:
        import uvicorn
        # Import the app directly. This benefits from the sys.path.insert above.
        from hypersplat.services.api.server import app

        # Run uvicorn programmatically
        print(f"\n\033[92mServer is live! Access it here: http://localhost:{port}\033[0m\n")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)
