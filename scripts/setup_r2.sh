#!/bin/bash
# ============================================
#  Cloudflare R2 Setup Script for GSplat
#  Run this to configure cloud storage
# ============================================

echo ""
echo "================================================"
echo "  CLOUDFLARE R2 CLOUD STORAGE SETUP"
echo "================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Install boto3 if not present
echo "Checking boto3..."
python3 -c "import boto3" 2>/dev/null || {
    echo "Installing boto3..."
    pip install boto3
}

# Run interactive setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/pipeline/r2_uploader.py" setup
