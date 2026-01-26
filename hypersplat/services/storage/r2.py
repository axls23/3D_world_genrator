#!/usr/bin/env python3
"""
Cloudflare R2 Cloud Storage Uploader for GSplat Outputs

S3-compatible storage integration using boto3.
Free tier: 10GB storage, no egress fees.

Usage:
    # As module
    from r2_uploader import R2Uploader
    uploader = R2Uploader()
    uploader.upload_directory("./results/acezero_3dgs", "scene_001")
    
    # CLI
    python r2_uploader.py ./results/acezero_3dgs scene_001
"""

import os
import json
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class R2Config:
    """R2 Configuration from environment variables"""
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    
    @classmethod
    def from_env(cls) -> Optional["R2Config"]:
        """Load config from environment. Returns None if not configured."""
        account_id = os.getenv("R2_ACCOUNT_ID")
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        bucket = os.getenv("R2_BUCKET_NAME", "gsplat-outputs")
        
        if not all([account_id, access_key, secret_key]):
            return None
        
        return cls(
            account_id=account_id,
            access_key_id=access_key,
            secret_access_key=secret_key,
            bucket_name=bucket
        )
    
    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


class R2Uploader:
    """
    Cloudflare R2 uploader with S3-compatible API.
    
    Gracefully handles missing credentials - logs warning and skips upload.
    """
    
    def __init__(self, config: Optional[R2Config] = None):
        self.config = config or R2Config.from_env()
        self._client = None
        
    @property
    def is_configured(self) -> bool:
        return self.config is not None
    
    @property
    def client(self):
        """Lazy-load boto3 client"""
        if self._client is None:
            if not self.is_configured:
                raise RuntimeError("R2 not configured. Set environment variables.")
            
            try:
                import boto3
                from botocore.config import Config
            except ImportError:
                raise ImportError("boto3 required. Install: pip install boto3")
            
            self._client = boto3.client(
                's3',
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                config=Config(
                    retries={'max_attempts': 3, 'mode': 'adaptive'},
                    signature_version='s3v4'
                )
            )
        return self._client
    
    def upload_file(
        self, 
        local_path: Path, 
        remote_key: str,
        content_type: Optional[str] = None
    ) -> bool:
        """
        Upload a single file to R2.
        
        Args:
            local_path: Path to local file
            remote_key: Key (path) in R2 bucket
            content_type: MIME type (auto-detected if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured:
            logger.warning("R2 not configured. Skipping upload.")
            return False
        
        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return False
        
        # Auto-detect content type
        if content_type is None:
            content_type, _ = mimetypes.guess_type(str(local_path))
            content_type = content_type or 'application/octet-stream'
        
        try:
            extra_args = {'ContentType': content_type}
            
            self.client.upload_file(
                str(local_path),
                self.config.bucket_name,
                remote_key,
                ExtraArgs=extra_args
            )
            logger.info(f"Uploaded: {local_path.name} -> r2://{self.config.bucket_name}/{remote_key}")
            return True
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def upload_directory(
        self,
        local_dir: Path,
        remote_prefix: str,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Upload entire directory to R2.
        
        Args:
            local_dir: Local directory path
            remote_prefix: Prefix for all uploaded files (e.g., "scene_001")
            extensions: File extensions to include (e.g., [".ply", ".pt"]). None = all files.
            
        Returns:
            Dict with 'uploaded' and 'failed' counts
        """
        if not self.is_configured:
            logger.warning("R2 not configured. Skipping directory upload.")
            return {"uploaded": 0, "failed": 0, "skipped": True}
        
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            logger.error(f"Not a directory: {local_dir}")
            return {"uploaded": 0, "failed": 1}
        
        stats = {"uploaded": 0, "failed": 0}
        
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            
            # Filter by extension if specified
            if extensions and file_path.suffix.lower() not in extensions:
                continue
            
            # Skip very large files (>500MB) with warning
            if file_path.stat().st_size > 500 * 1024 * 1024:
                logger.warning(f"Skipping large file: {file_path} (>500MB)")
                continue
            
            relative_path = file_path.relative_to(local_dir)
            remote_key = f"{remote_prefix}/{relative_path}".replace("\\", "/")
            
            if self.upload_file(file_path, remote_key):
                stats["uploaded"] += 1
            else:
                stats["failed"] += 1
        
        logger.info(f"Directory upload complete: {stats['uploaded']} uploaded, {stats['failed']} failed")
        return stats
    
    def upload_gsplat_results(
        self,
        result_dir: Path,
        scene_name: str,
        include_checkpoints: bool = False
    ) -> Dict:
        """
        Upload GSplat training results with smart filtering.
        
        Uploads:
        - Final .ply point cloud
        - Metadata/config files
        - Optionally: checkpoints
        
        Args:
            result_dir: Training result directory
            scene_name: Scene identifier for remote path
            include_checkpoints: Whether to upload .pt checkpoint files
        """
        result_dir = Path(result_dir)
        
        # Priority files to upload
        priority_extensions = [".ply", ".json", ".txt"]
        if include_checkpoints:
            priority_extensions.append(".pt")
        
        return self.upload_directory(
            result_dir,
            scene_name,
            extensions=priority_extensions
        )


def setup_r2_interactive():
    """Interactive setup wizard for R2 credentials"""
    print("\n" + "="*60)
    print("  CLOUDFLARE R2 SETUP WIZARD")
    print("="*60)
    print("\nYou'll need to create an R2 bucket at:")
    print("  https://dash.cloudflare.com/ -> R2 Object Storage\n")
    
    account_id = input("Enter your R2 Account ID: ").strip()
    access_key = input("Enter your R2 Access Key ID: ").strip()
    secret_key = input("Enter your R2 Secret Access Key: ").strip()
    bucket = input("Enter bucket name [gsplat-outputs]: ").strip() or "gsplat-outputs"
    
    if not all([account_id, access_key, secret_key]):
        print("\n❌ All credentials are required!")
        return False
    
    # Generate .env content
    env_content = f"""# Cloudflare R2 Configuration
R2_ACCOUNT_ID={account_id}
R2_ACCESS_KEY_ID={access_key}
R2_SECRET_ACCESS_KEY={secret_key}
R2_BUCKET_NAME={bucket}
"""
    
    # Save to .env file
    env_path = Path(__file__).parent.parent.parent / ".env.r2"
    env_path.write_text(env_content)
    
    print(f"\n✅ Configuration saved to: {env_path}")
    print("\nTo activate, run:")
    print(f"  # Windows (PowerShell)")
    print(f"  Get-Content {env_path} | ForEach-Object {{ $_ -match '^([^=]+)=(.*)$' | Out-Null; [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }}")
    print(f"\n  # Linux/WSL")
    print(f"  export $(cat {env_path} | xargs)")
    
    # Test connection
    print("\nTesting connection...")
    config = R2Config(
        account_id=account_id,
        access_key_id=access_key,
        secret_access_key=secret_key,
        bucket_name=bucket
    )
    
    try:
        uploader = R2Uploader(config)
        uploader.client.head_bucket(Bucket=bucket)
        print(f"✅ Successfully connected to bucket: {bucket}")
        return True
    except Exception as e:
        print(f"⚠️  Connection test failed: {e}")
        print("   (This might be OK if the bucket doesn't exist yet)")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cloudflare R2 Uploader for GSplat")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Interactive R2 setup")
    
    # Upload command  
    upload_parser = subparsers.add_parser("upload", help="Upload directory to R2")
    upload_parser.add_argument("local_dir", help="Local directory to upload")
    upload_parser.add_argument("scene_name", help="Scene name for remote path")
    upload_parser.add_argument("--with-checkpoints", action="store_true", help="Include .pt files")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test R2 connection")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.command == "setup":
        setup_r2_interactive()
    
    elif args.command == "upload":
        uploader = R2Uploader()
        if not uploader.is_configured:
            print("❌ R2 not configured. Run: python r2_uploader.py setup")
        else:
            uploader.upload_gsplat_results(
                args.local_dir,
                args.scene_name,
                include_checkpoints=args.with_checkpoints
            )
    
    elif args.command == "test":
        uploader = R2Uploader()
        if not uploader.is_configured:
            print("❌ R2 not configured. Run: python r2_uploader.py setup")
        else:
            try:
                uploader.client.list_buckets()
                print("✅ R2 connection successful!")
            except Exception as e:
                print(f"❌ Connection failed: {e}")
    
    else:
        parser.print_help()
