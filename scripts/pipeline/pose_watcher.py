"""
Pose Watcher Service for Live Streaming Pipeline

This module monitors for new pose files from ACE-Zero and signals the 
gsplat trainer to hot-reload new images/poses during training.
"""

import json
import time
import threading
import logging
from pathlib import Path
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)


class PoseWatcher:
    """Watches for new pose files from ACE-Zero and triggers callbacks."""
    
    def __init__(
        self, 
        streaming_status_file: Path,
        on_new_poses: Optional[Callable[[Dict[str, Any]], None]] = None,
        poll_interval: float = 2.0
    ):
        """
        Initialize the pose watcher.
        
        Args:
            streaming_status_file: Path to the streaming_status.json file
            on_new_poses: Callback function when new poses are detected
            poll_interval: How often to check for new poses (seconds)
        """
        self.streaming_status_file = Path(streaming_status_file)
        self.on_new_poses = on_new_poses
        self.poll_interval = poll_interval
        
        self.last_seen_iteration = -1
        self.last_seen_timestamp = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the watcher in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"[PoseWatcher] Started watching: {self.streaming_status_file}")
        
    def stop(self):
        """Stop the watcher thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("[PoseWatcher] Stopped.")
        
    def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                status = self._read_status()
                if status and self._is_new_update(status):
                    self._handle_new_status(status)
            except Exception as e:
                logger.warning(f"[PoseWatcher] Poll error: {e}")
            time.sleep(self.poll_interval)
            
    def _read_status(self) -> Optional[Dict[str, Any]]:
        """Read the streaming status file."""
        if not self.streaming_status_file.exists():
            return None
        try:
            with open(self.streaming_status_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
            
    def _is_new_update(self, status: Dict[str, Any]) -> bool:
        """Check if this is a new update we haven't seen before."""
        iteration = status.get("iteration", -1)
        timestamp = status.get("timestamp", 0.0)
        
        if iteration > self.last_seen_iteration:
            return True
        if iteration == self.last_seen_iteration and timestamp > self.last_seen_timestamp:
            return True
        return False
        
    def _handle_new_status(self, status: Dict[str, Any]):
        """Handle a new status update."""
        iteration = status.get("iteration", -1)
        timestamp = status.get("timestamp", 0.0)
        pose_file = status.get("pose_file", "")
        is_running = status.get("is_running", True)
        
        self.last_seen_iteration = iteration
        self.last_seen_timestamp = timestamp
        
        logger.info(f"[PoseWatcher] New poses detected - Iteration {iteration}, File: {pose_file}")
        
        if self.on_new_poses:
            self.on_new_poses(status)
            
        if not is_running:
            logger.info("[PoseWatcher] ACE-Zero finished. Stopping watcher.")
            self._running = False
            
    def check_once(self) -> Optional[Dict[str, Any]]:
        """Synchronous check for new poses (for use in training loop)."""
        try:
            status = self._read_status()
            if status and self._is_new_update(status):
                self._handle_new_status(status)
                return status
        except Exception as e:
            logger.warning(f"[PoseWatcher] Check error: {e}")
        return None
        
    def has_new_data(self) -> bool:
        """Quick check if there's new data available."""
        try:
            status = self._read_status()
            if status:
                return self._is_new_update(status)
        except Exception:
            pass
        return False
        
    @property
    def is_ace_zero_running(self) -> bool:
        """Check if ACE-Zero is still running."""
        try:
            status = self._read_status()
            if status:
                return status.get("is_running", False)
        except Exception:
            pass
        return False


class StreamingDataManager:
    """Manages streaming data updates for gsplat training."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the streaming data manager.
        
        Args:
            output_dir: The ACE-Zero output directory containing streaming_status.json
        """
        self.output_dir = Path(output_dir)
        self.streaming_status_file = self.output_dir / "streaming_status.json"
        self.watcher = PoseWatcher(self.streaming_status_file)
        
        self.current_iteration = -1
        self.images_ready = False
        
    def start_watching(self, callback: Optional[Callable] = None):
        """Start watching for new poses."""
        if callback:
            self.watcher.on_new_poses = callback
        self.watcher.start()
        
    def stop_watching(self):
        """Stop watching for new poses."""
        self.watcher.stop()
        
    def check_for_updates(self) -> Optional[Dict[str, Any]]:
        """Check for updates synchronously (call from training loop)."""
        return self.watcher.check_once()
        
    def get_latest_sparse_dir(self) -> Optional[Path]:
        """Get the path to the latest sparse directory with COLMAP data."""
        try:
            status = self.watcher._read_status()
            if status:
                sparse_dir = status.get("sparse_dir")
                if sparse_dir and Path(sparse_dir).exists():
                    return Path(sparse_dir)
        except Exception:
            pass
        return None
        
    def get_latest_images_dir(self) -> Optional[Path]:
        """Get the path to the latest images directory."""
        try:
            status = self.watcher._read_status()
            if status:
                images_dir = status.get("images_dir")
                if images_dir and Path(images_dir).exists():
                    return Path(images_dir)
        except Exception:
            pass
        return None


class PoseDataBuffer:
    """Buffer with retry logic for robust pose data fetching from post-training step."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the pose data buffer.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (exponential backoff applied)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cache: Dict[str, Any] = {}
        self._last_fetch_time: float = 0.0
        
    def fetch_with_retry(self, pose_file: Path, parse_func: Callable[[Path], Dict]) -> Optional[Dict]:
        """
        Fetch pose data with retry logic.
        
        Args:
            pose_file: Path to the pose file
            parse_func: Function to parse the pose file
            
        Returns:
            Parsed pose data or None if all retries failed
        """
        cache_key = str(pose_file)
        
        # Return cached if available
        if cache_key in self._cache:
            logger.debug(f"[PoseBuffer] Returning cached data for {pose_file.name}")
            return self._cache[cache_key]
        
        for attempt in range(self.max_retries):
            try:
                if not pose_file.exists():
                    logger.warning(f"[PoseBuffer] File not found: {pose_file} (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                    
                data = parse_func(pose_file)
                if data:
                    self._cache[cache_key] = data
                    self._last_fetch_time = time.time()
                    logger.info(f"[PoseBuffer] Successfully fetched {pose_file.name}")
                    return data
                    
            except Exception as e:
                logger.warning(f"[PoseBuffer] Fetch error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
        logger.error(f"[PoseBuffer] All {self.max_retries} retries failed for {pose_file}")
        return None
        
    def clear_cache(self):
        """Clear the cached pose data."""
        self._cache.clear()
        
    def get_cached(self, pose_file: Path) -> Optional[Dict]:
        """Get cached data without fetching."""
        return self._cache.get(str(pose_file))


if __name__ == "__main__":
    # Test the pose watcher
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("watch_dir", help="Directory to watch for streaming_status.json")
    args = parser.parse_args()
    
    def on_new_poses(status):
        print(f"New poses: {status}")
    
    logging.basicConfig(level=logging.INFO)
    
    watcher = PoseWatcher(
        Path(args.watch_dir) / "streaming_status.json",
        on_new_poses=on_new_poses
    )
    watcher.start()
    
    try:
        while watcher.is_ace_zero_running or watcher._running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
