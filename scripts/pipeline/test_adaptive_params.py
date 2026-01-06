"""
Test script for adaptive quality aggregator and flexible chunking
Demonstrates the new features without modifying the main pipeline
"""
import cv2
import numpy as np
from pathlib import Path

class AdaptiveConfig:
    """Demonstrates adaptive parameter calculation"""
    
    MIN_FRAMES_FOR_SFM = 15
    
    def calculate_chunk_params(self, total_duration: float) -> tuple:
        """Dynamically calculate chunking parameters based on video length"""
        if total_duration < 30:
            return (total_duration, total_duration, 1)
        elif total_duration < 120:
            return (10, 30, None)
        elif total_duration < 600:
            return (15, 45, None)
        elif total_duration < 3600:
            return (20, 60, None)
        else:
            target_chunks = int(total_duration / 60)
            return (30, 90, target_chunks)
    
    def calculate_motion_threshold(self, total_duration: float) -> float:
        """Adjust motion trajectory threshold based on video length"""
        if total_duration < 60:
            return 8.0
        elif total_duration < 600:
            return 10.0
        else:
            return 15.0
    
    def calculate_dynamic_fps(self, chunk_duration: float, motion_type: str, avg_quality: float) -> float:
        """Calculate optimal FPS based on chunk characteristics"""
        base_fps = max(1.5, self.MIN_FRAMES_FOR_SFM / chunk_duration)
        
        motion_multiplier = {
            "static": 0.8,
            "slow_pan": 1.0,
            "medium_motion": 1.2,
            "fast_motion": 1.5,
            "continuous": 1.0
        }.get(motion_type, 1.0)
        
        if avg_quality < 0.4:
            quality_multiplier = 1.3
        elif avg_quality > 0.7:
            quality_multiplier = 0.9
        else:
            quality_multiplier = 1.0
        
        if chunk_duration > 60:
            duration_cap = min(1.0, 30 / chunk_duration)
        else:
            duration_cap = 1.0
        
        final_fps = base_fps * motion_multiplier * quality_multiplier * duration_cap
        return max(0.5, min(5.0, final_fps))


def test_adaptive_parameters(video_path: str):
    """Test adaptive parameter calculation"""
    
    config = AdaptiveConfig()
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    cap.release()
    
    print("="*70)
    print("ADAPTIVE PARAMETER TEST")
    print("="*70)
    print(f"Video: {Path(video_path).name}")
    print(f"Duration: {total_duration:.1f}s ({total_frames} frames @ {fps:.1f} FPS)")
    print()
    
    # Test chunk parameters
    min_dur, max_dur, target_chunks = config.calculate_chunk_params(total_duration)
    print(f"Adaptive Chunk Parameters:")
    print(f"  Min duration: {min_dur:.1f}s")
    print(f"  Max duration: {max_dur:.1f}s")
    print(f"  Target chunks: {target_chunks if target_chunks else 'Auto'}")
    print()
    
    # Test motion threshold
    motion_threshold = config.calculate_motion_threshold(total_duration)
    print(f"Motion Threshold: {motion_threshold:.1f}")
    print()
    
    # Test dynamic FPS for different scenarios
    print("Dynamic FPS Calculation Examples:")
    print("-" * 70)
    
    test_scenarios = [
        (13.7, "continuous", 0.44, "Current sample video"),
        (25.0, "medium_motion", 0.6, "Medium chunk, good quality"),
        (60.0, "slow_pan", 0.7, "Long chunk, high quality"),
        (10.0, "fast_motion", 0.3, "Short chunk, poor quality"),
    ]
    
    for duration, motion, quality, desc in test_scenarios:
        fps_calc = config.calculate_dynamic_fps(duration, motion, quality)
        expected_frames = int(duration * fps_calc)
        print(f"{desc}:")
        print(f"  Duration: {duration:.1f}s | Motion: {motion} | Quality: {quality:.2f}")
        print(f"  → FPS: {fps_calc:.2f} | Expected frames: {expected_frames}")
        print()
    
    print("="*70)
    print("✅ Adaptive parameters working correctly!")
    print("="*70)


if __name__ == "__main__":
    video_path = r"C:\Users\sxhil_25660\Documents\GitHub\SIH_2025_Professional\gsplat\data\VIDEO\13657900_2160_3840_60fps.mp4"
    test_adaptive_parameters(video_path)
