"""
Blind Spot Detector for GeNVS-Vector Pipeline

Analyzes camera trajectory from ACE-Zero output to identify:
1. Spherical coverage gaps (blind spots)
2. Recommended view angles for novel synthesis
3. Coverage quality score

Used to programmatically determine where Zero123 should generate views.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json


class BlindSpotDetector:
    """
    Analyzes camera pose coverage to find missing viewing angles.
    
    Converts camera extrinsics to spherical coordinates and finds
    gaps in the azimuth/elevation coverage.
    """
    
    def __init__(
        self,
        azimuth_bins: int = 36,  # 10° resolution
        elevation_bins: int = 18,  # 10° resolution
        coverage_threshold: float = 0.3  # Minimum coverage to not be a blind spot
    ):
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins
        self.coverage_threshold = coverage_threshold
        
    def _load_poses_from_colmap(self, colmap_dir: Path) -> np.ndarray:
        """Load camera positions from COLMAP sparse output."""
        images_path = colmap_dir / "sparse" / "0" / "images.bin"
        images_txt = colmap_dir / "sparse" / "0" / "images.txt"
        
        positions = []
        
        # Try binary format first
        if images_path.exists():
            try:
                from scripts.genvs.utils.colmap_read_write_model import read_images_binary
                images = read_images_binary(str(images_path))
                for img in images.values():
                    # Extract camera position from quaternion + translation
                    qvec = img.qvec
                    tvec = img.tvec
                    # Camera position = -R^T @ t
                    R = self._qvec_to_rotmat(qvec)
                    pos = -R.T @ tvec
                    positions.append(pos)
            except Exception as e:
                print(f"[BlindSpot] Failed to read binary: {e}")
                
        # Fallback to text format
        elif images_txt.exists():
            try:
                with open(images_txt, 'r') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.startswith('#') or not line.strip():
                        continue
                    if i % 2 == 0:  # Image lines are every other line
                        parts = line.strip().split()
                        if len(parts) >= 8:
                            qw, qx, qy, qz = map(float, parts[1:5])
                            tx, ty, tz = map(float, parts[5:8])
                            R = self._qvec_to_rotmat([qw, qx, qy, qz])
                            pos = -R.T @ np.array([tx, ty, tz])
                            positions.append(pos)
            except Exception as e:
                print(f"[BlindSpot] Failed to read text: {e}")
        
        return np.array(positions) if positions else np.array([])
    
    def _qvec_to_rotmat(self, qvec):
        """Convert quaternion to rotation matrix."""
        qw, qx, qy, qz = qvec
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        return R
    
    def _positions_to_spherical(
        self, 
        positions: np.ndarray,
        center: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert camera positions to spherical coordinates relative to scene center."""
        if center is None:
            center = positions.mean(axis=0)
        
        # Relative positions
        rel = positions - center
        
        # Convert to spherical (azimuth, elevation)
        r = np.linalg.norm(rel, axis=1)
        azimuth = np.arctan2(rel[:, 1], rel[:, 0])  # [-pi, pi]
        elevation = np.arcsin(rel[:, 2] / (r + 1e-8))  # [-pi/2, pi/2]
        
        # Convert to degrees
        azimuth_deg = np.degrees(azimuth) + 180  # [0, 360]
        elevation_deg = np.degrees(elevation) + 90  # [0, 180]
        
        return np.stack([azimuth_deg, elevation_deg], axis=1)
    
    def analyze(
        self,
        colmap_dir: Path,
        scene_center: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Analyze pose coverage and find blind spots.
        
        Args:
            colmap_dir: Path to COLMAP/ACE-Zero output
            scene_center: Optional scene center override
            
        Returns:
            Dict with coverage analysis and recommended views
        """
        positions = self._load_poses_from_colmap(Path(colmap_dir))
        
        if len(positions) == 0:
            return {
                "error": "No poses found",
                "coverage_score": 0,
                "blind_spots": [],
                "recommended_views": []
            }
        
        # Convert to spherical
        spherical = self._positions_to_spherical(positions, scene_center)
        
        # Create coverage histogram
        az_bins = np.linspace(0, 360, self.azimuth_bins + 1)
        el_bins = np.linspace(0, 180, self.elevation_bins + 1)
        
        coverage, _, _ = np.histogram2d(
            spherical[:, 0], spherical[:, 1],
            bins=[az_bins, el_bins]
        )
        
        # Normalize
        coverage_norm = coverage / (coverage.max() + 1e-8)
        
        # Find blind spots (low coverage areas)
        blind_spots = []
        recommended_views = []
        
        for i in range(self.azimuth_bins):
            for j in range(self.elevation_bins):
                if coverage_norm[i, j] < self.coverage_threshold:
                    az_center = (az_bins[i] + az_bins[i+1]) / 2
                    el_center = (el_bins[j] + el_bins[j+1]) / 2
                    
                    # Convert back to standard coords
                    az_deg = az_center - 180  # [-180, 180]
                    el_deg = el_center - 90   # [-90, 90]
                    
                    blind_spots.append({
                        "azimuth": float(az_deg),
                        "elevation": float(el_deg),
                        "coverage": float(coverage_norm[i, j])
                    })
        
        # Sort by coverage (worst first)
        blind_spots.sort(key=lambda x: x["coverage"])
        
        # Top N recommendations (most important blind spots)
        recommended_views = [(b["azimuth"], b["elevation"]) for b in blind_spots[:10]]
        
        # Overall coverage score
        covered_bins = (coverage_norm >= self.coverage_threshold).sum()
        total_bins = self.azimuth_bins * self.elevation_bins
        coverage_score = covered_bins / total_bins
        
        return {
            "num_cameras": len(positions),
            "scene_center": scene_center.tolist() if scene_center is not None else positions.mean(axis=0).tolist(),
            "coverage_score": float(coverage_score),
            "num_blind_spots": len(blind_spots),
            "blind_spots": blind_spots,
            "recommended_views": recommended_views,
            "coverage_matrix_shape": coverage.shape
        }
    
    def get_zero123_angles(
        self,
        colmap_dir: Path,
        num_views: int = 6
    ) -> List[Tuple[float, float]]:
        """
        Get recommended azimuth/elevation angles for Zero123 generation.
        
        Args:
            colmap_dir: COLMAP output directory
            num_views: Number of views to generate
            
        Returns:
            List of (azimuth, elevation) tuples
        """
        analysis = self.analyze(colmap_dir)
        
        if "error" in analysis:
            # Fallback to default back/side views
            return [
                (180, 0),   # Back
                (90, 0),    # Right
                (270, 0),   # Left
                (180, 30),  # Back-up
                (180, -30), # Back-down
                (45, 0)     # Front-right
            ][:num_views]
        
        recommended = analysis["recommended_views"][:num_views]
        
        # If not enough blind spots, add evenly spaced views
        while len(recommended) < num_views:
            az = 360 * len(recommended) / num_views
            recommended.append((az - 180, 0))
        
        return recommended


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze camera coverage blind spots")
    parser.add_argument("--colmap_dir", type=str, required=True, help="COLMAP output dir")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()
    
    detector = BlindSpotDetector()
    analysis = detector.analyze(args.colmap_dir)
    
    print(f"\n=== Coverage Analysis ===")
    print(f"Cameras: {analysis.get('num_cameras', 0)}")
    print(f"Coverage Score: {analysis.get('coverage_score', 0):.1%}")
    print(f"Blind Spots: {analysis.get('num_blind_spots', 0)}")
    print(f"\nRecommended Views:")
    for az, el in analysis.get("recommended_views", [])[:5]:
        print(f"  Azimuth: {az:.1f}°, Elevation: {el:.1f}°")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved to: {args.output}")
