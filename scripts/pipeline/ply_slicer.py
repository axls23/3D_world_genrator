#!/usr/bin/env python3
"""
PlySlicer: Efficiently slices massive .ply files into spatial chunks.
Enhanced version with numpy support, robust logging, and detailed reporting.
"""

import os
import math
import json
import shutil
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PlySlicerConfig:
    """Configuration for PlySlicer"""
    def __init__(self, 
                 chunk_size: float = 10.0,
                 output_dir: str = "output_chunks",
                 min_points_per_chunk: int = 100):
        self.chunk_size = chunk_size
        self.output_dir = Path(output_dir)
        self.min_points_per_chunk = min_points_per_chunk

class PlySlicer:
    """
    Slices massive .ply files into spatial chunks (voxels/octree blocks)
    for efficient streaming.
    """
    def __init__(self, config: PlySlicerConfig):
        self.config = config
        self.stats = {
            "total_points": 0,
            "chunks_created": 0,
            "processing_time": 0,
            "bbox": {"min": None, "max": None}
        }

    def slice_file(self, input_path: str) -> Optional[str]:
        """
        Reads a PLY file and partitions points into chunks using numpy.
        """
        start_time = time.time()
        input_path = Path(input_path)
        logger.info(f"🔪 Slicing {input_path} into {self.config.chunk_size}m chunks...")

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return None

        # Create output directory
        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Read Header and Data
            header, vertex_data, property_names = self._read_ply(input_path)
            if vertex_data is None:
                return None
            
            self.stats["total_points"] = len(vertex_data)
            logger.info(f"Loaded {len(vertex_data)} points. analyzing spatial distribution...")

            # 2. Calculate Bounding Box
            coords = vertex_data[:, :3] # Assume first 3 cols are x, y, z
            min_bound = np.min(coords, axis=0)
            max_bound = np.max(coords, axis=0)
            self.stats["bbox"] = {"min": min_bound.tolist(), "max": max_bound.tolist()}
            
            # 3. Assign Points to Chunks
            # Vectorized calculation of grid indices
            grid_indices = np.floor(coords / self.config.chunk_size).astype(int)
            
            # Group by unique grid indices
            # This is a bit tricky in pure numpy without pandas, but we can use unique with inverse
            unique_grids, inverse_indices = np.unique(grid_indices, axis=0, return_inverse=True)
            
            chunks_manifest = {}
            
            logger.info(f"Found {len(unique_grids)} potential chunks.")

            # 4. Write Chunks
            for i, grid_idx in enumerate(unique_grids):
                # Mask for points in this chunk
                mask = (inverse_indices == i)
                chunk_points = vertex_data[mask]
                
                if len(chunk_points) < self.config.min_points_per_chunk:
                    continue

                chunk_id = f"{grid_idx[0]}_{grid_idx[1]}_{grid_idx[2]}"
                chunk_filename = f"chunk_{chunk_id}.ply"
                chunk_path = self.config.output_dir / chunk_filename
                
                self._write_ply_chunk(chunk_path, chunk_points, property_names)
                
                chunks_manifest[chunk_id] = {
                    "file": chunk_filename,
                    "points": len(chunk_points),
                    "grid_index": grid_idx.tolist()
                }
                self.stats["chunks_created"] += 1

            # 5. Generate Manifest
            manifest_path = self._generate_manifest(input_path, chunks_manifest, start_time)
            
            logger.info(f"✅ Slicing complete. Created {self.stats['chunks_created']} chunks.")
            return str(manifest_path)

        except Exception as e:
            logger.exception(f"❌ Slicing failed: {e}")
            return None

    def _read_ply(self, input_path: Path) -> Tuple[List[str], Optional[np.ndarray], List[str]]:
        """
        Reads PLY file. Currently supports ASCII and basic Binary Little Endian.
        Returns (header_lines, numpy_array_of_data, property_names).
        """
        header = []
        properties = []
        vertex_count = 0
        format_type = "ascii"
        end_header_offset = 0
        
        try:
            with open(input_path, 'rb') as f:
                while True:
                    line = f.readline()
                    end_header_offset += len(line)
                    line_str = line.decode('utf-8').strip()
                    header.append(line_str)
                    
                    if line_str.startswith("format"):
                        if "binary_little_endian" in line_str:
                            format_type = "binary_little_endian"
                        elif "ascii" not in line_str:
                            logger.error(f"Unsupported PLY format: {line_str}")
                            return header, None, properties
                            
                    if line_str.startswith("element vertex"):
                        vertex_count = int(line_str.split()[-1])
                        
                    if line_str.startswith("property"):
                        properties.append(line_str.split()[-1])
                        
                    if line_str == "end_header":
                        break
            
            # Now read the data
            if format_type == "ascii":
                # Use numpy loadtxt for ASCII - might be slow for huge files but safer than custom parsing
                # Skip header lines
                data = np.loadtxt(input_path, skiprows=len(header))
            else:
                # Binary read
                # Assuming all properties are float32 for simplicity in this version
                # In a full production version, we'd parse the property types dynamically
                dtype = np.float32
                # Calculate expected bytes
                expected_bytes = vertex_count * len(properties) * 4
                
                with open(input_path, 'rb') as f:
                    f.seek(end_header_offset)
                    data_bytes = f.read(expected_bytes)
                    data = np.frombuffer(data_bytes, dtype=dtype)
                    data = data.reshape((vertex_count, len(properties)))
                    
            return header, data, properties

        except Exception as e:
            logger.error(f"Error reading PLY: {e}")
            return header, None, properties

    def _write_ply_chunk(self, output_path: Path, data: np.ndarray, properties: List[str]):
        """Writes a subset of points to a new PLY file (ASCII for compatibility)."""
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(data)}\n")
            for prop in properties:
                f.write(f"property float {prop}\n")
            f.write("end_header\n")
            
            # Fast numpy write
            np.savetxt(f, data, fmt='%.6f')

    def _generate_manifest(self, input_path: Path, chunks: Dict, start_time: float) -> Path:
        """Generates a JSON manifest of the slicing operation."""
        duration = time.time() - start_time
        self.stats["processing_time"] = duration
        
        manifest = {
            "source_file": str(input_path),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "chunk_size": self.config.chunk_size,
                "min_points": self.config.min_points_per_chunk
            },
            "statistics": self.stats,
            "chunks": chunks
        }
        
        manifest_path = self.config.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        return manifest_path

def main():
    parser = argparse.ArgumentParser(description="Slice massive PLY files into chunks.")
    parser.add_argument("--input_path", required=True, help="Path to input .ply file")
    parser.add_argument("--output_dir", default="output_chunks", help="Directory to save chunks")
    parser.add_argument("--chunk_size", type=float, default=10.0, help="Size of spatial chunks in meters")
    
    args = parser.parse_args()
    
    config = PlySlicerConfig(
        chunk_size=args.chunk_size,
        output_dir=args.output_dir
    )
    
    slicer = PlySlicer(config)
    slicer.slice_file(args.input_path)

if __name__ == "__main__":
    main()
