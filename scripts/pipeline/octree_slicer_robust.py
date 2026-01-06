import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OctreeNode:
    def __init__(self, bbox_min: np.ndarray, bbox_max: np.ndarray, depth: int, id: str):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.depth = depth
        self.id = id
        self.children: List['OctreeNode'] = []
        self.point_indices: List[int] = [] # List of indices assigned to this node
        self.is_leaf = False

    def intersects(self, center: np.ndarray, radius: float) -> bool:
        """
        Checks if a sphere (center, radius) intersects with this AABB.
        """
        # Find the point on the AABB closest to the sphere center
        closest = np.maximum(self.bbox_min, np.minimum(center, self.bbox_max))
        distance_sq = np.sum((closest - center) ** 2)
        return distance_sq <= (radius ** 2)

class OctreeSlicerConfig:
    def __init__(self, 
                 max_points_per_chunk: int = 50000,
                 max_depth: int = 8,
                 min_chunk_size: float = 1.0,
                 output_dir: str = "output_octree",
                 expansion_factor: float = 3.0):
        self.max_points_per_chunk = max_points_per_chunk
        self.max_depth = max_depth
        self.min_chunk_size = min_chunk_size
        self.output_dir = Path(output_dir)
        self.expansion_factor = expansion_factor # Multiplier for scale to get radius (3-sigma)

class OctreeSlicer:
    def __init__(self, config: OctreeSlicerConfig):
        self.config = config
        self.stats = {
            "total_points": 0,
            "chunks_created": 0,
            "processing_time": 0,
            "tree_depth": 0,
            "duplicated_points": 0
        }

    def slice_file(self, input_path: str) -> Optional[str]:
        start_time = time.time()
        input_path = Path(input_path)
        logger.info(f"🌲 Robust Octree Slicing {input_path}...")

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return None

        if self.config.output_dir.exists():
            shutil.rmtree(self.config.output_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Read Data
            # We need positions AND scales to calculate extent
            data_dict, property_names = self._read_ply_full(input_path)
            if data_dict is None:
                return None
            
            coords = data_dict['coords']
            scales = data_dict['scales'] # These are log scales usually
            
            # Calculate radii
            # If scales are log scales, exp them. If they are raw, use them.
            # Heuristic: if max(scales) > 10, it's probably log scale (exp(10) is huge). 
            # Actually, 3DGS scales are usually small. log(0.01) = -4.6. 
            # If we see negative values, it's definitely log scale.
            if np.min(scales) < 0:
                logger.info("Detected log scales (negative values present). Applying exp().")
                radii = np.exp(scales)
            else:
                # If all positive, could be raw or log. 
                # If max is small (< 1), likely raw. If max is > 1, could be log.
                # Let's assume log if mean is negative, but here min < 0 covered it.
                # If all > 0, let's assume raw for safety unless they are huge.
                logger.info("Assuming raw scales.")
                radii = scales
                
            # Max radius per splat * expansion factor
            max_radii = np.max(radii, axis=1) * self.config.expansion_factor
            
            self.stats["total_points"] = len(coords)
            
            # 2. Global BBox
            min_bound = np.min(coords, axis=0)
            max_bound = np.max(coords, axis=0)
            
            # 3. Build Tree Structure (based on centroids only for balanced partition)
            logger.info(f"Building Octree Structure for {len(coords)} points...")
            root = OctreeNode(min_bound, max_bound, 0, "0")
            all_indices = np.arange(len(coords))
            self._build_structure(root, coords, all_indices)
            
            # 4. Assign Points with Overlap
            logger.info("Assigning points with overlap (Splat-Aware)...")
            self._assign_points_robust(root, coords, max_radii, all_indices)
            
            # 5. Export Chunks
            logger.info("Exporting chunks...")
            chunks_manifest = {}
            # We need to reconstruct the full vertex array for writing
            # We'll just use the raw data we read
            # But we need to index into it.
            # Let's keep the raw data bytes or array handy.
            # _read_ply_full returned structured dict, we can reconstruct.
            
            self._export_recursive(root, data_dict, property_names, chunks_manifest)
            
            # 6. Generate Manifest
            manifest_path = self._generate_manifest(input_path, chunks_manifest, start_time, min_bound, max_bound)
            
            logger.info(f"✅ Robust Slicing complete. Created {self.stats['chunks_created']} chunks.")
            return str(manifest_path)

        except Exception as e:
            logger.exception(f"❌ Slicing failed: {e}")
            return None

    def _build_structure(self, node: OctreeNode, coords: np.ndarray, indices: np.ndarray):
        """
        Builds the tree structure using strict centroid partitioning.
        """
        num_points = len(indices)
        node_size = np.max(node.bbox_max - node.bbox_min)
        
        if (num_points <= self.config.max_points_per_chunk or 
            node.depth >= self.config.max_depth or 
            node_size < self.config.min_chunk_size):
            node.is_leaf = True
            return

        mid = (node.bbox_min + node.bbox_max) / 2
        
        # Create children
        children_nodes = []
        for i in range(8):
            child_min = np.copy(node.bbox_min)
            child_max = np.copy(node.bbox_max)
            if i & 1: child_min[2] = mid[2]
            else:     child_max[2] = mid[2]
            if i & 2: child_min[1] = mid[1]
            else:     child_max[1] = mid[1]
            if i & 4: child_min[0] = mid[0]
            else:     child_max[0] = mid[0]
            
            children_nodes.append(OctreeNode(child_min, child_max, node.depth + 1, f"{node.id}-{i}"))

        # Partition points to children for the next recursive step
        # This is just to determine tree structure, so strict split is fine
        for i, child in enumerate(children_nodes):
            in_child = np.all((coords[indices] >= child.bbox_min) & (coords[indices] <= child.bbox_max), axis=1)
            child_indices = indices[in_child]
            
            if len(child_indices) > 0:
                node.children.append(child)
                self._build_structure(child, coords, child_indices)

    def _assign_points_robust(self, node: OctreeNode, coords: np.ndarray, radii: np.ndarray, indices: np.ndarray):
        """
        Assigns points to leaf nodes, duplicating them if they overlap multiple nodes.
        """
        if node.is_leaf:
            # In the leaf, we accept all indices passed to us
            # (These indices have been filtered by intersection tests in parents)
            node.point_indices = indices.tolist()
            return

        # For intermediate nodes, check intersection with children
        for child in node.children:
            # Find points that intersect with this child's bbox
            # We can vectorize this check
            
            # Get subset of data
            sub_coords = coords[indices]
            sub_radii = radii[indices]
            
            # Sphere-AABB intersection test
            # Closest point on AABB to sphere center
            closest = np.maximum(child.bbox_min, np.minimum(sub_coords, child.bbox_max))
            distance_sq = np.sum((closest - sub_coords) ** 2, axis=1)
            
            intersects = distance_sq <= (sub_radii ** 2)
            
            child_indices = indices[intersects]
            
            if len(child_indices) > 0:
                self._assign_points_robust(child, coords, radii, child_indices)

    def _read_ply_full(self, input_path: Path) -> Tuple[Optional[Dict], List[str]]:
        """
        Reads PLY and extracts coords and scales. Returns all data for writing later.
        """
        try:
            # We'll use a slightly more robust parsing to get all properties
            # Re-implementing a basic binary reader that captures all columns
            
            header = []
            properties = []
            vertex_count = 0
            format_type = "ascii"
            end_header_offset = 0
            
            with open(input_path, 'rb') as f:
                while True:
                    line = f.readline()
                    end_header_offset += len(line)
                    line_str = line.decode('utf-8').strip()
                    header.append(line_str)
                    if line_str.startswith("format"):
                        if "binary_little_endian" in line_str: format_type = "binary_little_endian"
                    if line_str.startswith("element vertex"):
                        vertex_count = int(line_str.split()[-1])
                    if line_str.startswith("property"):
                        properties.append(line_str.split()[-1])
                    if line_str == "end_header":
                        break
            
            # Map property names to indices
            prop_map = {name: i for i, name in enumerate(properties)}
            
            if format_type == "ascii":
                raw_data = np.loadtxt(input_path, skiprows=len(header))
            else:
                dtype = np.float32
                expected_bytes = vertex_count * len(properties) * 4
                with open(input_path, 'rb') as f:
                    f.seek(end_header_offset)
                    raw_data = np.frombuffer(f.read(expected_bytes), dtype=dtype)
                    raw_data = raw_data.reshape((vertex_count, len(properties)))
            
            # Extract Coords
            x_idx = prop_map.get('x')
            y_idx = prop_map.get('y')
            z_idx = prop_map.get('z')
            
            if x_idx is None or y_idx is None or z_idx is None:
                logger.error("Missing x, y, or z properties")
                return None, properties

            coords = raw_data[:, [x_idx, y_idx, z_idx]]
            
            # Extract Scales
            # Try scale_0, scale_1, scale_2
            s0_idx = prop_map.get('scale_0')
            s1_idx = prop_map.get('scale_1')
            s2_idx = prop_map.get('scale_2')
            
            if s0_idx is not None and s1_idx is not None and s2_idx is not None:
                scales = raw_data[:, [s0_idx, s1_idx, s2_idx]]
            else:
                logger.warning("Scale properties not found. Using default radius 0.1 for all.")
                scales = np.ones((vertex_count, 3)) * -2.3 # exp(-2.3) ~ 0.1
            
            return {
                "raw_data": raw_data,
                "coords": coords,
                "scales": scales
            }, properties

        except Exception as e:
            logger.error(f"Error reading PLY: {e}")
            return None, []

    def _export_recursive(self, node: OctreeNode, data_dict: Dict, property_names: List[str], manifest: Dict):
        if node.is_leaf:
            if not node.point_indices:
                return

            chunk_filename = f"chunk_{node.id}.ply"
            chunk_path = self.config.output_dir / chunk_filename
            
            # Get data for these indices
            indices = np.array(node.point_indices)
            chunk_data = data_dict['raw_data'][indices]
            
            self._write_ply_chunk(chunk_path, chunk_data, property_names)
            
            manifest[node.id] = {
                "file": chunk_filename,
                "points": len(chunk_data),
                "bbox": {
                    "min": node.bbox_min.tolist(),
                    "max": node.bbox_max.tolist()
                },
                "depth": node.depth
            }
            self.stats["chunks_created"] += 1
        else:
            for child in node.children:
                self._export_recursive(child, data_dict, property_names, manifest)

    def _write_ply_chunk(self, output_path: Path, data: np.ndarray, properties: List[str]):
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(data)}\n")
            for prop in properties:
                f.write(f"property float {prop}\n")
            f.write("end_header\n")
            np.savetxt(f, data, fmt='%.6f')

    def _generate_manifest(self, input_path: Path, chunks: Dict, start_time: float, min_b, max_b) -> Path:
        duration = time.time() - start_time
        self.stats["processing_time"] = duration
        
        # Calculate duplication factor
        total_chunk_points = sum(c['points'] for c in chunks.values())
        self.stats["duplicated_points"] = total_chunk_points - self.stats["total_points"]
        
        manifest = {
            "source_file": str(input_path),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "max_points_per_chunk": self.config.max_points_per_chunk,
                "max_depth": self.config.max_depth,
                "expansion_factor": self.config.expansion_factor
            },
            "global_bbox": {
                "min": min_b.tolist(),
                "max": max_b.tolist()
            },
            "statistics": self.stats,
            "chunks": chunks
        }
        
        manifest_path = self.config.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        return manifest_path

def main():
    parser = argparse.ArgumentParser(description="Robust Octree Slice for Gaussian Splats.")
    parser.add_argument("--input_path", required=True, help="Path to input .ply file")
    parser.add_argument("--output_dir", default="output_octree_robust", help="Directory to save chunks")
    parser.add_argument("--max_points", type=int, default=50000, help="Max points per chunk")
    parser.add_argument("--max_depth", type=int, default=6, help="Max octree depth")
    parser.add_argument("--expansion", type=float, default=3.0, help="Expansion factor for splat radius (sigma)")
    
    args = parser.parse_args()
    
    config = OctreeSlicerConfig(
        max_points_per_chunk=args.max_points,
        max_depth=args.max_depth,
        output_dir=args.output_dir,
        expansion_factor=args.expansion
    )
    
    slicer = OctreeSlicer(config)
    slicer.slice_file(args.input_path)

if __name__ == "__main__":
    main()
