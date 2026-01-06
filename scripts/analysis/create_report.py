import json
from pathlib import Path

# Create summary report
output_dir = Path('data/COLMAP_DATASET')
images_dir = output_dir / 'images'
downsampled_dir = output_dir / 'images_2'
sparse_dir = output_dir / 'sparse' / '0'

report = {
    'input_directory': 'data/IMAGES',
    'output_directory': str(output_dir),
    'downsample_factor': 2,
    'camera_model': 'SIMPLE_PINHOLE',
    'frames_processed': len(list(images_dir.glob('*.jpg'))),
    'downsampled_images_count': len(list(downsampled_dir.glob('*.jpg'))),
    'structure': {
        'images': str(images_dir),
        'downsampled_images': str(downsampled_dir),
        'sparse': str(sparse_dir),
        'database': str(output_dir / 'database.db')
    }
}

# Save report
report_path = output_dir / 'processing_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f'Summary report saved to: {report_path}')

# Print final structure
print('\nFinal directory structure:')
print(f'{output_dir}/')
print(f'├── images/           ({report["frames_processed"]} images)')
print(f'├── images_2/         ({report["downsampled_images_count"]} images)')
print(f'├── sparse/')
print(f'│   └── 0/            (COLMAP reconstruction)')
print(f'├── database.db       (COLMAP database)')
print(f'└── processing_report.json')
