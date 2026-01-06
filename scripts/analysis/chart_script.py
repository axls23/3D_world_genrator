# Create mermaid flowchart for enhanced video-to-COLMAP pipeline
diagram_code = """
flowchart TD
    A[Input Video] --> B[Frame Extract]
    B --> C[Quality Check]
    
    C --> D{Motion Blur}
    C --> E{Feature Count}
    C --> F{PSNR Calc}
    C --> G{Contrast}
    
    D --> H{Filter Decision}
    E --> H
    F --> H
    G --> H
    
    H -->|Pass| I[Filtered Frames]
    H -->|Fail| J[Discarded]
    
    I --> K[COLMAP Process]
    K --> L[Extract Feat]
    K --> M[Match Feat]
    K --> N[3D Mapping]
    N --> O[3D Output]
    
    C --> P[Quality Report]
    
    classDef goodPath fill:#A5D6A7,stroke:#2E8B57,stroke-width:2px
    classDef badPath fill:#FFCDD2,stroke:#DB4545,stroke-width:2px
    
    class I,K,L,M,N,O goodPath
    class J badPath
"""

# Create the mermaid diagram and save as both PNG and SVG
png_path, svg_path = create_mermaid_diagram(
    diagram_code, 
    png_filepath='pipeline_flowchart.png',
    svg_filepath='pipeline_flowchart.svg',
    width=1200, 
    height=1000
)

print(f"Flowchart saved as: {png_path} and {svg_path}")