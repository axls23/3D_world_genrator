from setuptools import setup, find_packages

setup(
    name="hypersplat",
    version="1.0.0",
    description="High-Velocity Interactive 3DGS Engine",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi",
        "uvicorn",
        "websockets",
        "boto3",
        "numpy",
        "torch",
        "opencv-python"
    ],
)
