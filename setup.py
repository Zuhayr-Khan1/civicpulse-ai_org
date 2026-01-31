from setuptools import setup, find_packages

setup(
    name="civicpulse-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "geopandas>=0.14.0",
        "rasterio>=1.3.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "jupyter>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.66.0",
    ],
    python_requires=">=3.8",
)
