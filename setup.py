"""
Setup file for BDBV Host Prediction package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

setup(
    name="bdbv-host-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Predicting BDBV host susceptibility using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/bdbv-host-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest>=7.4.0",
            "sphinx>=7.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
        ],
        "all": [
            "torch>=2.0.0",
            "torch-geometric>=2.3.0",
            "geopandas>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bdbv-train=scripts.run_training:main",
            "bdbv-predict=scripts.run_prediction:main",
            "bdbv-validate=scripts.run_validation:main",
            "bdbv-figures=scripts.generate_figures:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
