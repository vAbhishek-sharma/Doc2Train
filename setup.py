# NEW: setup.py - Make it pip installable
"""
Setup script to make Doc2Train installable via pip
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="doc2train",
    version="2.0.0",
    description="Convert documents to AI training data at lightning speed",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/doc2train",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "PyMuPDF>=1.23.0",
        "Pillow>=10.0.0",
        "pandas>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "web": [
            "streamlit>=1.28.0",
            "plotly>=5.17.0",
        ],
        "advanced": [
            "anthropic>=0.8.0",
            "google-generativeai>=0.3.0",
            "transformers>=4.35.0",
            "torch>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "doc2train=doc2train.cli:main",
            "doc2train-api=doc2train.api:start_server",
            "doc2train-web=doc2train.web:start_ui",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
    ],
    python_requires=">=3.8",
    keywords="ai, machine learning, document processing, training data, nlp",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/doc2train/issues",
        "Documentation": "https://doc2train.readthedocs.io/",
        "Source": "https://github.com/yourusername/doc2train",
    },
)
