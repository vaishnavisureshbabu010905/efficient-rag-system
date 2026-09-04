"""Setup configuration for efficient-rag-system"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="efficient-rag-system",
    version="0.1.0",
    author="Vaishnavi Sureshbabu",
    description="Efficient RAG System with Binary Quantization - M1: Document Ingestion & Chunking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vaishnavisureshbabu010905/efficient-rag-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pdfplumber>=0.11.0",
        "reportlab>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
)
