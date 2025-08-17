"""Setup script for SciEvo package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="scievo",
    version="0.1.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="A Longitudinal Scientometric Dataset Analysis Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous/SciEvo",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
            "bokeh>=2.0",
        ],
        "gpu": [
            "torch>=1.10.0",
            "torch-geometric>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "scievo=scievo.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)