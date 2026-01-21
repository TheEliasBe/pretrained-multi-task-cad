import os

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    "torch>=2.4.0",
    "lightning>=2.0.0",
    "transformers>=4.36.0",
    "bitsandbytes>=0.41.1",
    "deepspeed>=0.10.0",
    "dgl>=1.1.2",
    "peft>=0.5.0",
    "wandb",
    "ruamel.yaml",
    "psutil",
    "huggingface-hub",
    "omegaconf",
    "seaborn",
    "plotly",
    "trimesh",
    "torch-geometric",
    "cadquery",
    "tree-sitter-python",
    "codebleu",
    "rich",
    "memory-profiler",
    "line-profiler",
    "python-multipart",
]

# Optional dependencies
extras_require = {
    "logging": ["mlflow", "tensorboard"],
    "cuda": [
        "nvidia-pyindex; platform_system=='Linux'",
        "nvidia-cublas-cu11; platform_system=='Linux'",
    ],
    "dev": [
        "pytest>=6.0",
        "pytest-cov",
        "black",
        "flake8",
        "isort",
    ],
    "all": [
        "mlflow",
        "tensorboard",
        "pytest>=6.0",
        "pytest-cov",
        "black",
        "flake8",
        "isort",
    ],
}

setup(
    name="cadcoder",
    version="0.1.0",
    author="CADCoder Team",
    author_email="cadcoder@example.com",
    description="Lightweight project for generating CAD models from multiple input modalities (B-Rep, images, and text)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/cadcoder",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "cadcoder-train=cadcoder.train:main",
            "cadcoder-eval=cadcoder.modules.cad_evaluator:main",
        ],
    },
    package_data={
        "cadcoder": [
            "models/*.yaml",
            "config.yaml",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="cad, machine learning, computer vision, nlp, 3d modeling",
)
