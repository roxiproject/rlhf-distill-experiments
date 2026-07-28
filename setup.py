from setuptools import setup, find_packages

setup(
    name="rlhf-distill-experiments",
    version="0.1.0",
    description="Numpy-only knowledge distillation toolkit: train a small student MLP to mimic a larger teacher MLP.",
    packages=find_packages(include=["rlhf_distill", "rlhf_distill.*"]),
    install_requires=[
        "numpy",
        "pyyaml",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "rlhf-distill=rlhf_distill.cli:main",
        ],
    },
)
