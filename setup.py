from setuptools import setup, find_packages

setup(
    name="sweep-infer",
    version="2.0.0",
    description="SWEEP: Single-pass Wavefront Entropy-gated Execution Pipeline for 7B/8B LLMs",
    author="Umberto Altieri",
    author_email="altieriumb7@gmail.com",
    license="MIT",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "sweep=sweep.cli:main",
            "sweep-server=sweep.server:main",
        ],
    },
    install_requires=[
        "numpy>=1.20.0",
    ],
    python_requires=">=3.9",
)
