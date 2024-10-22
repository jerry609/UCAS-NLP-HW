from setuptools import setup, find_packages

setup(
    name="catdog_dataset",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A dataset loader for cat and dog image classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/catdog_dataset",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "Pillow>=7.0.0",
    ],
)
