import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Detector-YOLOv3",
    version=os.getenv("GITHUB_REF_NAME", "v0.0.1"),
    author="Jiho Park",
    author_email="pjh5672.dev@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(exclude=["*tests*","*example*","*weights*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    setup_requires=['wheel'],
    install_requires=[
        'dvc>=2.18.0',
        'torch>=1.12.0',
        'torchvision>=0.13.0'
    ],
    python_requires='>=3.8',
)