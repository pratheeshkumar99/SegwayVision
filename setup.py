from setuptools import setup, find_packages

setup(
    name="segmentation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'albumentations',
        'opencv-python',
        'imutils',
        'numpy',
    ]
)