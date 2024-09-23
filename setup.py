from setuptools import setup, find_packages

setup(
    name='nnSampleVerification',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',   # For plotting
        'numpy',        # For numerical calculations
        'torch',        # PyTorch
    ],
)