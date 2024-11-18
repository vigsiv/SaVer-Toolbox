from setuptools import setup, find_packages

setup(
    name='SaVer_Toolbox',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.4.3',   # For plotting
        'numpy>=1.21.2',       # For numerical calculations
        'torch>=1.9.0',        # PyTorch
        'cvxpy>=1.1.15',       # For optimization
        'torchvision>=0.10.0', # For image processing
        'h5py>=3.3.0',         # For reading and writing HDF5 files
        'scipy>=1.7.1',        # For optimization
        requires-python = ">=3.9"
    ],
)