from setuptools import setup, find_packages

setup(
    name='SaVer_Toolbox',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',   # For plotting
        'numpy',       # For numerical calculations
        'torch',        # PyTorch
        'cvxpy',       # For optimization
        'torchvision', # For image processing
        'h5py',         # For reading and writing HDF5 files
        'scipy',        # For optimization
        'requests',     # For downloading files
    ],
)