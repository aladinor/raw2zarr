from setuptools import setup, find_packages

setup(
    name="raw2zarr",
    version="0.3.0",
    description="Tools for working with radar data and converting it to Zarr format",
    author="Alfonso Ladino, Max Grover",
    author_email="alfonso8@illinois.edu",
    url="https://github.com/aladinor/raw2zarr",
    packages=find_packages(),  # Automatically finds all packages in the directory
    python_requires=">=3.12",
    install_requires=[
        "cartopy",
        "fsspec",
        "dask",
        "netCDF4",
        "bottleneck",
        "zarr",
        "s3fs>=2024.3.1",
        "arm_pyart>=1.18.1",
        "wradlib",
        "hvplot",
        "datashader",
        "xarray>=2024.10",
        "xradar>=0.8.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "flake8"],  # Add development dependencies here
        "notebooks": ["jupyterlab"],  # Dependencies for working with notebooks
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
