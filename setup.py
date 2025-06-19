from setuptools import setup, find_packages

setup(
    name="raw2zarr",
    version="0.3.0",
    description="Tools for working with radar data and converting it to Zarr format",
    author="Alfonso Ladino, Max Grover",
    author_email="alfonso8@illinois.edu",
    url="https://github.com/aladinor/raw2zarr",
    packages=find_packages(include=["raw2zarr", "raw2zarr.*"]),
    include_package_data=True,
    package_data={"raw2zarr.config": ["*.json"]},
    python_requires=">=3.12",
    install_requires=[
        "pydantic",
        "cartopy",
        "fsspec",
        "dask",
        "netCDF4",
        "bottleneck",
        "zarr",
        "s3fs>=2024.3.1",
        "wradlib",
        "hvplot",
        "datashader",
        "xarray>=2025",
        "xradar>=0.9.0",
        "icechunk",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "flake8"],
        "notebooks": ["jupyterlab"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
