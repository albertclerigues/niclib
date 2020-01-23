import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="niclib", # Replace with your own username
    version="1.0b0",
    author="Albert Clèrigues",
    author_email="albert.clerigues@udg.edu",
    description="Neuroimaging and Deep Learning utilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/albertclerigues/niclib",
    package_dir={'': 'src'},  # Optional
    packages=setuptools.find_packages(where='src'),  # Required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
		'torch>=1.3',
		'nibabel>=2.4.1',
		'numpy>=1.17',
		'scikit-image>=0.14',
		'art>=3.1',
		'scipy>=1.3',
		'SimpleITK>=1.2.4']
)