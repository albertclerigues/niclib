import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="niclib-albertclerigues", # Replace with your own username
    version="0.2",
    author="Albert Clèrigues",
    author_email="albert.clerigues@udg.edu",
    description="Neuroimaging and Deep Learning utilities to ease development.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/albertclerigues/niclib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)