import setuptools
import os

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="benchita",
    version="0.0.1",
    author="Federico A. Galatolo",
    author_email="federico.galatolo@unipi.it",
    description="",
    url="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["benchita"],
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta"
    ],
)