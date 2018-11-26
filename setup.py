# -*- coding: utf-8 -*-
import setuptools


with open("README.md") as f:

    readme = f.read()

with open("LICENSE.txt") as f:

    license = f.read()

setuptools.setup(
    name = "fempy",
    version = "0.1.x-alpha",
    description = "Flexible, extensible, and maintainable finite element models using Firedrake",
    long_description = readme,
    author= "Alexander G. Zimmerman",
    author_email = "zimmerman@aices.rwth-aachen.de",
    url = "https://github.com/alexanderzimmerman/fempy",
    license = license,
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
    ],
)