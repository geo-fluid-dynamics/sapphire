import setuptools


with open("README.md") as f:

    readme = f.read()

with open("LICENSE.txt") as f:

    license = f.read()

setuptools.setup(
    name = "fempy",
    version = "0.1.x-alpha",
    packages = setuptools.find_packages(),
    url = "https://github.com/alexanderzimmerman/fempy",
    description = "Finite element models using Firedrake",
    long_description = readme,
    license = license,
    author = "Alexander G. Zimmerman",
    author_email = "zimmerman@aices.rwth-aachen.de",
)
