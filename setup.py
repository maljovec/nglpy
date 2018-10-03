"""
      Setup script for nglpy, a wrapper library for the C++
      implementataion of the neighborhood graph library (NGL).
"""

from setuptools import setup, Extension
import re


def get_property(prop, project):
    """
        Helper function for retrieving properties from a project's
        __init__.py file
        @In, prop, string representing the property to be retrieved
        @In, project, string representing the project from which we will
        retrieve the property
        @Out, string, the value of the found property
    """
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


VERSION = get_property("__version__", "nglpy")


def long_description():
    """ Reads the README.rst file and extracts the portion tagged between
        specific LONG_DESCRIPTION comment lines.
    """
    description = ""
    recording = False
    with open("README.rst") as f:
        for line in f:
            if "END_LONG_DESCRIPTION" in line:
                return description
            elif "LONG_DESCRIPTION" in line:
                recording = True
                continue

            if recording:
                description += line


# Consult here: https://packaging.python.org/tutorials/distributing-packages/
setup(
    name="nglpy",
    packages=["nglpy"],
    version=VERSION,
    description="A pure python library that reimplements the neighborhood "
    + "graph library (NGL) for computing empty region graphs",
    long_description=long_description(),
    author="Dan Maljovec",
    author_email="maljovec002@gmail.com",
    license="BSD",
    test_suite="nglpy.tests",
    url="https://github.com/maljovec/nglpy",
    download_url="https://github.com/maljovec/nglpy/archive/"
    + VERSION
    + ".tar.gz",
    keywords=[
        "geometry",
        "neighborhood",
        "empty region graph",
        "neighborhood graph library",
        "beta skeleton",
        "relative neighbor",
        "Gabriel graph",
    ],
    # Consult here: https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=["numpy", "scipy", "scikit-learn"],
    python_requires=">=2.7, <4",
)
