"""
      Setup script for nglpy, a wrapper library for the C++
      implementataion of the neighborhood graph library (NGL).
"""

import sys

from setuptools import Extension, setup

requirements = open("requirements.txt").read().strip().split("\n")
extra_compile_args = []  # type: ignore
extra_link_args = []  # type: ignore
if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++"]
    extra_link_args = ["-stdlib=libc++"]


FILES = ["ngl_wrap.cpp", "GraphStructure.cpp", "UnionFind.cpp"]


def long_description():
    """Reads the README.rst file and extracts the portion tagged between
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
    description="A wrapper library for exposing the C++ neighborhood "
    + "graph library (NGL) for computing empty region graphs to "
    + "python",
    long_description=long_description(),
    test_suite="nglpy.tests",
    install_requires=requirements,
    python_requires=">=2.7, <4",
    ext_modules=[
        Extension(
            "nglpy._ngl",
            FILES,
            extra_compile_args=[
                "-std=c++11",
                "-O3",
                *extra_compile_args,
            ],
            extra_link_args=extra_link_args,
        )
    ],
)
