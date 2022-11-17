#!/bin/sh

python setup.py sdist
twine upload --repository pypitest dist/*
