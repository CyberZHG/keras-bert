#!/usr/bin/env bash
rm -f README.rst && m2r README.md
rm -f dist/*
python3 setup.py sdist
twine upload dist/*
