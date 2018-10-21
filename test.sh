#!/usr/bin/env bash
nosetests  --nocapture --with-coverage --cover-html --cover-html-dir=htmlcov --cover-package="keras_bert" tests