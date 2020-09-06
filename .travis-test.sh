#!/bin/bash
set -ev
# patch is needed for patching
sudo apt-get update && sudo apt-get install -y patch wget

cp -r $APP_DIR guptri_py && cd guptri_py && ls -la
sage --version

export NPY_DISTUTILS_APPEND_FLAGS=1
sage -python setup.py config_fc --help-fcompiler
sage -sh -c 'gfortran --version'

sage -pip install --upgrade --no-index -v --user .
sage setup.py test
