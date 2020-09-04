#!/bin/bash
set -ev
# patch is needed for patching
sudo apt-get update && sudo apt-get install -y patch wget

cp -r $APP_DIR guptri_py && cd guptri_py && ls -la
sage --version

sage -pip install --upgrade --no-index -v --user .
sage setup.py test
