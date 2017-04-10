#!/bin/bash

if [ -f bin/activate ]; then
  source bin/activate
fi
export JUPYTER_PATH=${PWD}/.jupyter
export IPYTHONDIR=${PWD}/.ipython
export PYTHONPATH=${PWD}/modules:$PYTHONPATH
