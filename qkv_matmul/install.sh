#!/bin/bash
rm -rf build dist qkv_matmul.egg-info
pip uninstall -y qkv-matmul
python setup.py install --user
