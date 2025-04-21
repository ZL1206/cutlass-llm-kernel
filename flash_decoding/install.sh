#!/bin/bash
rm -rf build dist int4_attention.egg-info
pip uninstall -y int4-attention
python setup.py install
