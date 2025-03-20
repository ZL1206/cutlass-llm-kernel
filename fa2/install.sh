#!/bin/bash
rm -rf build dist flash_attn_zl.egg-info
pip uninstall -y flash_attn_zl
python setup.py install --user
