#!/bin/bash
rm -rf build dist mini_flash_attn.egg-info
pip uninstall -y mini_flash_attn
python setup.py install --user
