rm -rf build dist int4_qk_matmul.egg-info
pip uninstall -y int4-qk-matmul
python setup.py install --user
