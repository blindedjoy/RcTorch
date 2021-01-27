#!/bin/bash
pip -V

pip uninstall --yes reservoir
yes | python -m pip install .
