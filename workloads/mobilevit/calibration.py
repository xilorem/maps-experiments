#!/usr/bin/env python3
"""Use the same MAGIA-v3 calibration reporting as the MobileViT slice."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent.parent / 'mobilevit_170-179/calibration.py'), run_name='__main__')
