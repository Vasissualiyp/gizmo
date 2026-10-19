"""test/chemcool_cooling_path with CHEMISTRYNETWORK=3 (full low-metallicity network)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chemcool_cooling_path"))
from test_chemcool_cooling_path import run_cooling_path_checks


def test_chemcool_cooling_path_net3():
    run_cooling_path_checks("chemcool_cooling_path_net3")
