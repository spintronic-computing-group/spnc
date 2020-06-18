# A module for machine learning with the SPNC resevoirs

# This code depends on the machine_learning_library repository
# v0.1.1
# It will need to be on your path for this to work! See the the code for importing local repos

#libraries
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

# Add local modules and paths to local repos
import repo_tools
repo_tools.repo_path_finder(Path.home() / "repos",
                            'machine_learning_library')
from machine_learning_library.single_node_res import single_node_reservoir

snr = single_node_reservoir(1,40,1, m0 = 1, dilution = 1.0, identity = False,
                            res = None, ravel_order = 'C')
