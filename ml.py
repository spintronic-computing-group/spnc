# A module for machine learning with the SPNC resevoirs

# This code depends on the machine_learning_library repository
# v0.1.0
# It will need to be on your path for this to work! See the the code for importing local repos

#libraries
from pathlib import Path

# Add local modules and repos
import repo_tools
repo_tools.repo_path_finder(Path.home() / "repos",
                            'machine_learning_library')
import machine_learning_library as mll
