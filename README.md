# SPNC - Superparamagnetic Neuromorphic computing:

This is code for the SPNC project!

## Cloning the code:

- Pull the code from git
- Navigate to the code directory in terminal/powershell/etc
- run `conda env create --file [environment].yml` to create the environment with the correct dependencies (you'll have to have conda!)
- Activate the environment `conda activate [environment]`
- run `conda config --env --add channels conda-forge` to ensure conda-forge channel is in the environment


## Updating dependencies
-  Ideally install new packages with a specific version number so that the proper environment can be reproduced elsewhere, e.g.: `conda install [package]=[version number]`
- run: `conda env export -f [environment].yml  --from-history` to update the environment file
- If pulling an updated .yml file, then run: `conda env update -f [environment].yml --prune` to update and remove old dependencies

## Jupyter notebooks
- jupytext has been used to store python notebooks as text files for better version control.
- Matching .ipynb files may exist and if so should update the text files when saved.
- If .ipynb files don't exist (not stored in git) rebuild each of the notebooks using the command: `jupytext --to notebook [notebook].[extension]`
- Then, updating the notebooks should update the text files
- If only the notebook files exist, then *probably* jupytext wasn't used for them.

#### Atom and Hydrogen
*Hydrogen is a means of running python scripts like jupyter lab notebooks, it provides inline running of code in the atom editor. If using you can follow the below. It allows, for example, linked .py files of .ipynb notebooks to be run in atom*
- The ipykernel in the .yml file provides the background kernel for making this work
- run `python -m ipykernel install --user --name [environment] --display-name "Python ([environment])"`
- [See docs](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) if required
- Open atom from the shell with conda env activated `atom .` to get all the path variables
