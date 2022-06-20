# SPNC - Superparamagnetic Neuromorphic computing:

This is code for the SPNC project!

## Cloning the code:

- Pull the code from git
- Navigate to the code directory in terminal/powershell/etc
- run `conda env create --file [environment].yml` to create the environment with the correct dependencies (you'll have to have conda!)
- Activate the environment `conda activate [environment]`
- run `conda config --env --add channels conda-forge` to ensure conda-forge channel is in the environment
- To enable ipywidgets in jupyter lab you must install the extension: `jupyter labextension install @jupyter-widgets/jupyterlab-manager` (or install it in jupyter lab's extension manager)
- pre-commit is used to handle git hooks, please see the section below about installing this on your system


## Updating dependencies
-  Ideally install new packages with a specific version number so that the proper environment can be reproduced elsewhere, e.g.: `conda install [package]=[version number]`
- run: `conda env export -f [environment].yml  --from-history` to update the environment file. ***If this does not work, you may need to update te yaml file manually.***
- If pulling an updated .yml file, then run: `conda env update -f [environment].yml --prune` to update and remove old dependencies

## Precommit git hooks
- This repo uses git hooks via pre-commit to enable common practice across users/development environments
- To use these pre commits, they need to be installed. Activate the env and run: ```pre-commit install```. You'll need to do this on each dev machine.
- hooks are only run on commited files. To run on all files use ```pre-commit run --all-files```


## Running the code
- All code should be run from the root directory of the project in order to properly resolve paths e.g.
    ``` python plotting/APL_2021/reservoir_response.py ```
- This might need updating in legacy files (before this snippet was added in GIT)

## Jupyter notebooks
- jupytext has been used to store python notebooks as text files for better version control.
- Matching .ipynb files may exist and if so should update the text files when saved or committed
- If .ipynb files don't exist (not stored in git) you can rebuild each of the notebooks using the command: `jupytext --to notebook [notebook].[extension]` NB: This is good if you don't want to version the notebook. Consider adding the created .ipynb file to .gitignore so as to not accidently commit it. If you do want to version the notebook along with the text file, you should use the sync commands like those below. Additionaly, you could use the sync commands if you don't want to version, but you want to make changes in the notebook and sync them back to the text file (or use VScode as below...).
- If only the notebook files exist, then *probably* jupytext wasn't used for them.
- If you want to link a new notebook to a text file (or visa versa) you will need to make this happen...
- In jupyter this is avaliable in drop down menus, otherwise this can be done from the command line...
- Running (in activated env) ``` jupytext --set-formats [formats] --sync [notebook].ipynb ``` where ```[formats]``` could be, for example, ``` ipynb,py:percent ``` or ``` ipynb,md ``` will create a new text file and link it the notebook. Similar commands can be used to link in the other direction.
- After this the git hooks (see section on precommit git hooks) will ensure the linked files are synced based on timestamp (Jupyter Lab will handle this as well on save if used)
- .ipynb versions should only be versioned in git if the outputs want to be versioned (e.g. code acting as documentation or discovery). Use the .gitignore file to exclude particular .ipynb if they're used for viewing, but not versioned (i.e. outputs reproduced every time). NB: If you do this, then updates to a notebook version of the file **will not be synced back by pre-commit**.

### Atom and Hydrogen
*Hydrogen is a means of running python scripts like jupyter lab notebooks, it provides inline running of code in the atom editor. If using you can follow the below. It allows, for example, linked .py files of .ipynb notebooks to be run in atom*
- The ipykernel in the .yml file provides the background kernel for making this work
- run `python -m ipykernel install --user --name [environment] --display-name "Python ([environment])"`
- [See docs](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) if required
- Open atom from the shell with conda env activated `atom .` to get all the path variables

### VScode and jupytyer notebooks
- Using the Jupyter plugin allows live editing of notebooks and notebooks diffs in git
- May need to have the kernel installed as Atom/Hydrogen above
- A jupytext plugin allows opening of .py files as notebooks which save back to .py, but don't leave a notebook (.ipynb) file on disc
- However, for full sync, need to use githooks (see above)

