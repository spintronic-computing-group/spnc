"""
Tools for working with other local repositories

Functions
---------
repo_path_finder(searchpath, reponame)
    Searches for a repository and adds it and its parent directory to path

"""

import sys
from pathlib import Path


# Fn for finding and adding the repo to the path
def repo_path_finder(searchpath,reponame):
    """
    Find a particular repository on a search path and adds it to sys.path.

    Adds both parent directory and repo directory
    Will use the first match it finds

    Parameters
    ---------
    searchpath : Path
        path to search
    reponame : str
        the name of the repository to be added

    Examples
    --------
    repo_path_finder(Path.home() / "repos",
                                'machine_learning_library')
        This will search for the machine_learning_library repo under the
        directory ~/repos
    """

    # Search recursivly for the repository
    try:
        repopath = next(searchpath.rglob('**/' + reponame))
    except StopIteration:
        print("No such repository: " + reponame + ", on path: " +
              str(searchpath))
        return

    repopathparent = repopath.parent

    # Add parent to path if not already there
    try:
        sys.path.index(str(repopathparent))
    except ValueError:
        sys.path.append(str(repopathparent))

    # Add repo to path if not already there
    try:
        sys.path.index(str(repopath))
    except ValueError:
        sys.path.append(str(repopath))
