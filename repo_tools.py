"""
Tools for working with other local repositories

Functions
---------
repo_path_finder(searchpath, reponame)
    Searches for a repository and adds it and its parent directory to path

repos_path_finder(searchpaths, reponames)
    Find a list of repositories on list of search paths. Add them to sys.paths
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

def repos_path_finder(searchpaths, reponames):
    """
    Find a list of repositories on list of search paths. Add them to sys.paths

    Utilises repo_path_finder for each entry
    Adds both parent directory and repo directory
    Will use the first match it finds on each search path

    Parameters
    ----------
    searchpaths : tuple of path
        paths to search
    reponames : tuple of str
        the names of the repository to be added

    Examples
    --------
        repo_path_finder( (Path('path1'), Path('path2') ),
                         ( 'machine_learning_library', ) )
            This will search for the machine_learning_library repo under
            'path1' & 'path2'
    """

    for searchpath in searchpaths:
        for reponame in reponames:
            repo_path_finder(searchpath, reponame)
