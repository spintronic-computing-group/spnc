# Setting up the path to find other repos
import sys
from pathlib import Path


#Some user parameters

# Path to repositories in general
searchpath = Path.home() / "repos"

# names of repositories to add as a tuple of strings
reponames = ('machine_learning_library',)


# Code

# Fn for finding and adding the repo to the path
def search_and_add(searchpath,reponame):
    """
    Find a particular repository on a search path and adds it to sys.path.

    Adds both parent directory and repo directory
    Will use the first match it finds
    Arguments:
    searchpath -- path to search as type Path
    reponame -- the name of the repository to be added as a string
    """

    # Search recursivly for the repository
    repopath = next(searchpath.rglob('**/' + reponame))
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

# Run the function for all repos to add
for reponame in reponames: search_and_add(searchpath,reponame)
