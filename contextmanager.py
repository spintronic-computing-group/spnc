import sys
import io
from contextlib import contextmanager
@contextmanager

def suppress_stdout(suppress=True):
    """
    Control the print output
    suppress=True: prohibit print
    suppress=False: allow print
    """

    if suppress:
        # save the standard output
        old_stdout = sys.stdout
        # prohibit print
        sys.stdout = io.StringIO()
        try:
            yield
        finally:
            # recover the standard output
            sys.stdout = old_stdout
    else:
        yield