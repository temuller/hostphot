import os
import sys
from pathlib import Path
from contextlib import contextmanager


def check_work_dir(wokrdir: str | Path) -> None:
    """Checks if the working directory exists. If it
    does not, one is created.

    Parameters
    ----------
    wokrdir: str
        Working directory path.
    """
    work_path = Path(wokrdir)
    if work_path.is_dir() is False:
        work_path.mkdir(parents=True)


@contextmanager
def suppress_stdout():
    """Suppresses annoying outputs.

    Useful with astroquery and aplpy packages.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
