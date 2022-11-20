"""
    Utility library for NGLPy
"""

import warnings


def consume_extra_args(fail_on_missing=False, **kwargs):
    """
    Supports forward compatibility.

    This helper function will take all of the extra arguments not currently
    used by a calling function/environment, and print a warning to standard
    error letting them know they are using an unsupported feature passed into a
    function.

    Parameters
    ----------
    fail_on_missing : bool
      A flag specifying whether missing arguments should throw a warning
      (False) or an exception (True)
    kwargs : dict
      A dictionary of keyword arguments not used by an existing interface.

    Returns
    -------
    None

    """
    for kw in kwargs:
        msg = f"Warning: the current version of nglpy does not accept {kw}. "
        if fail_on_missing:
            raise NotImplementedError(msg)
        else:
            msg += "It will be ignored."
            warnings.warn(msg, UserWarning, stacklevel=3)
