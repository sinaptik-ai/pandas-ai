"""Module to import optional dependencies.

Source: Taken from pandas/compat/_optional.py
"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
from pandas.util.version import Version

import pandasai.pandas as pd
from pandasai.constants import WHITELISTED_BUILTINS

if TYPE_CHECKING:
    import types

# Minimum version required for each optional dependency

VERSIONS = {
    "sklearn": "1.2.2",
    "statsmodels": "0.14.0",
    "seaborn": "0.12.2",
    "plotly": "5.14.1",
    "ggplot": "0.11.5",
    "scipy": "1.9.0",
    "streamlit": "1.23.1",
}

# A mapping from import name to package name (on PyPI) for packages where
# these two names are different.

INSTALL_MAPPING = {}


def get_version(module: types.ModuleType) -> str:
    """Get the version of a module."""
    version = getattr(module, "__version__", None)

    if version is None:
        raise ImportError(f"Can't determine version for {module.__name__}")

    return version


def get_environment(additional_deps: List[dict]) -> dict:
    """
    Returns the environment for the code to be executed.

    Returns (dict): A dictionary of environment variables
    """
    env = {
        "pd": pd,
        "plt": plt,
        "np": np,
    }

    for lib in additional_deps:
        module_name = lib["module"]
        symbol_name = lib["name"]
        alias = lib["alias"]

        module = import_dependency(module_name)

        # If symbol exists in the module, import it
        if hasattr(module, symbol_name):
            env[alias] = getattr(module, symbol_name)
        # Else, fallback to importing the full module and aliasing it
        else:
            full_module_name = f"{module_name}.{symbol_name}"
            try:
                env[alias] = import_dependency(full_module_name)
            except ImportError:
                env[alias] = module  # fallback to parent module

    env["__builtins__"] = {
        **{builtin: __builtins__[builtin] for builtin in WHITELISTED_BUILTINS},
        "__build_class__": __build_class__,
        "__name__": "__main__",
    }

    return env


def import_dependency(
    name: str,
    extra: str = "",
    errors: str = "raise",
    min_version: str | None = None,
):
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Args:
        name (str): The module name.
        extra (str): An additional text to include in the ImportError message.
        errors (str): Representing an action to do when a dependency
            is not found or its version is too old.
            Possible values: "raise", "warn", "ignore":
                * raise : Raise an ImportError
                * warn : Only applicable when a module's version is too old.
                  Warns that the version is too old and returns None
                * ignore: If the module is not installed, return None, otherwise,
                  return the module, even if the version is too old.
                  It's expected that users validate the version locally when
                  using ``errors="ignore"`` (see. ``io/html.py``)
        min_version (str): Specify a minimum version that is different from
            the global pandas minimum version required. Defaults to None.

    Returns:
         Optional[module]:
            The imported module, when found and the version is correct.
            None is returned when the package is not found and `errors`
            is False, or when the package's version is too old and `errors`
            is `'warn'`.
    """

    assert errors in {"warn", "raise", "ignore"}

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"Missing optional dependency '{install_name}'. {extra} "
        f"Use pip or conda to install {install_name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        if errors == "raise":
            raise ImportError(msg) from exc
        return None

    # Handle submodules: if we have submodule, grab parent module from sys.modules
    parent = name.split(".")[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        version = get_version(module_to_get)
        if version and Version(version) < Version(minimum_version):
            msg = (
                f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
                f"(version '{version}' currently installed)."
            )
            if errors == "warn":
                warnings.warn(
                    msg,
                    UserWarning,
                )
                return None
            if errors == "raise":
                raise ImportError(msg)

    return module
