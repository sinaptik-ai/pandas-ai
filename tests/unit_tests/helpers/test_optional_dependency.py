"""Unit tests for the import_optional_dependency function.

Source: Taken from pandas/tests/test_optional_dependency.py
"""

import pytest

from pandasai.core.code_execution.environment import (
    get_environment,
    import_dependency,
)


def test_import_optional():
    match = "Missing .*notapackage.* pip .* conda .* notapackage"
    with pytest.raises(ImportError, match=match) as exc_info:
        import_dependency("notapackage")
    # The original exception should be there as context:
    assert isinstance(exc_info.value.__context__, ImportError)

    result = import_dependency("notapackage", errors="ignore")
    assert result is None


def test_xlrd_version_fallback():
    pytest.importorskip("xlrd")
    import_dependency("xlrd")


def test_env_for_necessary_deps():
    env = get_environment()
    assert "pd" in env
    assert "plt" in env
    assert "np" in env
    assert "px" in env
