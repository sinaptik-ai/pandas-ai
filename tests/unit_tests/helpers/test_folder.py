import os
import shutil
from pathlib import Path

import pytest

from pandasai import find_project_root
from pandasai.constants import DEFAULT_CHART_DIRECTORY
from pandasai.helpers.folder import Folder


def test_create_chart_directory():
    """Test if a folder is created properly."""
    Folder.create(DEFAULT_CHART_DIRECTORY)
    path = Path(os.path.join((str(find_project_root())), DEFAULT_CHART_DIRECTORY))
    # Convert Path to string
    assert path.exists()
    assert path.is_dir()
