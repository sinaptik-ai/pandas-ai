import os
import re
from io import BytesIO
from typing import Union

from ..helpers.sql_sanitizer import sanitize_file_name


def find_project_root(filename=None):
    """
    Check if Custom workspace path provide use that otherwise iterate to
    find project root
    """

    current_file_path = os.path.abspath(os.getcwd())

    # Navigate back until we either find a $filename file or there is no parent
    # directory left.
    root_folder = current_file_path
    while True:
        # Custom way to identify the project root folder
        if filename is not None:
            env_file_path = os.path.join(root_folder, filename)
            if os.path.isfile(env_file_path):
                break

        # Most common ways to identify a project root folder
        if (
            os.path.isfile(os.path.join(root_folder, "pyproject.toml"))
            or os.path.isfile(os.path.join(root_folder, "setup.py"))
            or os.path.isfile(os.path.join(root_folder, "requirements.txt"))
        ):
            break

        parent_folder = os.path.dirname(root_folder)
        if parent_folder == root_folder:
            # if project root is not found return cwd
            return os.getcwd()

        root_folder = parent_folder

    return root_folder


def find_closest(filename):
    return os.path.join(find_project_root(filename), filename)


def validate_name_format(value):
    """
    Validate name format to be 'my-org'
    """
    return bool(re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", value))


def validate_underscore_name_format(value):
    """
    Validate name format to be 'my_organization'
    """
    return bool(re.match(r"^[a-z0-9]+(?:_[a-z0-9]+)*$", value))


def transform_dash_to_underscore(value: str) -> str:
    return value.replace("-", "_")


def transform_underscore_to_dash(value: str) -> str:
    return value.replace("_", "-")


def get_validated_dataset_path(path: str):
    # Validate path format
    path_parts = path.split("/")
    if len(path_parts) != 2:
        raise ValueError("Path must be in format 'organization/dataset'")

    org_name, dataset_name = path_parts

    if not org_name or not dataset_name:
        raise ValueError("Both organization and dataset names are required")

    # Validate organization and dataset name format
    if not validate_name_format(org_name):
        raise ValueError(
            "Organization name must be lowercase and use hyphens instead of spaces (e.g. 'my-org')"
        )

    if not validate_name_format(dataset_name):
        raise ValueError(
            "Dataset path name must be lowercase and use hyphens instead of spaces (e.g. 'my-dataset')"
        )

    return org_name, dataset_name


def get_table_name_from_path(filepath: Union[str, BytesIO]) -> str:
    return (
        f"table_{sanitize_file_name(filepath)}"
        if isinstance(filepath, str)
        else "table_from_bytes"
    )
