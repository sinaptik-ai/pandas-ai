""" Unit tests for the IBMwatsonx class. """
import sys

import pytest

from extensions.llms.ibm.pandasai_ibm.ibm_watsonx import IBMwatsonx
from pandasai.exceptions import APIKeyNotFoundError


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
class TestIBMwatsonx:
    """Unit tests for the IBMwatsonx class."""

    def test_without_key(self):
        """Test that an APIKeyNotFoundError is raised when the API key is missing.

    This test verifies that the IBMwatsonx class raises an APIKeyNotFoundError
    when an attempt is made to use it without providing an API key."""
        with pytest.raises(APIKeyNotFoundError):
            IBMwatsonx()

    def test_without_url(self):
        """Test case for APIKeyNotFoundError when no URL is provided.

    This test ensures that an APIKeyNotFoundError is raised when an instance 
    of IBMwatsonx is created with an API key but without a URL.

    Args:
        self: The test instance.

    Returns:
        None"""
        with pytest.raises(APIKeyNotFoundError):
            IBMwatsonx(api_key="test")

    def test_without_project_id(self):
        """Test case to ensure an APIKeyNotFoundError is raised when no project ID is provided.

    This test verifies that the IBMwatsonx class raises an APIKeyNotFoundError
    if instantiated with an API key and URL but without a project ID.

    Args:
        self: An instance of the test class."""
        with pytest.raises(APIKeyNotFoundError):
            IBMwatsonx(api_key="test", url="http://test.com")

    def test_watsonx_invalid_model(self):
        from ibm_watsonx_ai.wml_client_error import WMLClientError

        with pytest.raises(WMLClientError):
            IBMwatsonx(
                api_key="test",
                watsonx_url="https://us-south.ml.cloud.ibm.com",
                watsonx_project_id="test",
                model="invalid_model",
            )

    def test_watsonx_invalid_params(self):
        # pass a parameter called top_m which is not valid
        with pytest.raises(KeyError):
            IBMwatsonx(
                api_key="test",
                watsonx_url="https://us-south.ml.cloud.ibm.com",
                watsonx_project_id="test",
                top_m=4,
            )
