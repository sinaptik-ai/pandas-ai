import io
import os
import zipfile
from unittest.mock import MagicMock, mock_open, patch

import pytest

import pandasai
from pandasai.data_loader.semantic_layer_schema import Column, SemanticLayerSchema
from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import DatasetNotFound, InvalidConfigError, PandaAIApiKeyError
from pandasai.helpers.filemanager import DefaultFileManager
from pandasai.llm.bamboo_llm import BambooLLM


def create_test_zip():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("test.csv", "a,b,c\n1,2,3")
    return zip_buffer.getvalue()


class TestPandaAIInit:
    @pytest.fixture
    def mysql_connection_json(self):
        return {
            "type": "mysql",
            "connection": {
                "host": "localhost",
                "port": 3306,
                "database": "test_db",
                "user": "test_user",
                "password": "test_password",
            },
            "table": "countries",
        }

    @pytest.fixture
    def postgresql_connection_json(self):
        return {
            "type": "postgres",
            "connection": {
                "host": "localhost",
                "port": 3306,
                "database": "test_db",
                "user": "test_user",
                "password": "test_password",
            },
            "table": "countries",
        }

    @pytest.fixture
    def sqlite_connection_json(self):
        return {"type": "sqlite", "path": "/path/to/database.db", "table": "countries"}

    def test_chat_creates_agent(self, sample_df):
        with patch("pandasai.Agent") as MockAgent:
            pandasai.chat("Test query", sample_df)
            MockAgent.assert_called_once_with([sample_df], sandbox=None)

    def test_chat_sandbox_passed_to_agent(self, sample_df):
        with patch("pandasai.Agent") as MockAgent:
            sandbox = MagicMock()
            pandasai.chat("Test query", sample_df, sandbox=sandbox)
            MockAgent.assert_called_once_with([sample_df], sandbox=sandbox)

    def test_chat_without_dataframes_raises_error(self):
        with pytest.raises(ValueError, match="At least one dataframe must be provided"):
            pandasai.chat("Test query")

    def test_follow_up_without_chat_raises_error(self):
        pandasai._current_agent = None
        with pytest.raises(ValueError, match="No existing conversation"):
            pandasai.follow_up("Follow-up query")

    def test_follow_up_after_chat(self, sample_df):
        with patch("pandasai.Agent") as MockAgent:
            mock_agent = MockAgent.return_value
            pandasai.chat("Test query", sample_df)
            pandasai.follow_up("Follow-up query")
            mock_agent.follow_up.assert_called_once_with("Follow-up query")

    def test_chat_with_multiple_dataframes(self, sample_dataframes):
        with patch("pandasai.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.chat.return_value = "Mocked response"

            result = pandasai.chat("What is the sum of column A?", *sample_dataframes)

            MockAgent.assert_called_once_with(sample_dataframes, sandbox=None)
            mock_agent_instance.chat.assert_called_once_with(
                "What is the sum of column A?"
            )
            assert result == "Mocked response"

    def test_chat_with_single_dataframe(self, sample_dataframes):
        with patch("pandasai.Agent") as MockAgent:
            mock_agent_instance = MagicMock()
            MockAgent.return_value = mock_agent_instance
            mock_agent_instance.chat.return_value = "Mocked response"

            result = pandasai.chat(
                "What is the average of column X?", sample_dataframes[1]
            )

            MockAgent.assert_called_once_with([sample_dataframes[1]], sandbox=None)
            mock_agent_instance.chat.assert_called_once_with(
                "What is the average of column X?"
            )
            assert result == "Mocked response"

    @patch("pandasai.helpers.path.find_project_root")
    @patch("os.path.exists")
    def test_load_valid_dataset(
        self, mock_exists, mock_find_project_root, mock_loader_instance, sample_schema
    ):
        """Test loading a valid dataset."""

        mock_find_project_root.return_value = os.path.join("mock", "root")
        mock_exists.return_value = True

        dataset_path = "org/dataset-name"
        result = pandasai.load(dataset_path)

        # Verify the class method was called
        mock_loader_instance.load.assert_called_once()
        assert result.equals(mock_loader_instance.load.return_value)

    @patch("zipfile.ZipFile")
    @patch("io.BytesIO")
    @patch("os.environ")
    def test_load_dataset_not_found(self, mockenviron, mock_bytes_io, mock_zip_file):
        """Test loading when dataset does not exist locally and API returns not found."""
        mockenviron.return_value = {"PANDABI_API_URL": "localhost:8000"}
        mock_request_session = MagicMock()
        pandasai.get_pandaai_session = mock_request_session
        pandasai.get_pandaai_session.return_value = MagicMock()
        mock_request_session.get.return_value.status_code = 404

        dataset_path = "org/dataset-name"

        with pytest.raises(DatasetNotFound):
            pandasai.load(dataset_path)

    @patch("pandasai.os.path.exists")
    @patch("pandasai.os.environ", {})
    @patch("pandasai.get_pandaai_session")
    def test_load_missing_not_found_locally_and_no_remote_key(
        self, mock_session, mock_exists
    ):
        """Test loading when API URL is missing."""
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.return_value.get.return_value = mock_response
        dataset_path = "org/dataset-name"

        with pytest.raises(
            PandaAIApiKeyError,
            match='The dataset "org/dataset-name" does not exist in your local datasets directory. In addition, no API Key has been provided. Set an API key with valid permits if you want to fetch the dataset from the remote server.',
        ):
            pandasai.load(dataset_path)

    @patch("pandasai.os.path.exists")
    @patch("pandasai.os.environ", {"PANDABI_API_KEY": "key"})
    def test_load_missing_api_url(self, mock_exists):
        """Test loading when API URL is missing."""
        mock_exists.return_value = False
        dataset_path = "org/dataset-name"

        with pytest.raises(DatasetNotFound):
            pandasai.load(dataset_path)

    @patch("pandasai.os.path.exists")
    @patch("pandasai.os.environ", {"PANDABI_API_KEY": "key"})
    @patch("pandasai.get_pandaai_session")
    def test_load_missing_not_found(self, mock_session, mock_exists):
        """Test loading when API URL is missing."""
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.return_value.get.return_value = mock_response
        dataset_path = "org/dataset-name"

        with pytest.raises(DatasetNotFound):
            pandasai.load(dataset_path)

    @patch("pandasai.os.environ", new_callable=dict)
    @patch("pandasai.os.path.exists")
    @patch("pandasai.get_pandaai_session")
    @patch("pandasai.ZipFile")
    @patch("pandasai.BytesIO")
    def test_load_successful_zip_extraction(
        self,
        mock_bytes_io,
        mock_zip_file,
        mock_get_pandaai_session,
        mock_exists,
        mock_os_environ,
        mock_loader_instance,
    ):
        """Test loading when dataset is not found locally but is successfully downloaded."""
        mock_exists.return_value = False
        mock_os_environ.update({"PANDABI_API_KEY": "key", "PANDABI_API_URL": "url"})
        mock_request_session = MagicMock()
        mock_get_pandaai_session.return_value = mock_request_session
        mock_request_session.get.return_value.status_code = 200
        mock_request_session.get.return_value.content = b"mock zip content"

        dataset_path = "org/dataset-name"

        # Mock the zip file extraction
        mock_zip_file.return_value.__enter__.return_value.extractall = MagicMock()

        result = pandasai.load(dataset_path)

        mock_zip_file.return_value.__enter__.return_value.extractall.assert_called_once()
        assert isinstance(result, DataFrame)

    @patch("pandasai.os.environ", {})
    def test_load_without_api_credentials(
        self,
    ):
        """Test that load raises PandaAIApiKeyError when no API credentials are provided"""
        with pytest.raises(PandaAIApiKeyError) as exc_info:
            pandasai.load("test/dataset")
        assert (
            str(exc_info.value)
            == 'The dataset "test/dataset" does not exist in your local datasets directory. In addition, no API Key has been provided. Set an API key with valid permits if you want to fetch the dataset from the remote server.'
        )

    def test_load_invalid_name(self):
        with pytest.raises(
            ValueError,
            match="Organization name must be lowercase and use hyphens instead of spaces",
        ):
            pandasai.load("test_test/data_set")

    @patch.dict(os.environ, {"PANDABI_API_KEY": "test-key"})
    @patch("pandasai.get_pandaai_session")
    @patch("pandasai.os.path.exists")
    @patch("pandasai.helpers.path.find_project_root")
    @patch("pandasai.os.makedirs")
    def test_load_with_default_api_url(
        self, mock_makedirs, mock_root, mock_exists, mock_session, mock_loader_instance
    ):
        """Test that load uses DEFAULT_API_URL when no URL is provided"""
        mock_root.return_value = "/tmp/test_project"
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = create_test_zip()
        mock_session.return_value.get.return_value = mock_response

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("zipfile.ZipFile") as mock_zip_file:
                mock_zip_file.return_value.__enter__.return_value.extractall = (
                    MagicMock()
                )
                pandasai.load("org/dataset")

        mock_session.return_value.get.assert_called_once_with(
            "/datasets/pull",
            headers={
                "accept": "application/json",
                "x-authorization": "Bearer test-key",
            },
            params={"path": "org/dataset"},
        )

    @patch.dict(
        os.environ,
        {"PANDABI_API_KEY": "test-key", "PANDABI_API_URL": "https://custom.api.url"},
    )
    @patch("pandasai.get_pandaai_session")
    @patch("pandasai.os.path.exists")
    @patch("pandasai.helpers.path.find_project_root")
    @patch("pandasai.os.makedirs")
    def test_load_with_custom_api_url(
        self, mock_makedirs, mock_root, mock_exists, mock_session, mock_loader_instance
    ):
        """Test that load uses custom URL from environment"""
        mock_root.return_value = "/tmp/test_project"
        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = create_test_zip()
        mock_session.return_value.get.return_value = mock_response

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("zipfile.ZipFile") as mock_zip_file:
                mock_zip_file.return_value.__enter__.return_value.extractall = (
                    MagicMock()
                )
                pandasai.load("org/dataset")

        mock_session.return_value.get.assert_called_once_with(
            "/datasets/pull",
            headers={
                "accept": "application/json",
                "x-authorization": "Bearer test-key",
            },
            params={"path": "org/dataset"},
        )

    def test_create_valid_dataset_no_params(
        self, sample_df, mock_loader_instance, mock_file_manager
    ):
        """Test creating a dataset with valid inputs."""
        with patch.object(sample_df, "to_parquet") as mock_to_parquet:
            result = pandasai.create("test-org/test-dataset", sample_df)

            # Check if directories were created
            mock_file_manager.mkdir.assert_called_once_with(
                os.path.join("test-org", "test-dataset")
            )

            # Check if DataFrame was saved
            mock_to_parquet.assert_called_once()
            assert mock_to_parquet.call_args[0][0].endswith("data.parquet")
            assert mock_to_parquet.call_args[1]["index"] is False

            # Check if schema was saved
            mock_file_manager.write.assert_called_once()

            # Check returned DataFrame
            assert isinstance(result, DataFrame)
            assert result.schema.name == sample_df.schema.name
            assert result.schema.description is None
            assert mock_loader_instance.load.call_count == 1

    def test_create_valid_dataset_group_by(
        self, sample_df, mock_loader_instance, mock_file_manager
    ):
        """Test creating a dataset with valid inputs."""
        with patch.object(sample_df, "to_parquet") as mock_to_parquet:
            result = pandasai.create(
                "test-org/test-dataset",
                sample_df,
                columns=[
                    {"name": "A"},
                    {"name": "B", "expression": "avg(B)", "alias": "average_b"},
                ],
                group_by=["A"],
            )
            assert result.schema.group_by == ["A"]

    def test_create_invalid(self, sample_df, mock_loader_instance, mock_file_manager):
        """Test creating a dataset with valid inputs."""
        with pytest.raises(InvalidConfigError):
            pandasai.create("test-org/test-dataset")

    def test_create_invalid_path_format(self, sample_df):
        """Test creating a dataset with invalid path format."""
        with pytest.raises(
            ValueError, match="Path must be in format 'organization/dataset'"
        ):
            pandasai.create("invalid_path", sample_df)

    def test_create_invalid_org_name(self, sample_df):
        """Test creating a dataset with invalid organization name."""
        with pytest.raises(ValueError, match="Organization name must be lowercase"):
            pandasai.create("Invalid-Org/test-dataset", sample_df)

    def test_create_invalid_dataset_name(self, sample_df):
        """Test creating a dataset with invalid dataset name."""
        with pytest.raises(ValueError, match="Dataset path name must be lowercase"):
            pandasai.create("test-org/Invalid-Dataset", sample_df)

    def test_create_empty_org_name(self, sample_df):
        """Test creating a dataset with empty organization name."""
        with pytest.raises(
            ValueError, match="Both organization and dataset names are required"
        ):
            pandasai.create("/test-dataset", sample_df)

    def test_create_empty_dataset_name(self, sample_df):
        """Test creating a dataset with empty dataset name."""
        with pytest.raises(
            ValueError, match="Both organization and dataset names are required"
        ):
            pandasai.create("test-org/", sample_df)

    @patch("pandasai.helpers.path.find_project_root")
    def test_create_existing_dataset(self, mock_find_project_root, sample_df, llm):
        """Test creating a dataset that already exists."""
        mock_find_project_root.return_value = os.path.join("mock", "root")

        with patch("os.path.exists") as mock_exists:
            # Mock that both directory and schema file exist
            mock_exists.side_effect = lambda path: True

            with pytest.raises(
                ValueError,
                match="Dataset already exists at path: test-org/test-dataset",
            ):
                pandasai.config.set(
                    {
                        "llm": llm,
                    }
                )
                pandasai.create("test-org/test-dataset", sample_df)

    @patch("pandasai.helpers.path.find_project_root")
    def test_create_existing_directory_no_dataset(
        self, mock_find_project_root, sample_df, mock_loader_instance
    ):
        """Test creating a dataset in an existing directory but without existing dataset files."""
        mock_find_project_root.return_value = os.path.join("mock", "root")

        def mock_exists_side_effect(path):
            # Return True for directory, False for schema and data files
            return not (path.endswith("schema.yaml") or path.endswith("data.parquet"))

        with patch("os.path.exists", side_effect=mock_exists_side_effect), patch(
            "os.makedirs"
        ) as mock_makedirs, patch(
            "builtins.open", mock_open()
        ) as mock_file, patch.object(sample_df, "to_parquet") as mock_to_parquet, patch(
            "pandasai.find_project_root", return_value=os.path.join("mock", "root")
        ):
            result = pandasai.create("test-org/test-dataset", sample_df)

            # Verify dataset was created successfully
            assert isinstance(result, DataFrame)
            assert result.schema.name == sample_df.schema.name
            mock_to_parquet.assert_called_once()
            mock_makedirs.assert_called_once()
            mock_file.assert_called_once()
            mock_loader_instance.load.assert_called_once()

    def test_create_valid_dataset_with_description(
        self, sample_df, mock_loader_instance, mock_file_manager
    ):
        """Test creating a dataset with valid inputs."""

        from pandasai.data_loader.semantic_layer_schema import Source

        schema = SemanticLayerSchema(
            name="test_dataset",
            description="test_description",
            source=Source(type="parquet", path="data.parquet"),
        )
        sample_df.schema = schema

        with patch.object(sample_df, "to_parquet") as mock_to_parquet:
            result = pandasai.create(
                "test-org/test-dataset", sample_df, description="test_description"
            )

            # Check if directories were created
            mock_file_manager.mkdir.assert_called_once_with(
                os.path.join("test-org", "test-dataset")
            )

            # Check if DataFrame was saved
            mock_to_parquet.assert_called_once()
            assert mock_to_parquet.call_args[0][0].endswith("data.parquet")
            assert mock_to_parquet.call_args[1]["index"] is False

            # Check if schema was saved
            mock_file_manager.write.assert_called_once()

            # Check returned DataFrame
            assert isinstance(result, DataFrame)
            assert result.schema.name == sample_df.schema.name
            assert result.schema.description == "test_description"
            mock_loader_instance.load.assert_called_once()

    def test_create_valid_dataset_with_columns(
        self, sample_df, mock_loader_instance, mock_file_manager
    ):
        """Test creating a dataset with valid inputs."""

        with patch.object(sample_df, "to_parquet") as mock_to_parquet:
            columns_dict = [{"name": "a"}, {"name": "b"}]
            result = pandasai.create(
                "test-org/test-dataset", sample_df, columns=columns_dict
            )

            # Check if directories were created
            mock_file_manager.mkdir.assert_called_once_with(
                os.path.join("test-org", "test-dataset")
            )

            # Check if DataFrame was saved
            mock_to_parquet.assert_called_once()
            assert mock_to_parquet.call_args[0][0].endswith("data.parquet")
            assert mock_to_parquet.call_args[1]["index"] is False

            # Check if schema was saved
            mock_file_manager.write.assert_called_once()

            # Check returned DataFrame
            assert isinstance(result, DataFrame)
            assert result.schema.name == sample_df.schema.name
            assert result.schema.description is None
            assert result.schema.columns == list(
                map(lambda column: Column(**column), columns_dict)
            )
            mock_loader_instance.load.assert_called_once()

    @patch("pandasai.helpers.path.find_project_root")
    @patch("os.makedirs")
    def test_create_dataset_wrong_columns(
        self, mock_makedirs, mock_find_project_root, sample_df, mock_file_manager
    ):
        """Test creating a dataset with valid inputs."""
        mock_find_project_root.return_value = os.path.join("mock", "root")

        with patch("builtins.open", mock_open()) as mock_file, patch.object(
            sample_df, "to_parquet"
        ) as mock_to_parquet, patch(
            "pandasai.find_project_root", return_value=os.path.join("mock", "root")
        ):
            columns_dict = [{"no-name": "a"}, {"name": "b"}]

            with pytest.raises(ValueError):
                pandasai.create(
                    "test-org/test-dataset", sample_df, columns=columns_dict
                )

    def test_create_valid_dataset_with_mysql(
        self, sample_df, mysql_connection_json, mock_loader_instance, mock_file_manager
    ):
        """Test creating a dataset with valid inputs."""

        with patch("builtins.open", mock_open()) as mock_file, patch.object(
            sample_df, "to_parquet"
        ) as mock_to_parquet, patch(
            "pandasai.find_project_root", return_value=os.path.join("mock", "root")
        ):
            columns_dict = [{"name": "a"}, {"name": "b"}]
            result = pandasai.create(
                "test-org/test-dataset",
                source=mysql_connection_json,
                columns=columns_dict,
            )

            # Check if directories were created
            mock_file_manager.mkdir.assert_called_once_with(
                os.path.join("test-org", "test-dataset")
            )

            # Check returned DataFrame
            assert isinstance(result, DataFrame)
            assert result.schema.name == sample_df.schema.name
            assert result.schema.description is None
            assert mock_loader_instance.load.call_count == 1

    def test_create_valid_dataset_with_postgres(
        self, sample_df, mysql_connection_json, mock_loader_instance, mock_file_manager
    ):
        with patch("builtins.open", mock_open()) as mock_file, patch.object(
            sample_df, "to_parquet"
        ) as mock_to_parquet, patch(
            "pandasai.find_project_root", return_value=os.path.join("mock", "root")
        ):
            columns_dict = [{"name": "a"}, {"name": "b"}]
            result = pandasai.create(
                "test-org/test-dataset",
                source=mysql_connection_json,
                columns=columns_dict,
            )

            # Check returned DataFrame
            assert isinstance(result, DataFrame)
            assert result.schema.name == sample_df.schema.name
            assert result.schema.description is None
            assert mock_loader_instance.load.call_count == 1

    @patch("pandasai.helpers.path.find_project_root")
    @patch("os.makedirs")
    def test_create_with_no_dataframe_and_connector(
        self, mock_makedirs, mock_find_project_root, mock_file_manager
    ):
        with pytest.raises(
            InvalidConfigError,
            match="Please provide either a DataFrame, a Source or a View",
        ):
            pandasai.create("test-org/test-dataset")

    @patch("pandasai.helpers.path.find_project_root")
    @patch("os.makedirs")
    def test_create_with_no_dataframe_with_incorrect_type(
        self,
        mock_makedirs,
        mock_find_project_root,
    ):
        with pytest.raises(ValueError, match="df must be a PandaAI DataFrame"):
            pandasai.create("test-org/test-dataset", df={"test": "test"})

    def test_create_valid_view(
        self, sample_df, mock_loader_instance, mock_file_manager
    ):
        """Test creating a dataset with valid inputs."""

        with patch("builtins.open", mock_open()) as mock_file, patch(
            "pandasai.find_project_root", return_value=os.path.join("mock", "root")
        ):
            columns = [
                {
                    "name": "parents.id",
                },
                {
                    "name": "parents.name",
                },
                {
                    "name": "children.name",
                },
            ]

            relations = [{"from": "parents.id", "to": "children.parent_id"}]

            result = pandasai.create(
                "test-org/test-dataset", columns=columns, relations=relations, view=True
            )

            # Check returned DataFrame
            assert isinstance(result, DataFrame)
            assert result.schema.name == sample_df.schema.name
            assert result.schema.description is None
            assert mock_loader_instance.load.call_count == 1

    def test_config_change_after_df_creation(
        self, sample_df, mock_loader_instance, llm
    ):
        with patch.object(sample_df, "to_parquet") as mock_to_parquet, patch(
            "pandasai.core.code_generation.base.CodeGenerator.validate_and_clean_code"
        ) as mock_validate_and_clean_code, patch(
            "pandasai.agent.base.Agent.execute_code"
        ) as mock_execute_code:
            # Check if directories were created

            # mock file manager to without mocking complete config
            class MockFileManager(DefaultFileManager):
                def exists(self, path):
                    return False

            mock_file_manager = MockFileManager()
            pandasai.config.set(
                {
                    "file_manager": mock_file_manager,
                }
            )

            df = pandasai.create("test-org/test-dataset", sample_df)

            # set code generation output
            llm.generate_code = MagicMock()
            llm.generate_code.return_value = (
                'df=execute_sql_query("select * from table")'
            )

            mock_execute_code.return_value = {"type": "number", "value": 42}

            assert isinstance(pandasai.config.get().llm, BambooLLM)

            pandasai.config.set({"llm": llm})

            df.chat("test")

            llm.generate_code.assert_called_once()
