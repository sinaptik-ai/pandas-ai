import unittest
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
from docker.errors import ImageNotFound
from pandasai_docker import DockerSandbox


class TestDockerSandbox(unittest.TestCase):
    def setUp(self):
        self.image_name = "test_image"
        self.dfs = [MagicMock()]

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    def test_destructor(self, mock_docker):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_container = mock_client.containers.run.return_value
        sandbox._container = mock_container

        del sandbox
        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    def test_image_exists(self, mock_docker):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_client.images.get.return_value = True
        self.assertTrue(sandbox._image_exists())

        mock_client.images.get.side_effect = ImageNotFound("Image not found")
        self.assertFalse(sandbox._image_exists())

    @patch("builtins.open")
    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    @patch("pandasai_docker.docker_sandbox.subprocess")
    def test_build_image(self, mock_subprocess, mock_docker, mock_open):
        # Create a single BytesIO object to mock the file content
        mock_file = MagicMock(spec=BytesIO)
        mock_file.__enter__.return_value = BytesIO(b"FROM python:3.9")
        mock_file.__exit__.return_value = None
        mock_open.return_value = mock_file

        # Arrange
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        dockerfile_path = sandbox._dockerfile_path
        image_name = self.image_name

        # Act
        sandbox._build_image()

        # Create the expected fileobj (using the same object reference)
        expected_fileobj = mock_file.__enter__.return_value

        # Assert
        mock_subprocess.run.assert_called_once()

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    def test_start_and_stop_container(self, mock_docker):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_client.containers = MagicMock()
        mock_client.containers.run = MagicMock(return_value=MagicMock())

        sandbox.start()
        mock_client.containers.run.assert_called_once_with(
            self.image_name,
            command="sleep infinity",
            network_disabled=True,
            detach=True,
            tty=True,
        )

        sandbox.stop()
        self.assertIsNone(sandbox._container)

    def test_extract_sql_queries_from_code(self):
        sandbox = DockerSandbox(image_name=self.image_name)
        code = """
sql_query = 'SELECT COUNT(*) FROM table'
result = execute_sql_query(sql_query)
        """
        queries = sandbox._extract_sql_queries_from_code(code)
        self.assertEqual(queries, ["SELECT COUNT(*) FROM table"])

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    def test_transfer_file(self, mock_docker):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_container = mock_client.containers.run.return_value
        sandbox._container = mock_container

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        sandbox.transfer_file(df, filename="test.csv")

        mock_container.put_archive.assert_called()

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    def test_exec_code(self, mock_docker):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_container = mock_client.containers.run.return_value
        mock_container.exec_run.return_value = (
            0,
            (b'{"type": "number", "value": 42}', b""),
        )
        sandbox._container = mock_container

        mock_execute_sql_func = MagicMock()
        env = {"execute_sql_query": mock_execute_sql_func}

        code = 'result = {"type": "number", "value": 42}'
        result = sandbox._exec_code(code, env)
        self.assertEqual(result, {"type": "number", "value": 42})

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    @patch("pandasai_docker.docker_sandbox.DockerSandbox.transfer_file")
    def test_exec_code_with_sql_queries(self, mock_transfer_file, mock_docker):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_container = mock_client.containers.run.return_value
        mock_container.exec_run.return_value = (
            0,
            (b'{"type": "number", "value": 42}', b""),
        )
        sandbox._container = mock_container

        # Mock SQL execution
        mock_execute_sql_func = MagicMock()
        env = {"execute_sql_query": mock_execute_sql_func}

        code = """
sql_query = 'SELECT COUNT(DISTINCT Artist) AS total_artists FROM artists'
total_artists_df = execute_sql_query(sql_query)
total_artists = total_artists_df['total_artists'].iloc[0]
result = {'type': 'number', 'value': total_artists}
        """
        result = sandbox._exec_code(code, env)
        self.assertEqual(result, {"type": "number", "value": 42})
        mock_execute_sql_func.assert_called_once_with(
            "SELECT COUNT(DISTINCT Artist) AS total_artists FROM artists"
        )

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    @patch("pandasai_docker.docker_sandbox.DockerSandbox.transfer_file")
    def test_exec_code_with_sql_queries_raise_no_env(
        self, mock_transfer_file, mock_docker
    ):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_container = mock_client.containers.run.return_value
        mock_container.exec_run.return_value = (
            0,
            (b'{"type": "number", "value": 42}', b""),
        )
        sandbox._container = mock_container

        # Mock SQL execution
        env = {}

        code = """
sql_query = 'SELECT COUNT(DISTINCT Artist) AS total_artists FROM artists'
total_artists_df = execute_sql_query(sql_query)
total_artists = total_artists_df['total_artists'].iloc[0]
result = {'type': 'number', 'value': total_artists}
        """
        with self.assertRaises(RuntimeError):
            sandbox._exec_code(code, env)

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    @patch("pandasai_docker.docker_sandbox.DockerSandbox.transfer_file")
    @patch("pandasai_docker.docker_sandbox.ResponseSerializer.deserialize")
    def test_exec_code_with_sql_queries_with_plot(
        self, mock_deserialize, mock_transfer_file, mock_docker
    ):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_container = mock_client.containers.run.return_value
        mock_container.exec_run.return_value = (
            0,
            (b'{"type": "plot", "value": "base64img"}', b""),
        )
        sandbox._container = mock_container

        # Mock SQL execution
        mock_execute_sql_func = MagicMock()
        env = {"execute_sql_query": mock_execute_sql_func}

        code = """
import pandas as pd
import matplotlib.pyplot as plt
sql_query = \"\"\"
SELECT Artist, Streams
FROM table_artists
ORDER BY CAST(REPLACE(Streams, ',', '') AS FLOAT) DESC
LIMIT 5
\"\"\"
top_artists_df = execute_sql_query(sql_query)
top_artists_df['Streams'] = top_artists_df['Streams'].str.replace(',', '').astype(float)
plt.figure(figsize=(10, 6))
plt.barh(top_artists_df['Artist'], top_artists_df['Streams'], color='skyblue')
plt.xlabel('Streams (in millions)')
plt.title('Top Five Artists by Streams')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('/exports/charts/temp_chart.png')
result = {'type': 'plot', 'value': '/exports/charts/temp_chart.png'}
        """
        result = sandbox._exec_code(code, env)

        assert result is not None
        mock_deserialize.assert_called_once_with(
            '{"type": "plot", "value": "base64img"}', "/exports/charts/temp_chart.png"
        )

    @patch("pandasai_docker.docker_sandbox.docker.from_env")
    @patch("pandasai_docker.docker_sandbox.DockerSandbox.transfer_file")
    @patch("pandasai_docker.docker_sandbox.ResponseSerializer.deserialize")
    def test_exec_code_with_sql_queries_with_dataframe(
        self, mock_deserialize, mock_transfer_file, mock_docker
    ):
        sandbox = DockerSandbox(image_name=self.image_name)
        mock_client = mock_docker.return_value
        mock_container = mock_client.containers.run.return_value
        mock_container.exec_run.return_value = (
            0,
            (
                b'{"type": "dataframe", "value": {"columns": [], "data": [], "index": []}}',
                b"",
            ),
        )
        sandbox._container = mock_container

        # Mock SQL execution
        mock_execute_sql_func = MagicMock()
        env = {"execute_sql_query": mock_execute_sql_func}

        code = """
import pandas as pd
import matplotlib.pyplot as plt
sql_query = \"\"\"
SELECT Artist, Streams
FROM table_artists
ORDER BY CAST(REPLACE(Streams, ',', '') AS FLOAT) DESC
LIMIT 5
\"\"\"
top_artists_df = execute_sql_query(sql_query)
result = {'type': 'dataframe', 'value': top_artists_df}
        """
        result = sandbox._exec_code(code, env)

        assert result is not None
        mock_deserialize.assert_called_once_with(
            '{"type": "dataframe", "value": {"columns": [], "data": [], "index": []}}',
            None,
        )

    def test_extract_sql_queries_from_code_with_bool_constant(self):
        sandbox = DockerSandbox(image_name=self.image_name)
        code = """
test = True
sql_query = 'SELECT COUNT(*) FROM table'
result = execute_sql_query(sql_query)
        """
        queries = sandbox._extract_sql_queries_from_code(code)
        self.assertEqual(queries, ["SELECT COUNT(*) FROM table"])

    def test_extract_sql_queries_from_code_with_cte(self):
        sandbox = DockerSandbox(image_name=self.image_name)
        code = """
test = True
sql_query = 'WITH temp AS (SELECT * FROM table) SELECT * FROM temp'
result = execute_sql_query(sql_query)
        """
        queries = sandbox._extract_sql_queries_from_code(code)
        self.assertEqual(
            queries, ["WITH temp AS (SELECT * FROM table) SELECT * FROM temp"]
        )

    def test_extract_sql_queries_from_code_with_malicious_query(self):
        sandbox = DockerSandbox(image_name=self.image_name)
        code = """
test = True
sql_query = 'DROP * FROM table'
result = execute_sql_query(sql_query)
        """
        queries = sandbox._extract_sql_queries_from_code(code)
        self.assertEqual(queries, [])


if __name__ == "__main__":
    unittest.main()
