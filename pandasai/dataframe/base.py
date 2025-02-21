from __future__ import annotations

import hashlib
import os
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from zipfile import ZipFile

import pandas as pd
from pandas._typing import Axes, Dtype

import pandasai as pai
from pandasai import get_validated_dataset_path
from pandasai.config import Config, ConfigManager
from pandasai.constants import LOCAL_SOURCE_TYPES
from pandasai.core.response import BaseResponse
from pandasai.data_loader.semantic_layer_schema import (
    Column,
    SemanticLayerSchema,
    Source,
)
from pandasai.exceptions import DatasetNotFound, PandaAIApiKeyError
from pandasai.helpers.dataframe_serializer import DataframeSerializer
from pandasai.helpers.session import get_pandaai_session
from pandasai.sandbox.sandbox import Sandbox

if TYPE_CHECKING:
    from pandasai.agent.base import Agent


class DataFrame(pd.DataFrame):
    """
    PandaAI DataFrame that extends pandas DataFrame with natural language capabilities.

    Attributes:
        name (Optional[str]): Name of the dataframe
        description (Optional[str]): Description of the dataframe
        schema (Optional[SemanticLayerSchema]): Schema definition for the dataframe
        config (Config): Configuration settings
    """

    _metadata = [
        "_agent",
        "_column_hash",
        "_table_name",
        "config",
        "path",
        "schema",
    ]

    def __init__(
        self,
        data=None,
        index: Axes | None = None,
        columns: Axes | None = None,
        dtype: Dtype | None = None,
        copy: bool | None = None,
        **kwargs,
    ) -> None:
        _schema: Optional[SemanticLayerSchema] = kwargs.pop("schema", None)
        _path: Optional[str] = kwargs.pop("path", None)
        _table_name: Optional[str] = kwargs.pop("_table_name", None)

        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )

        if _table_name:
            self._table_name = _table_name

        self._column_hash = self._calculate_column_hash()
        self.schema = _schema or DataFrame.get_default_schema(self)
        self.path = _path
        self._agent: Optional[Agent] = None

    def __repr__(self) -> str:
        """Return a string representation of the DataFrame."""
        name_str = f"name='{self.schema.name}'"
        desc_str = (
            f"description='{self.schema.description}'"
            if self.schema.description
            else ""
        )
        metadata = ", ".join(filter(None, [name_str, desc_str]))

        return f"PandaAI DataFrame({metadata})\n{super().__repr__()}"

    def _calculate_column_hash(self):
        column_string = ",".join(self.columns)
        return hashlib.md5(column_string.encode()).hexdigest()

    @property
    def column_hash(self):
        return self._column_hash

    @property
    def type(self) -> str:
        return "pd.DataFrame"

    def chat(self, prompt: str, sandbox: Optional[Sandbox] = None) -> BaseResponse:
        """
        Interact with the DataFrame using natural language.

        Args:
            prompt (str): The natural language query or instruction.
            sandbox (Sandbox, optional): The sandbox to execute code securely.

        Returns:
            str: The response to the prompt.
        """
        if self._agent is None:
            from pandasai.agent import (
                Agent,
            )

            self._agent = Agent([self], sandbox=sandbox)

        return self._agent.chat(prompt)

    def follow_up(self, query: str, output_type: Optional[str] = None):
        if self._agent is None:
            raise ValueError(
                "No existing conversation. Please use chat() to start a new conversation."
            )
        return self._agent.follow_up(query, output_type)

    @property
    def rows_count(self) -> int:
        return len(self)

    @property
    def columns_count(self) -> int:
        return len(self.columns)

    def serialize_dataframe(self) -> str:
        """
        Serialize DataFrame to string representation.

        Returns:
            str: Serialized string representation of the DataFrame
        """
        source = self.schema.source or None

        if source:
            dialect = "duckdb" if source.type in LOCAL_SOURCE_TYPES else source.type
        else:
            dialect = "postgres"

        return DataframeSerializer.serialize(self, dialect)

    def get_head(self):
        return self.head()

    def push(self):
        if self.path is None:
            raise ValueError(
                "Please save the dataset before pushing to the remote server."
            )

        SemanticLayerSchema.model_validate(self.schema)
        org_name, dataset_name = get_validated_dataset_path(self.path)

        api_key = os.environ.get("PANDABI_API_KEY", None)

        request_session = get_pandaai_session()

        params = {
            "path": f"{org_name}/{dataset_name}",
            "description": self.schema.description,
            "name": self.schema.name,
        }

        file_manager = ConfigManager.get().file_manager
        headers = {"accept": "application/json", "x-authorization": f"Bearer {api_key}"}

        schema_file_path = os.path.join(self.path, "schema.yaml")
        data_file_path = os.path.join(self.path, "data.parquet")

        # Open schema.yaml
        schema_file = file_manager.load_binary(schema_file_path)

        files = [("files", ("schema.yaml", schema_file, "application/x-yaml"))]

        # Check if data.parquet exists and open it
        if file_manager.exists(data_file_path):
            data_file = file_manager.load_binary(data_file_path)
            files.append(
                ("files", ("data.parquet", data_file, "application/octet-stream"))
            )

        # Send the POST request
        request_session.post(
            "/datasets/push",
            files=files,
            params=params,
            headers=headers,
        )

        print("Your dataset was successfully pushed to the remote server!")
        print(f"🔗 URL: https://app.pandabi.ai/datasets/{self.path}")

    def pull(self):
        api_key = os.environ.get("PANDABI_API_KEY", None)

        if not api_key:
            raise PandaAIApiKeyError()

        request_session = get_pandaai_session()

        headers = {"accept": "application/json", "x-authorization": f"Bearer {api_key}"}

        file_data = request_session.get(
            "/datasets/pull", headers=headers, params={"path": self.path}
        )
        if file_data.status_code != 200:
            raise DatasetNotFound("Remote dataset not found to pull!")

        with ZipFile(BytesIO(file_data.content)) as zip_file:
            for file_name in zip_file.namelist():
                target_path = os.path.join(self.path, file_name)

                file_manager = ConfigManager.get().file_manager
                # Check if the file already exists
                if file_manager.exists(target_path):
                    print(f"Replacing existing file: {target_path}")

                # Ensure target directory exists
                file_manager.mkdir(os.path.dirname(target_path))

                # Extract the file
                file_manager.write_binary(target_path, zip_file.read(file_name))

        # Reloads the Dataframe
        from pandasai import DatasetLoader

        dataset_loader = DatasetLoader.create_loader_from_path(self.path)
        df = dataset_loader.load()
        self.__init__(
            df,
            schema=df.schema,
            data_loader=dataset_loader,
            path=df.path,
        )

        print(f"Dataset pulled successfully from path: {self.path}")

    @staticmethod
    def get_column_type(column_dtype) -> Optional[str]:
        """
        Map pandas dtype to a valid column type.
        """
        if pd.api.types.is_string_dtype(column_dtype):
            return "string"
        elif pd.api.types.is_integer_dtype(column_dtype):
            return "integer"
        elif pd.api.types.is_float_dtype(column_dtype):
            return "float"
        elif pd.api.types.is_datetime64_any_dtype(column_dtype):
            return "datetime"
        elif pd.api.types.is_bool_dtype(column_dtype):
            return "boolean"
        else:
            return None

    @classmethod
    def get_default_schema(cls, dataframe: DataFrame) -> SemanticLayerSchema:
        columns_list = [
            Column(name=str(name), type=DataFrame.get_column_type(dtype))
            for name, dtype in dataframe.dtypes.items()
        ]

        table_name = getattr(
            dataframe, "_table_name", f"table_{dataframe._column_hash}"
        )

        return SemanticLayerSchema(
            name=table_name,
            source=Source(
                type="parquet",
                path="data.parquet",
            ),
            columns=columns_list,
        )
