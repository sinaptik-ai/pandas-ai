import uuid
import warnings
from functools import cached_property
from io import StringIO
from typing import Any, List, Optional, Union

import pandas as pd

from pandasai.agent import Agent
from pandasai.dataframe.base import DataFrame

from ..config import Config
from ..helpers.logger import Logger


class SmartDataframe:
    _table_name: str
    _table_description: str
    _custom_head: str = None
    _original_import: any

    def __init__(
        self,
        df: pd.DataFrame,
        name: str = None,
        description: str = None,
        custom_head: pd.DataFrame = None,
        config: Config = None,
    ):
        warnings.warn(
            "\n"
            + "*" * 80
            + "\n"
            + "\033[1;33mDEPRECATION WARNING:\033[0m\n"
            + "SmartDataframe will soon be deprecated. Please use df.chat() instead.\n"
            + "*" * 80
            + "\n",
            DeprecationWarning,
            stacklevel=2,
        )

        self._original_import = df
        self.dataframe = self.load_df(df, name, description, custom_head)
        self._agent = Agent([self.dataframe], config=config)
        self._table_description = description
        self._table_name = name
        if custom_head is not None:
            self._custom_head = custom_head.to_csv(index=False)

    def load_df(self, df, name: str, description: str, custom_head: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            df = DataFrame(
                df,
                name=name,
                description=description,
            )
        else:
            raise ValueError("Invalid input data. We cannot convert it to a dataframe.")
        return df

    def chat(self, query: str, output_type: Optional[str] = None):
        """
        Run a query on the dataframe.
        Args:
            query (str): Query to run on the dataframe
            output_type (Optional[str]): Add a hint for LLM of which
                type should be returned by `analyze_data()` in generated
                code. Possible values: "number", "dataframe", "plot", "string":
                    * number - specifies that user expects to get a number
                        as a response object
                    * dataframe - specifies that user expects to get
                        pandas dataframe as a response object
                    * plot - specifies that user expects LLM to build
                        a plot
                    * string - specifies that user expects to get text
                        as a response object
        Raises:
            ValueError: If the query is empty
        """
        return self._agent.chat(query, output_type)

    @cached_property
    def head_df(self):
        """
        Get the head of the dataframe as a dataframe.
        Returns:
            pd.DataFrame: Pandas dataframe
        """
        return self.dataframe.get_head()

    @cached_property
    def head_csv(self):
        """
        Get the head of the dataframe as a CSV string.
        Returns:
            str: CSV string
        """
        df_head = self.dataframe.get_head()
        return df_head.to_csv(index=False)

    @property
    def last_prompt(self):
        return self._agent.last_prompt

    @property
    def last_prompt_id(self) -> uuid.UUID:
        return self._agent.last_prompt_id

    @property
    def last_code_generated(self):
        return self._agent.last_code_generated

    @property
    def last_code_executed(self):
        return self._agent.last_code_executed

    def original_import(self):
        return self._original_import

    @property
    def logger(self):
        return self._agent.logger

    @logger.setter
    def logger(self, logger: Logger):
        self._agent.logger = logger

    @property
    def logs(self):
        return self._agent.context.config.logs

    @property
    def verbose(self):
        return self._agent.context.config.verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        self._agent.context.config.verbose = verbose

    @property
    def save_logs(self):
        return self._agent.context.config.save_logs

    @save_logs.setter
    def save_logs(self, save_logs: bool):
        self._agent.context.config.save_logs = save_logs

    @property
    def save_charts(self):
        return self._agent.context.config.save_charts

    @save_charts.setter
    def save_charts(self, save_charts: bool):
        self._agent.context.config.save_charts = save_charts

    @property
    def save_charts_path(self):
        return self._agent.context.config.save_charts_path

    @save_charts_path.setter
    def save_charts_path(self, save_charts_path: str):
        self._agent.context.config.save_charts_path = save_charts_path

    @property
    def table_name(self):
        return self._table_name

    @property
    def table_description(self):
        return self._table_description

    @property
    def custom_head(self):
        data = StringIO(self._custom_head)
        return pd.read_csv(data)

    def __len__(self):
        return len(self.dataframe)

    def __eq__(self, other):
        return self.dataframe.equals(other.dataframe)

    def __getattr__(self, name):
        if name in self.dataframe.__dir__():
            return getattr(self.dataframe, name)
        else:
            return self.__getattribute__(name)

    def __getitem__(self, key):
        return self.dataframe.__getitem__(key)

    def __setitem__(self, key, value):
        return self.dataframe.__setitem__(key, value)


def load_smartdataframes(
    dfs: List[Union[pd.DataFrame, Any]], config: Config
) -> List[SmartDataframe]:
    """
    Load all the dataframes to be used in the smart datalake.
    Args:
        dfs (List[Union[pd.DataFrame, Any]]): List of dataframes to be used
    """
    smart_dfs = []
    for df in dfs:
        if not isinstance(df, SmartDataframe):
            smart_dfs.append(SmartDataframe(df, config=config))
        else:
            smart_dfs.append(df)
    return smart_dfs
