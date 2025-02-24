import uuid
import warnings
from typing import List, Optional, Union

import pandas as pd

from pandasai.agent import Agent
from pandasai.dataframe.base import DataFrame

from ..config import Config


class SmartDatalake:
    def __init__(
        self,
        dfs: List[pd.DataFrame],
        config: Optional[Union[Config, dict]] = None,
    ):
        warnings.warn(
            "\n"
            + "*" * 80
            + "\n"
            + "\033[1;33mDEPRECATION WARNING:\033[0m\n"
            + "SmartDatalake will be deprecated soon. Use df.chat() instead.\n"
            + "*" * 80
            + "\n",
            DeprecationWarning,
            stacklevel=2,
        )
        dfs = self.load_dfs(dfs)
        self._agent = Agent(dfs, config=config)

    def load_dfs(self, dfs: List[pd.DataFrame]):
        load_dfs = []
        for df in dfs:
            if isinstance(df, pd.DataFrame):
                load_dfs.append(DataFrame(df))
            else:
                raise ValueError(
                    "Invalid input data. We cannot convert it to a dataframe."
                )
        return load_dfs

    def chat(self, query: str, output_type: Optional[str] = None):
        """
        Run a query on the dataframe.
        Args:
            query (str): Query to run on the dataframe
            output_type (Optional[str]): Add a hint for LLM which
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
                If none `output_type` is specified, the type can be any
                of the above or "text".
        Raises:
            ValueError: If the query is empty
        """
        return self._agent.chat(query, output_type)

    def clear_memory(self):
        """
        Clears the memory
        """
        self._agent.clear_memory()

    @property
    def last_prompt(self):
        return self._agent.last_prompt

    @property
    def last_prompt_id(self) -> uuid.UUID:
        """Return the id of the last prompt that was run."""
        if self._agent.last_prompt_id is None:
            raise ValueError("Pandas AI has not been run yet.")
        return self._agent.last_prompt_id

    @property
    def logs(self):
        return self._agent.logger.logs

    @property
    def logger(self):
        return self._agent.logger

    @logger.setter
    def logger(self, logger):
        self._agent.logger = logger

    @property
    def config(self):
        return self._agent.context.config

    @property
    def verbose(self):
        return self._agent.context.config.verbose

    @verbose.setter
    def verbose(self, verbose: bool):
        self._agent.context.config.verbose = verbose
        self._agent.logger.verbose = verbose

    @property
    def save_logs(self):
        return self._agent.context.config.save_logs

    @save_logs.setter
    def save_logs(self, save_logs: bool):
        self._agent.context.config.save_logs = save_logs
        self._agent.logger.save_logs = save_logs

    @property
    def custom_prompts(self):
        return self._agent.context.config.custom_prompts

    @custom_prompts.setter
    def custom_prompts(self, custom_prompts: dict):
        self._agent.context.config.custom_prompts = custom_prompts

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
    def last_code_generated(self):
        return self._agent.last_code_generated

    @property
    def last_code_executed(self):
        return self._agent.last_code_executed

    @property
    def last_result(self):
        return self._agent.last_result

    @property
    def last_error(self):
        return self._agent.last_error

    @property
    def dfs(self):
        return self._agent.context.dfs

    @property
    def memory(self):
        return self._agent.context.memory
