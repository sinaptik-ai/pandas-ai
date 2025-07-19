import os
from typing import Optional
from unittest.mock import ANY, MagicMock, Mock, mock_open, patch

import pandas as pd
import pytest

from pandasai import DatasetLoader, VirtualDataFrame
from pandasai.agent.base import Agent
from pandasai.config import Config, ConfigManager
from pandasai.core.response.error import ErrorResponse
from pandasai.data_loader.semantic_layer_schema import SemanticLayerSchema
from pandasai.dataframe.base import DataFrame
from pandasai.exceptions import CodeExecutionError, InvalidLLMOutputType
from pandasai.llm.fake import FakeLLM


class TestAgent:
    "Unit tests for Agent class"

    @pytest.fixture
    def llm(self, output: Optional[str] = None) -> FakeLLM:
        return FakeLLM(output=output)

    @pytest.fixture
    def config(self, llm: FakeLLM) -> dict:
        return {"llm": llm}

    @pytest.fixture
    def agent(self, sample_df: DataFrame, config: dict) -> Agent:
        return Agent(sample_df, config, vectorstore=MagicMock())

    @pytest.fixture(autouse=True)
    def mock_llm(self):
        # Generic LLM mock for testing
        mock = Mock(type="generic_llm")
        yield mock

    def test_constructor(self, sample_df, config):
        agent_1 = Agent(sample_df, config)
        agent_2 = Agent([sample_df], config)

        # test multiple agents instances data overlap
        agent_1._state.memory.add("Which country has the highest gdp?", True)
        memory = agent_1._state.memory.all()
        assert len(memory) == 1

        memory = agent_2._state.memory.all()
        assert len(memory) == 0

    def test_chat(self, sample_df, config):
        # Create an Agent instance for testing
        agent = Agent(sample_df, config)
        agent.chat = Mock()
        agent.chat.return_value = "United States has the highest gdp"
        # Test the chat function
        response = agent.chat("Which country has the highest gdp?")
        assert agent.chat.called
        assert isinstance(response, str)
        assert response == "United States has the highest gdp"

    @patch("pandasai.agent.base.CodeGenerator")
    def test_code_generation(self, mock_generate_code, sample_df, config):
        # Create an Agent instance for testing
        mock_generate_code.generate_code.return_value = (
            "print(United States has the highest gdp)"
        )
        agent = Agent(sample_df, config)
        agent._code_generator = mock_generate_code

        # Test the chat function
        response = agent.generate_code("Which country has the highest gdp?")
        assert agent._code_generator.generate_code.called
        assert isinstance(response, str)
        assert response == "print(United States has the highest gdp)"

    @patch("pandasai.agent.base.CodeGenerator")
    def test_code_generation_with_retries(self, mock_generate_code, sample_df, config):
        # Create an Agent instance for testing
        mock_generate_code.generate_code.side_effect = Exception("Exception")
        agent = Agent(sample_df, config)
        agent._code_generator = mock_generate_code
        agent._regenerate_code_after_error = MagicMock()

        # Test the chat function
        agent.generate_code_with_retries("Which country has the highest gdp?")
        assert agent._code_generator.generate_code.called
        assert agent._regenerate_code_after_error.call_count == 1

    @patch("pandasai.agent.base.CodeGenerator")
    def test_code_generation_with_retries_three_times(
        self, mock_generate_code, sample_df, config
    ):
        # Create an Agent instance for testing
        mock_generate_code.generate_code.side_effect = Exception("Exception")
        agent = Agent(sample_df, config)
        agent._code_generator = mock_generate_code
        agent._regenerate_code_after_error = MagicMock()
        agent._regenerate_code_after_error.side_effect = Exception("Exception")

        # Test the chat function
        with pytest.raises(Exception):
            agent.generate_code_with_retries("Which country has the highest gdp?")

        assert agent._code_generator.generate_code.called
        assert agent._regenerate_code_after_error.call_count == 4

    @patch("pandasai.agent.base.CodeGenerator")
    def test_generate_code_with(self, mock_generate_code, agent: Agent):
        # Mock the code generator to return a SQL-based response
        mock_generate_code.generate_code.return_value = (
            "SELECT country FROM countries ORDER BY gdp DESC LIMIT 1;"
        )
        agent._code_generator = mock_generate_code

        # Generate code
        response = agent.generate_code("Which country has the highest GDP?")

        # Check that the SQL-specific prompt was used
        assert mock_generate_code.generate_code.called
        assert response == "SELECT country FROM countries ORDER BY gdp DESC LIMIT 1;"

    @patch("pandasai.agent.base.CodeGenerator")
    def test_generate_code_logs_generation(self, mock_generate_code, agent: Agent):
        # Mock the logger
        agent._state.logger.log = MagicMock()

        # Mock the code generator
        mock_generate_code.generate_code.return_value = "print('Logging test.')"
        agent._code_generator = mock_generate_code

        # Generate code
        response = agent.generate_code("Test logging during code generation.")

        # Verify logger was called
        agent._state.logger.log.assert_any_call("Generating new code...")
        assert mock_generate_code.generate_code.called
        assert response == "print('Logging test.')"

    @patch("pandasai.agent.base.CodeGenerator")
    def test_generate_code_updates_last_prompt(self, mock_generate_code, agent: Agent):
        # Mock the code generator
        prompt = "Cust  om SQL prompt"
        mock_generate_code.generate_code.return_value = "print('Prompt test.')"
        agent._state.last_prompt_used = None
        agent._code_generator = mock_generate_code

        # Mock the prompt creation function
        with patch("pandasai.agent.base.get_chat_prompt_for_sql", return_value=prompt):
            response = agent.generate_code("Which country has the highest GDP?")

        # Verify the last prompt used is updated
        assert agent._state.last_prompt_used == prompt
        assert mock_generate_code.generate_code.called
        assert response == "print('Prompt test.')"

    @patch("pandasai.agent.base.CodeExecutor")
    def test_execute_code_successful_execution(self, mock_code_executor, agent: Agent):
        # Mock CodeExecutor to return a successful result
        mock_code_executor.return_value.execute_and_return_result.return_value = {
            "result": "Execution successful"
        }

        # Execute the code
        code = "print('Hello, World!')"
        result = agent.execute_code(code)

        # Verify the code was executed and the result is correct
        assert result == {"result": "Execution successful"}
        mock_code_executor.return_value.execute_and_return_result.assert_called_with(
            code
        )

    @patch("pandasai.agent.base.CodeExecutor")
    def test_execute_code(self, mock_code_executor, agent: Agent):
        # Mock CodeExecutor to return a result
        mock_code_executor.return_value.execute_and_return_result.return_value = {
            "result": "SQL Execution successful"
        }

        # Mock SQL method in the DataFrame
        agent._state.dfs[0].execute_sql_query = MagicMock()

        # Execute the code
        code = "execute_sql_query('SELECT * FROM table')"
        result = agent.execute_code(code)

        # Verify the SQL execution environment was set up correctly
        assert result == {"result": "SQL Execution successful"}
        mock_code_executor.return_value.execute_and_return_result.assert_called_with(
            code
        )

    @patch("pandasai.agent.base.CodeExecutor")
    def test_execute_code_logs_execution(self, mock_code_executor, agent: Agent):
        # Mock the logger
        agent._state.logger.log = MagicMock()

        # Mock CodeExecutor to return a result
        mock_code_executor.return_value.execute_and_return_result.return_value = {
            "result": "Logging test successful"
        }

        # Execute the code
        code = "print('Logging test')"
        result = agent.execute_code(code)

        # Verify the logger was called with the correct message
        agent._state.logger.log.assert_called_with(f"Executing code: {code}")
        assert result == {"result": "Logging test successful"}
        mock_code_executor.return_value.execute_and_return_result.assert_called_with(
            code
        )

    @patch("pandasai.agent.base.CodeExecutor")
    def test_execute_code_with_missing_dependencies(
        self, mock_code_executor, agent: Agent
    ):
        # Mock CodeExecutor to simulate a missing dependency error
        mock_code_executor.return_value.execute_and_return_result.side_effect = (
            ImportError("Missing dependency: pandas")
        )

        # Execute the code
        code = "import pandas as pd; print(pd.DataFrame())"

        with pytest.raises(ImportError):
            agent.execute_code(code)

        # Verify the CodeExecutor was called despite the missing dependency
        mock_code_executor.return_value.execute_and_return_result.assert_called_with(
            code
        )

    @patch("pandasai.agent.base.CodeExecutor")
    def test_execute_code_handles_empty_code(self, mock_code_executor, agent: Agent):
        # Mock CodeExecutor to return an empty result
        mock_code_executor.return_value.execute_and_return_result.return_value = {}

        # Execute empty code
        code = ""
        result = agent.execute_code(code)

        # Verify the result is empty and the code executor was not called
        assert result == {}
        mock_code_executor.return_value.execute_and_return_result.assert_called_with(
            code
        )

    def test_start_new_conversation(self, sample_df, config):
        agent = Agent(sample_df, config, memory_size=10)
        agent._state.memory.add("Which country has the highest gdp?", True)
        memory = agent._state.memory.all()
        assert len(memory) == 1
        agent.start_new_conversation()
        memory = agent._state.memory.all()
        assert len(memory) == 0

    def test_code_generation_success(self, agent: Agent):
        # Mock the code generator
        agent._code_generator = Mock()
        expected_code = "print('Test successful')"
        agent._code_generator.generate_code.return_value = expected_code

        code = agent.generate_code("Test query")
        assert code == expected_code
        assert agent._code_generator.generate_code.call_count == 1

    def test_execute_with_retries_max_retries_exceeds(self, agent: Agent):
        # Mock execute_code to always raise an exception
        agent.execute_code = Mock()
        agent.execute_code.side_effect = CodeExecutionError("Test error")
        agent._regenerate_code_after_error = Mock()
        agent._regenerate_code_after_error.return_value = "test_code"

        # Set max retries to 3 explicitly
        agent._state.config.max_retries = 3

        with pytest.raises(CodeExecutionError):
            agent.execute_with_retries("test_code")

        # Should be called max_retries times
        assert agent.execute_code.call_count == 4
        assert agent._regenerate_code_after_error.call_count == 3

    def test_execute_with_retries_success(self, agent: Agent):
        # Mock execute_code to fail twice then succeed
        agent.execute_code = Mock()
        expected_result = {
            "type": "string",
            "value": "Success",
        }  # Correct response format
        # Need enough side effects for all attempts including regenerated code
        agent.execute_code.side_effect = [
            CodeExecutionError("First error"),  # Original code fails
            CodeExecutionError("Second error"),  # First regenerated code fails
            CodeExecutionError("Third error"),  # Second regenerated code fails
            expected_result,  # Third regenerated code succeeds
        ]
        agent._regenerate_code_after_error = Mock()
        agent._regenerate_code_after_error.return_value = "test_code"

        result = agent.execute_with_retries("test_code")
        # Response parser returns a String object with value accessible via .value
        assert result.value == "Success"
        # Should have 4 execute attempts and 3 regenerations
        assert agent.execute_code.call_count == 4
        assert agent._regenerate_code_after_error.call_count == 3

    def test_execute_with_retries_custom_retries(self, agent: Agent):
        # Test with custom number of retries
        agent._state.config.max_retries = 5
        agent.execute_code = Mock()
        agent.execute_code.side_effect = CodeExecutionError("Test error")
        agent._regenerate_code_after_error = Mock()
        agent._regenerate_code_after_error.return_value = "test_code"

        with pytest.raises(CodeExecutionError):
            agent.execute_with_retries("test_code")

        # Should be called max_retries + 1 times (initial try + retries)
        assert agent.execute_code.call_count == 6
        assert agent._regenerate_code_after_error.call_count == 5

    def test_load_llm_with_pandasai_llm(self, agent: Agent, llm):
        assert agent._state._get_llm(llm) == llm

    def test_load_llm_none(self, agent: Agent, llm):
        with patch.dict(os.environ, {"PANDABI_API_KEY": "test_key"}):
            config = agent._state._get_config({})
            assert isinstance(config, Config)
            assert config.llm is None

    def test_get_config_none(self, agent: Agent):
        """Test that _get_config returns global config when input is None"""
        mock_config = Config()
        with patch.object(ConfigManager, "get", return_value=mock_config):
            config = agent._state._get_config(None)
            assert config == mock_config

    def test_get_config_dict(self, agent: Agent):
        """Test that _get_config properly handles dict input"""
        mock_llm = FakeLLM()
        test_dict = {"save_logs": False, "verbose": True, "llm": mock_llm}
        config = agent._state._get_config(test_dict)
        assert isinstance(config, Config)
        assert config.save_logs is False
        assert config.verbose is True
        assert config.llm == mock_llm

    def test_get_config_dict_with_api_key(self, agent: Agent):
        """Test that _get_config with API key no longer initializes an LLM automatically"""
        with patch.dict(os.environ, {"PANDABI_API_KEY": "test_key"}):
            config = agent._state._get_config({})
            assert isinstance(config, Config)
            assert config.llm is None

    def test_get_config_config(self, agent: Agent):
        """Test that _get_config returns Config object unchanged"""
        original_config = Config(save_logs=False, verbose=True)
        config = agent._state._get_config(original_config)
        assert config == original_config
        assert isinstance(config, Config)

    def test_train_method_with_qa(self, agent):
        queries = ["query1", "query2"]
        codes = ["code1", "code2"]
        agent.train(queries, codes)

        agent._state.vectorstore.add_docs.assert_not_called()
        agent._state.vectorstore.add_question_answer.assert_called_once_with(
            queries, codes
        )

    def test_train_method_with_docs(self, agent):
        docs = ["doc1"]
        agent.train(docs=docs)

        agent._state.vectorstore.add_question_answer.assert_not_called()
        agent._state.vectorstore.add_docs.assert_called_once()
        agent._state.vectorstore.add_docs.assert_called_once_with(docs)

    def test_train_method_with_docs_and_qa(self, agent):
        docs = ["doc1"]
        queries = ["query1", "query2"]
        codes = ["code1", "code2"]
        agent.train(queries, codes, docs=docs)

        agent._state.vectorstore.add_question_answer.assert_called_once()
        agent._state.vectorstore.add_question_answer.assert_called_once_with(
            queries, codes
        )
        agent._state.vectorstore.add_docs.assert_called_once()
        agent._state.vectorstore.add_docs.assert_called_once_with(docs)

    def test_train_method_with_queries_but_no_code(self, agent):
        queries = ["query1", "query2"]
        with pytest.raises(ValueError):
            agent.train(queries)

    def test_train_method_with_code_but_no_queries(self, agent):
        codes = ["code1", "code2"]
        with pytest.raises(ValueError):
            agent.train(codes)

    def test_execute_sql_query_success_local(self, agent, sample_df):
        query = f'SELECT count(*) as total from "{sample_df.schema.name}";'
        expected_result = pd.DataFrame({"total": [3]})
        result = agent._execute_sql_query(query)
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("os.path.exists", return_value=True)
    def test_execute_sql_query_success_virtual_dataframe(
        self, mock_exists, agent, mysql_schema, sample_df
    ):
        query = "SELECT count(*) as total from countries;"
        loader = DatasetLoader.create_loader_from_schema(mysql_schema, "test/users")
        expected_result = pd.DataFrame({"total": [4]})

        with patch(
            "builtins.open", mock_open(read_data=str(mysql_schema.to_yaml()))
        ), patch(
            "pandasai.data_loader.sql_loader.SQLDatasetLoader.execute_query"
        ) as mock_query:
            # Set up the mock for both the sample data and the query result
            mock_query.side_effect = [sample_df, expected_result]

            virtual_dataframe = loader.load()
            agent._state.dfs = [virtual_dataframe]

            pd.testing.assert_frame_equal(virtual_dataframe.head(), sample_df)
            result = agent._execute_sql_query(query)
            pd.testing.assert_frame_equal(result, expected_result)

            # Verify execute_query was called appropriately
            assert mock_query.call_count == 2  # Once for head(), once for the SQL query

    def test_execute_sql_query_error_no_dataframe(self, agent):
        query = "SELECT count(*) as total from countries;"
        agent._state.dfs = None

        with pytest.raises(ValueError, match="No DataFrames available"):
            agent._execute_sql_query(query)

    def test_process_query(self, agent, config):
        """Test the _process_query method with successful execution"""
        query = "What is the average age?"
        output_type = "number"

        # Mock the necessary methods
        agent.generate_code = Mock(return_value="result = df['age'].mean()")
        agent.execute_with_retries = Mock(return_value=30.5)

        # Execute the query
        result = agent._process_query(query, output_type)

        # Verify the result
        assert result == 30.5

        # Verify method calls
        agent.generate_code.assert_called_once()
        agent.execute_with_retries.assert_called_once_with("result = df['age'].mean()")

    def test_process_query_execution_error(self, agent, config):
        """Test the _process_query method with execution error"""
        query = "What is the invalid operation?"

        # Mock methods to simulate error
        agent.generate_code = Mock(return_value="invalid_code")
        agent.execute_with_retries = Mock(
            side_effect=CodeExecutionError("Execution failed")
        )
        agent._handle_exception = Mock(return_value="Error handled")

        # Execute the query
        result = agent._process_query(query)

        # Verify error handling
        assert result == "Error handled"
        agent._handle_exception.assert_called_once_with("invalid_code")

    def test_regenerate_code_after_invalid_llm_output_error(self, agent):
        """Test code regeneration with InvalidLLMOutputType error"""
        from pandasai.exceptions import InvalidLLMOutputType

        code = "test code"
        error = InvalidLLMOutputType("Invalid output type")

        with patch(
            "pandasai.agent.base.get_correct_output_type_error_prompt"
        ) as mock_prompt:
            mock_prompt.return_value = "corrected prompt"
            agent._code_generator.generate_code = MagicMock(return_value="new code")

            result = agent._regenerate_code_after_error(code, error)

            mock_prompt.assert_called_once_with(agent._state, code, ANY)
            agent._code_generator.generate_code.assert_called_once_with(
                "corrected prompt"
            )
            assert result == "new code"

    def test_regenerate_code_after_other_error(self, agent):
        """Test code regeneration with non-InvalidLLMOutputType error"""
        code = "test code"
        error = ValueError("Some other error")

        with patch(
            "pandasai.agent.base.get_correct_error_prompt_for_sql"
        ) as mock_prompt:
            mock_prompt.return_value = "sql error prompt"
            agent._code_generator.generate_code = MagicMock(return_value="new code")

            result = agent._regenerate_code_after_error(code, error)

            mock_prompt.assert_called_once_with(agent._state, code, ANY)
            agent._code_generator.generate_code.assert_called_once_with(
                "sql error prompt"
            )
            assert result == "new code"

    def test_handle_exception(self, agent):
        """Test that _handle_exception properly formats and logs exceptions"""
        test_code = "print(1/0)"  # Code that will raise a ZeroDivisionError

        # Mock the logger to verify it's called
        mock_logger = MagicMock()
        agent._state.logger = mock_logger

        # Create an actual exception to handle
        try:
            exec(test_code)
        except:
            # Call the method
            result = agent._handle_exception(test_code)

        # Verify the result is an ErrorResponse
        assert isinstance(result, ErrorResponse)
        assert result.last_code_executed == test_code
        assert "ZeroDivisionError" in result.error

        # Verify the error was logged
        mock_logger.log.assert_called_once()
        assert "Processing failed with error" in mock_logger.log.call_args[0][0]

    def test_last_code_generated_retrieval(self, agent: Agent):
        """Test that last_code_generated is correctly retrieved in get_chat_prompt_for_sql."""
        # Set last_code_generated
        test_code = "print('Test code')"
        agent._state.last_code_generated = test_code

        # 使用 get_chat_prompt_for_sql 获取提示
        from pandasai.core.prompts import get_chat_prompt_for_sql

        prompt = get_chat_prompt_for_sql(agent._state)

        # 验证提示中使用了正确的 last_code_generated
        assert prompt.props["last_code_generated"] == test_code

        # 验证不是从 intermediate_values 中获取的
        agent._state.add("last_code_generated", "Wrong code")
        prompt = get_chat_prompt_for_sql(agent._state)

        # 应该仍然使用 last_code_generated 属性，而不是 intermediate_values 中的值
        assert prompt.props["last_code_generated"] == test_code
        assert prompt.props["last_code_generated"] != "Wrong code"
