# PandasAI Library API Documentation

**Version:** 3.0.0-beta.19  
**Description:** Chat with your database (SQL, CSV, pandas, mongodb, noSQL, etc). PandasAI makes data analysis conversational using LLMs (GPT 3.5 / 4, Anthropic, VertexAI) and RAG.  
**Repository:** https://github.com/sinaptik-ai/pandas-ai  
**Documentation:** https://docs.getpanda.ai/  

## Installation

```bash
pip install pandasai
```

## Core Classes and Functions

### Agent (Primary Interface)
- **Import:** `from pandasai import Agent`
- **Constructor:** `Agent(dfs, config=None, memory_size=10, vectorstore=None, description=None, sandbox=None)`
  - `dfs`: Union[DataFrame, VirtualDataFrame] or List of them
  - `config`: Optional[Union[Config, dict]] (deprecated)
  - `memory_size`: Optional[int] = 10
  - `vectorstore`: Optional[VectorStore]
  - `description`: str
  - `sandbox`: Sandbox
- **Methods:**
  - `chat(query: str, output_type: Optional[str] = None) -> Any` # Start new chat interaction
  - `follow_up(query: str, output_type: Optional[str] = None) -> Any` # Continue existing chat
  - `generate_code(query: Union[UserQuery, str]) -> str` # Generate code for query
  - `execute_code(code: str) -> dict` # Execute generated code
  - `train(queries: Optional[List[str]] = None, codes: Optional[List[str]] = None, docs: Optional[List[str]] = None) -> None` # Train the agent
  - `clear_memory() -> None` # Clear conversation memory
  - `add_message(message, is_user=False) -> None` # Add message to memory
  - `start_new_conversation() -> None` # Start new conversation
- **Properties:**
  - `last_generated_code` # Last generated code
  - `last_code_executed` # Last executed code  
  - `last_prompt_used` # Last prompt used

### SmartDataframe (DEPRECATED - Use Agent instead)
- **Import:** `from pandasai import SmartDataframe`
- **Constructor:** `SmartDataframe(df, name=None, description=None, custom_head=None, config=None)`
- **Methods:**
  - `chat(query: str, output_type: Optional[str] = None) -> Any` # Run query on dataframe

### SmartDatalake (DEPRECATED - Use Agent instead)
- **Import:** `from pandasai import SmartDatalake`
- **Constructor:** `SmartDatalake(dfs, config=None)`
- **Methods:**
  - `chat(query: str, output_type: Optional[str] = None) -> Any` # Run query on datalake
  - `clear_memory() -> None` # Clear memory

### Global Functions
- `chat(query: str, *dataframes: DataFrame, sandbox: Optional[Sandbox] = None) -> Any` # Global chat function
- `follow_up(query: str) -> Any` # Global follow-up function
- `create(path: str, df: Optional[DataFrame] = None, description: Optional[str] = None, columns: Optional[List[dict]] = None, source: Optional[dict] = None, relations: Optional[List[dict]] = None, view: bool = False, group_by: Optional[List[str]] = None, transformations: Optional[List[dict]] = None) -> Union[DataFrame, VirtualDataFrame]` # Create dataset

### Global Variables
- `config` # ConfigManager instance
- `api_key` # APIKeyManager instance


## Configuration and Data Loading

### Config
- **Import:** `from pandasai.config import Config`
- **Constructor:** `Config(save_logs=True, verbose=False, max_retries=3, llm=None, file_manager=DefaultFileManager())`
- **Methods:** 
  - `from_dict(config: Dict[str, Any]) -> Config` # Create Config from dictionary

### ConfigManager
- **Import:** `from pandasai.config import ConfigManager`
- **Constructor:** Singleton class (no direct instantiation)
- **Methods:**
  - `set(config_dict: Dict[str, Any]) -> None` # Set the global configuration
  - `get() -> Config` # Get the global configuration
  - `update(config_dict: Dict[str, Any]) -> None` # Update existing configuration with new values

### APIKeyManager
- **Import:** `from pandasai.config import APIKeyManager`
- **Constructor:** Singleton class (no direct instantiation)
- **Methods:**
  - `set(api_key: str) -> None` # Set API key
  - `get() -> Optional[str]` # Get API key

### DatasetLoader
- **Import:** `from pandasai.data_loader.loader import DatasetLoader`
- **Constructor:** `DatasetLoader(schema: SemanticLayerSchema, dataset_path: str)`
- **Methods:**
  - `create_loader_from_schema(schema: SemanticLayerSchema, dataset_path: str) -> DatasetLoader` # Factory method to create appropriate loader
  - `create_loader_from_path(dataset_path: str) -> DatasetLoader` # Factory method from path
  - `execute_query(query: str, params: Optional[list] = None)` # Execute query (abstract)
  - `load() -> DataFrame` # Load data into DataFrame

### LocalDatasetLoader
- **Import:** `from pandasai.data_loader.local_loader import LocalDatasetLoader`
- **Constructor:** `LocalDatasetLoader(schema: SemanticLayerSchema, dataset_path: str)`
- **Methods:**
  - `register_table() -> None` # Register table in DuckDB
  - `load() -> DataFrame` # Load local data into DataFrame
  - `execute_query(query: str, params: Optional[list] = None) -> pd.DataFrame` # Execute query on local data

### SQLDatasetLoader
- **Import:** `from pandasai.data_loader.sql_loader import SQLDatasetLoader`
- **Constructor:** `SQLDatasetLoader(schema: SemanticLayerSchema, dataset_path: str)`
- **Methods:**
  - `load() -> VirtualDataFrame` # Load SQL data into VirtualDataFrame
  - `execute_query(query: str, params: Optional[list] = None) -> pd.DataFrame` # Execute SQL query
  - `load_head() -> pd.DataFrame` # Load head of data
  - `get_row_count() -> int` # Get total row count

### DataFrame Classes
- **Import:** `from pandasai.dataframe import DataFrame`
- **Constructor:** `DataFrame(data=None, index=None, columns=None, dtype=None, copy=None, schema=None, path=None)`
- **Methods:**
  - `chat(prompt: str, sandbox: Optional[Sandbox] = None) -> BaseResponse` # Interact with DataFrame using natural language
  - `follow_up(query: str, output_type: Optional[str] = None)` # Follow up on previous query
  - `rows_count() -> int` # Get number of rows
  - `columns_count() -> int` # Get number of columns
  - `serialize_dataframe() -> str` # Serialize DataFrame to string
  - `get_head()` # Get head of DataFrame
  - `get_default_schema(dataframe: DataFrame) -> SemanticLayerSchema` # Get default schema

### VirtualDataFrame
- **Import:** `from pandasai.dataframe.virtual_dataframe import VirtualDataFrame`
- **Constructor:** `VirtualDataFrame(*args, data_loader=None, **kwargs)`
- **Methods:**
  - `head()` # Get head of virtual data
  - `execute_sql_query(query: str) -> pd.DataFrame` # Execute SQL query on virtual data
- **Properties:**
  - `rows_count -> int` # Get total row count
  - `query_builder` # Get query builder instance


## LLM and Query Processing

### LLM Base
- **Import:** `from pandasai.llm import LLM`
- **Constructor:** `LLM(api_key: Optional[str] = None, **kwargs: Any)`
- **Methods:**
  - `is_pandasai_llm() -> bool` # Return True if the LLM is from pandasAI
  - `type() -> str` # Return type of LLM
  - `call(instruction: BasePrompt, context: AgentState = None) -> str` # Execute the LLM with given prompt
  - `generate_code(instruction: BasePrompt, context: AgentState) -> str` # Generate the code based on the instruction and the given prompt
  - `get_system_prompt(memory: Memory) -> Any` # Generate system prompt with agent info and previous conversations
  - `get_messages(memory: Memory) -> Any` # Return formatted messages
  - `prepend_system_prompt(prompt: BasePrompt, memory: Memory)` # Append system prompt to the chat prompt

### Fake LLM (for testing)
- **Import:** `from pandasai.llm.fake import FakeLLM`
- **Constructor:** `FakeLLM(output: Optional[str] = None, type: str = "fake")`
- **Methods:**
  - `call(instruction: BasePrompt, context: AgentState = None) -> str` # Execute the LLM with given prompt

### Query Builders
- **Import:** `from pandasai.query_builders import SqlQueryBuilder, ViewQueryBuilder, LocalQueryBuilder`

#### Base Query Builder
- **Constructor:** `BaseQueryBuilder(schema: SemanticLayerSchema)`
- **Methods:**
  - `validate_query_builder()` # Validate the generated SQL query
  - `build_query() -> str` # Build the main SQL query
  - `get_head_query(n=5)` # Get query to fetch first n rows
  - `get_row_count()` # Get query to count total rows
  - `check_compatible_sources(sources: List[Source]) -> bool` # Check if sources are compatible

#### SQL Query Builder
- **Constructor:** `SqlQueryBuilder(schema: SemanticLayerSchema)`
- Inherits all methods from BaseQueryBuilder

#### View Query Builder
- **Constructor:** `ViewQueryBuilder(schema: SemanticLayerSchema, schema_dependencies_dict: Dict[str, DatasetLoader])`
- **Methods:**
  - `normalize_view_column_name(name: str) -> str` # Normalize view column name
  - `normalize_view_column_alias(name: str) -> str` # Normalize view column alias
  - Inherits all methods from BaseQueryBuilder

#### Local Query Builder
- **Constructor:** `LocalQueryBuilder(schema: SemanticLayerSchema, dataset_path: str)`
- Inherits all methods from BaseQueryBuilder

### Response Types
- **Import:** `from pandasai.core.response import BaseResponse, DataFrameResponse, ChartResponse, StringResponse, NumberResponse, ErrorResponse`

#### Base Response
- **Constructor:** `BaseResponse(value: Any = None, type: str = None, last_code_executed: str = None, error: str = None)`
- **Methods:**
  - `__str__() -> str` # Return the string representation of the response
  - `__repr__() -> str` # Return a detailed string representation for debugging
  - `to_dict() -> dict` # Return a dictionary representation
  - `to_json() -> str` # Return a JSON representation

#### DataFrame Response
- **Constructor:** `DataFrameResponse(value: Any = None, last_code_executed: str = None)`
- **Methods:**
  - `format_value(value)` # Format value as DataFrame
  - Inherits all methods from BaseResponse

#### Chart Response
- **Constructor:** `ChartResponse(value: Any, last_code_executed: str)`
- **Methods:**
  - `_get_image() -> Image.Image` # Get PIL Image from chart data
  - Inherits all methods from BaseResponse

#### String Response
- **Constructor:** `StringResponse(value: Any = None, last_code_executed: str = None)`
- Inherits all methods from BaseResponse

#### Number Response
- **Constructor:** `NumberResponse(value: Any = None, last_code_executed: str = None)`
- Inherits all methods from BaseResponse

#### Error Response
- **Constructor:** `ErrorResponse(value: Any = None, last_code_executed: str = None)`
- Inherits all methods from BaseResponse

### Vector Stores
- **Import:** `from pandasai.vectorstores import VectorStore`

#### Vector Store (Abstract Base)
- **Methods:**
  - `add_question_answer(queries: Iterable[str], codes: Iterable[str], ids: Optional[Iterable[str]] = None, metadatas: Optional[List[dict]] = None) -> List[str]` # Add question and answer(code) to the training set
  - `add_docs(docs: Iterable[str], ids: Optional[Iterable[str]] = None, metadatas: Optional[List[dict]] = None) -> List[str]` # Add docs to the training set
  - `update_question_answer(ids: Iterable[str], queries: Iterable[str], codes: Iterable[str])` # Update question and answer pairs
  - `update_docs(ids: Iterable[str], docs: Iterable[str])` # Update documents
  - `delete_question_and_answers(ids: Optional[List[str]] = None) -> Optional[bool]` # Delete question and answers by ID
  - `delete_docs(ids: Optional[List[str]] = None) -> Optional[bool]` # Delete documents by ID
  - `delete_collection(collection_name: str) -> Optional[bool]` # Delete the collection
  - `get_relevant_question_answers(question: str, k: int = 1) -> List[dict]` # Returns relevant question answers based on search
  - `get_relevant_docs(question: str, k: int = 1) -> List[dict]` # Returns relevant documents based search
  - `get_relevant_question_answers_by_id(ids: Iterable[str]) -> List[dict]` # Returns relevant question answers based on ids
  - `get_relevant_docs_by_id(ids: Iterable[str]) -> List[dict]` # Returns relevant documents based on ids
  - `get_relevant_qa_documents(question: str, k: int = 1) -> List[str]` # Returns relevant question answers documents only
  - `get_relevant_docs_documents(question: str, k: int = 1) -> List[str]` # Returns relevant question answers documents only


## Utilities and Exceptions

### Exceptions
- **Import:** `from pandasai.exceptions import ExceptionName`
- `InvalidRequestError`: Raised when the request is not successful
- `APIKeyNotFoundError`: Raised when the API key is not defined/declared
- `LLMNotFoundError`: Raised when the LLM is not provided
- `NoCodeFoundError`: Raised when no code is found in the response
- `NoResultFoundError`: Raised when no result is found
- `MethodNotImplementedError`: Raised when a method is not implemented
- `UnsupportedModelError`: Raised when the model is not supported
- `MissingModelError`: Raised when the model is missing
- `BadImportError`: Raised when there's a bad import
- `TemplateFileNotFoundError`: Raised when template file is not found
- `UnSupportedLogicUnit`: Raised for unsupported logic units
- `InvalidWorkspacePathError`: Raised when workspace path is invalid
- `InvalidConfigError`: Raised when configuration is invalid
- `MaliciousQueryError`: Raised when query is malicious
- `InvalidLLMOutputType`: Raised when LLM output type is invalid
- `InvalidOutputValueMismatch`: Raised when output value doesn't match
- `ExecuteSQLQueryNotUsed`: Raised when SQL query execution is not used
- `PipelineConcatenationError`: Raised during pipeline concatenation errors
- `MissingVectorStoreError`: Raised when vector store is missing
- `PandasAIApiKeyError`: Raised for PandasAI API key errors
- `PandasAIApiCallError`: Raised for PandasAI API call errors
- `PandasConnectorTableNotFound`: Raised when connector table is not found
- `InvalidTrainJson`: Raised when training JSON is invalid
- `InvalidSchemaJson`: Raised when schema JSON is invalid
- `LazyLoadError`: Raised when trying to access data that hasn't been loaded in lazy load mode
- `InvalidDataSourceType`: Raised error with invalid data source provided
- `MaliciousCodeGenerated`: Raised when malicious code is generated
- `DatasetNotFound`: Raised when dataset is not found
- `CodeExecutionError`: Raised during code execution errors
- `VirtualizationError`: Raised when there is an error with DataFrame virtualization
- `UnsupportedTransformation`: Raised when a transformation is not supported

### Helper Functions

#### DataframeSerializer
- **Import:** `from pandasai.helpers.dataframe_serializer import DataframeSerializer`
- `serialize(df, dialect="postgres") -> str` # Convert df to CSV-like format wrapped in table tags
- `_truncate_dataframe(df) -> DataFrame` # Truncate dataframe for serialization

#### Environment
- **Import:** `from pandasai.helpers.env import load_dotenv`
- `load_dotenv() -> None` # Load .env file from project root folder

#### File Management
- **Import:** `from pandasai.helpers.filemanager import FileManager, DefaultFileManager`
- `FileManager.load(file_path) -> str` # Abstract method to read file content
- `FileManager.load_binary(file_path) -> bytes` # Abstract method to read file as bytes
- `FileManager.write(file_path, content) -> None` # Abstract method to write file content
- `FileManager.exists(file_path) -> bool` # Abstract method to check if file exists
- `DefaultFileManager.load(file_path) -> str` # Read file content from local filesystem
- `DefaultFileManager.write(file_path, content) -> None` # Write content to local file

#### Logging
- **Import:** `from pandasai.helpers.logger import Logger`
- `Logger.__init__(save_logs=True, verbose=False)` # Initialize logger
- `Logger.log(message, level=logging.INFO)` # Log a message with specified level
- `Logger.logs() -> List[str]` # Get all logged messages

#### Memory
- **Import:** `from pandasai.helpers.memory import Memory`
- `Memory.add(message, is_user)` # Add message to conversation memory
- `Memory.count() -> int` # Get count of messages in memory
- `Memory.all() -> list` # Get all messages from memory
- `Memory.get_conversation(limit=None) -> str` # Get formatted conversation string
- `Memory.clear()` # Clear all messages from memory

#### Path Utilities
- **Import:** `from pandasai.helpers.path import *`
- `find_project_root(filename=None)` # Find the root directory of the project
- `find_closest(filename)` # Find closest file in directory hierarchy
- `validate_name_format(value)` # Validate name format
- `get_table_name_from_path(filepath) -> str` # Extract table name from file path

#### Session Management
- **Import:** `from pandasai.helpers.session import Session, get_PandasAI_session`
- `Session.get(path=None, **kwargs)` # Make GET request
- `Session.post(path=None, **kwargs)` # Make POST request
- `get_PandasAI_session() -> Session` # Get PandasAI session instance

#### SQL Sanitization
- **Import:** `from pandasai.helpers.sql_sanitizer import *`
- `sanitize_view_column_name(relation_name) -> str` # Sanitize column names for SQL views
- `sanitize_sql_table_name(table_name) -> str` # Sanitize SQL table names
- `is_sql_query_safe(query, dialect="postgres") -> bool` # Check if SQL query is safe
- `is_sql_query(query) -> bool` # Check if string is a SQL query

#### JSON Encoding
- **Import:** `from pandasai.helpers.json_encoder import convert_numpy_types`
- `convert_numpy_types(obj)` # Convert numpy types to JSON serializable types

#### Folder Management
- **Import:** `from pandasai.helpers.folder import Folder`
- `Folder.create(path, config=FolderConfig())` # Create folder with configuration

#### Telemetry
- **Import:** `from pandasai.helpers.telemetry import scarf_analytics`
- `scarf_analytics()` # Send analytics data

### Sandbox Functions
- **Import:** `from pandasai.sandbox.sandbox import Sandbox`
- `Sandbox.start()` # Start the sandbox environment
- `Sandbox.stop()` # Stop the sandbox environment
- `Sandbox.execute(code, environment) -> dict` # Execute code in sandbox
- `Sandbox.transfer_file(csv_data, filename="file.csv")` # Transfer file to sandbox

### Constants
- **Import:** `from pandasai.constants import CONSTANT_NAME`
- `DEFAULT_API_URL`: "https://api.pandabi.ai"
- `DEFAULT_CHART_DIRECTORY`: "exports/charts"
- `DEFAULT_FILE_PERMISSIONS`: 0o755
- `SUPPORTED_SOURCE_CONNECTORS`: Dictionary mapping connector types to packages
- `LOCAL_SOURCE_TYPES`: ["csv", "parquet"]
- `REMOTE_SOURCE_TYPES`: ["mysql", "postgres", "cockroachdb", "data", "yahoo_finance", "bigquery", "snowflake", "databricks", "oracle"]
- `SQL_SOURCE_TYPES`: ["mysql", "postgres", "cockroachdb", "oracle"]
- `VALID_COLUMN_TYPES`: ["string", "integer", "float", "datetime", "boolean"]
- `VALID_TRANSFORMATION_TYPES`: List of supported transformation operations


## Semantic Layer Schema

### Schema Components
- **Import:** `from pandasai.data_loader.semantic_layer_schema import Column, Relation, SemanticLayerSchema, Source, Transformation`

#### Column
- **Constructor:** `Column(name: str, type: str, description: Optional[str] = None, alias: Optional[str] = None)`
- **Properties:**
  - `name`: Column name
  - `type`: Column data type (string, integer, float, datetime, boolean)
  - `description`: Optional description
  - `alias`: Optional alias

#### Source
- **Constructor:** `Source(name: str, type: str, path: str, columns: List[Column])`
- **Properties:**
  - `name`: Source name
  - `type`: Source type (csv, parquet, mysql, postgres, etc.)
  - `path`: Path to data source
  - `columns`: List of Column objects

#### Relation
- **Constructor:** `Relation(source: str, target: str, type: str, on: List[str])`
- **Properties:**
  - `source`: Source table name
  - `target`: Target table name
  - `type`: Relation type (one-to-one, one-to-many, many-to-many)
  - `on`: List of column names for join

#### Transformation
- **Constructor:** `Transformation(name: str, type: str, params: dict)`
- **Properties:**
  - `name`: Transformation name
  - `type`: Transformation type
  - `params`: Transformation parameters

#### SemanticLayerSchema
- **Constructor:** `SemanticLayerSchema(sources: List[Source], relations: Optional[List[Relation]] = None, transformations: Optional[List[Transformation]] = None)`
- **Properties:**
  - `sources`: List of Source objects
  - `relations`: Optional list of Relation objects
  - `transformations`: Optional list of Transformation objects

## Usage Examples

### Basic Usage with Agent
```python
import pandas as pd
from pandasai import Agent

# Create dataframe
df = pd.read_csv("data.csv")

# Initialize agent
agent = Agent(df)

# Chat with your data
result = agent.chat("What is the average sales by region?")
print(result)

# Follow up questions
follow_up = agent.follow_up("Show me a chart of this data")
```

### Configuration
```python
from pandasai import Agent
from pandasai.config import Config

# Configure with custom settings
config = Config(
    save_logs=True,
    verbose=True,
    max_retries=5
)

agent = Agent(df, config=config)
```

### Using with Multiple DataFrames
```python
from pandasai import Agent

# Multiple dataframes
df1 = pd.read_csv("sales.csv")
df2 = pd.read_csv("customers.csv")

agent = Agent([df1, df2])
result = agent.chat("Join sales and customer data to show top customers")
```

### Training the Agent
```python
# Train with custom examples
queries = ["Show me sales trends", "What are the top products?"]
codes = ["df.groupby('date').sum()", "df.groupby('product').sum().sort_values(ascending=False)"]

agent.train(queries=queries, codes=codes)
```

### Using Vector Store for RAG
```python
from pandasai.vectorstores import VectorStore

# Initialize with vector store for enhanced context
vectorstore = VectorStore()  # Configure your preferred vector store
agent = Agent(df, vectorstore=vectorstore)
```

### Error Handling
```python
from pandasai.exceptions import PandasAIApiKeyError, NoCodeFoundError

try:
    result = agent.chat("Analyze the data")
except PandasAIApiKeyError:
    print("API key not found or invalid")
except NoCodeFoundError:
    print("No code generated for the query")
```

## Migration from v2 to v3

### Old (v2) vs New (v3)
```python
# OLD - SmartDataframe (deprecated)
from pandasai import SmartDataframe
sdf = SmartDataframe(df)
result = sdf.chat("What is the sum of sales?")

# NEW - Agent (recommended)
from pandasai import Agent
agent = Agent(df)
result = agent.chat("What is the sum of sales?")
```

### Key Changes
- `SmartDataframe` and `SmartDatalake` are deprecated, use `Agent` instead
- Improved memory management with configurable `memory_size`
- Enhanced training capabilities with `train()` method
- Better error handling and response types
- Vector store integration for RAG capabilities
- Sandbox execution for secure code running

