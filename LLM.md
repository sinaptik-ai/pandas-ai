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
  - `memory_size`: int = 10 (conversation memory)
  - `vectorstore`: Optional[VectorStore] for RAG
  - `description`: Optional[str] agent description
  - `sandbox`: Optional[Sandbox] for secure code execution

- **Methods:**
  - `chat(query: str, output_type: str = "string") -> Any`: Main interaction method
  - `train(qa: List[Dict[str, str]]) -> None`: Train with Q&A pairs
  - `add_skills(skills: List[Skill]) -> None`: Add custom skills
  - `clear_memory() -> None`: Clear conversation history
  - `explain() -> str`: Get explanation of last response
  - `last_code_generated -> str`: Get last generated code
  - `last_code_executed -> str`: Get last executed code
  - `last_result -> Any`: Get last execution result
  - `last_error -> str`: Get last error message

### DataFrame (Enhanced pandas DataFrame)
- **Import:** `from pandasai import DataFrame`
- **Constructor:** `DataFrame(data, config=None, name=None, description=None)`
  - `data`: pandas.DataFrame, dict, or file path
  - `config`: Optional[Config] configuration
  - `name`: Optional[str] dataframe name
  - `description`: Optional[str] dataframe description

- **Methods:**
  - `chat(query: str, output_type: str = "string") -> Any`: Query the dataframe
  - `head(n: int = 5) -> DataFrame`: Get first n rows
  - `tail(n: int = 5) -> DataFrame`: Get last n rows
  - `info() -> str`: Get dataframe info
  - `describe() -> DataFrame`: Get statistical summary
  - `shape -> Tuple[int, int]`: Get dimensions
  - `columns -> List[str]`: Get column names
  - `dtypes -> Dict[str, str]`: Get column data types

### SmartDataframe (Legacy - use DataFrame instead)
- **Import:** `from pandasai import SmartDataframe`
- **Note:** Deprecated in favor of DataFrame class

### Config (Configuration Management)
- **Import:** `from pandasai import Config`
- **Constructor:** `Config(**kwargs)`

#### Key Configuration Options:
- **LLM Settings:**
  - `llm`: LLM instance (required)
  - `max_retries`: int = 3
  - `temperature`: float = 0.0

- **Output Settings:**
  - `save_logs`: bool = True
  - `verbose`: bool = False
  - `response_parser`: ResponseParser instance
  - `output_type`: str = "string"

- **Security Settings:**
  - `enable_cache`: bool = True
  - `use_error_correction_framework`: bool = True
  - `custom_whitelisted_dependencies`: List[str] = []

- **File Paths:**
  - `save_charts_path`: str = "exports/charts"
  - `save_logs_path`: str = "logs"

### VirtualDataFrame (Database Connections)
- **Import:** `from pandasai import VirtualDataFrame`
- **Constructor:** `VirtualDataFrame(connector, name=None, description=None)`
  - `connector`: Database connector instance
  - `name`: Optional[str] table/view name
  - `description`: Optional[str] description

### Connectors (Database Integration)

#### SQL Connectors:
```python
from pandasai.connectors import MySQLConnector, PostgreSQLConnector, SQLiteConnector

# MySQL
mysql_conn = MySQLConnector(
    host="localhost",
    port=3306,
    database="mydb",
    username="user",
    password="pass",
    table="mytable"
)

# PostgreSQL
postgres_conn = PostgreSQLConnector(
    host="localhost",
    port=5432,
    database="mydb",
    username="user",
    password="pass",
    table="mytable"
)

# SQLite
sqlite_conn = SQLiteConnector(
    database="path/to/db.sqlite",
    table="mytable"
)
```

#### NoSQL Connectors:
```python
from pandasai.connectors import MongoDBConnector

mongo_conn = MongoDBConnector(
    host="localhost",
    port=27017,
    database="mydb",
    collection="mycollection"
)
```

### LLM Integration

#### OpenAI:
```python
from pandasai.llm import OpenAI
llm = OpenAI(api_token="your-api-key", model="gpt-4")
```

#### Anthropic:
```python
from pandasai.llm import Claude
llm = Claude(api_token="your-api-key")
```

#### Google:
```python
from pandasai.llm import GooglePalm, GoogleVertexAI
llm = GooglePalm(api_key="your-api-key")
llm = GoogleVertexAI(project_id="your-project")
```

#### Azure OpenAI:
```python
from pandasai.llm import AzureOpenAI
llm = AzureOpenAI(
    api_token="your-api-key",
    azure_endpoint="your-endpoint",
    api_version="2023-05-15"
)
```

#### Local/Open Source:
```python
from pandasai.llm import HuggingFaceLLM, Ollama
llm = HuggingFaceLLM(model_name="microsoft/DialoGPT-medium")
llm = Ollama(model="llama2")
```

### Skills (Custom Functions)
```python
from pandasai.skills import skill

@skill
def calculate_profit_margin(df):
    """Calculate profit margin percentage"""
    return (df['profit'] / df['revenue']) * 100

agent.add_skills([calculate_profit_margin])
```

### Memory Management
```python
from pandasai.memory import Memory

# Custom memory size
agent = Agent(df, memory_size=20)

# Clear memory
agent.clear_memory()
```

### Sandbox (Secure Execution)
```python
from pandasai.sandbox import CodeExecutionSandbox

sandbox = CodeExecutionSandbox()
agent = Agent(df, sandbox=sandbox)
```

## Usage Examples

### Basic Usage:
```python
import pandasai as pai
from pandasai.llm import OpenAI

# Configure LLM
llm = OpenAI(api_token="your-api-key")
pai.config.set({"llm": llm})

# Create DataFrame
df = pai.DataFrame({
    "country": ["USA", "UK", "France"],
    "revenue": [5000, 3200, 2900]
})

# Query the data
result = df.chat("What is the total revenue?")
print(result)  # "The total revenue is 11100"
```

### Multiple DataFrames:
```python
employees = pai.DataFrame({
    'id': [1, 2, 3],
    'name': ['John', 'Jane', 'Bob'],
    'department': ['IT', 'HR', 'Sales']
})

salaries = pai.DataFrame({
    'id': [1, 2, 3],
    'salary': [70000, 65000, 60000]
})

agent = pai.Agent([employees, salaries])
result = agent.chat("Who has the highest salary?")
```

### Database Integration:
```python
from pandasai.connectors import PostgreSQLConnector

connector = PostgreSQLConnector(
    host="localhost",
    database="sales_db",
    username="user",
    password="pass",
    table="transactions"
)

df = pai.VirtualDataFrame(connector)
result = df.chat("Show me sales trends by month")
```

### Custom Configuration:
```python
config = pai.Config(
    llm=OpenAI(api_token="your-key"),
    save_logs=True,
    verbose=True,
    max_retries=5,
    temperature=0.1
)

df = pai.DataFrame(data, config=config)
```

### Chart Generation:
```python
result = df.chat(
    "Create a bar chart showing revenue by country",
    output_type="plot"
)
# Returns matplotlib figure object
```

### Training the Agent:
```python
training_data = [
    {"question": "What is our best product?", "answer": "Product A with 45% market share"},
    {"question": "Who is our top customer?", "answer": "Customer X with $2M revenue"}
]

agent.train(training_data)
```

## Error Handling

### Common Exceptions:
- `LLMNotFoundError`: No LLM configured
- `DataFrameNotFoundError`: Invalid dataframe reference
- `InvalidConfigError`: Invalid configuration
- `ExecutionError`: Code execution failed
- `ParsingError`: Response parsing failed

### Error Handling Example:
```python
try:
    result = df.chat("Complex query here")
except Exception as e:
    print(f"Error: {e}")
    print(f"Last error: {agent.last_error}")
    print(f"Last code: {agent.last_code_generated}")
```

## Best Practices

### 1. LLM Configuration:
```python
# Use appropriate temperature
config = pai.Config(
    llm=OpenAI(api_token="key", temperature=0.0),  # Deterministic
    max_retries=3
)
```

### 2. Data Preparation:
```python
# Clean column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Add descriptions
df = pai.DataFrame(data, description="Sales data from 2023")
```

### 3. Query Optimization:
```python
# Be specific in queries
result = df.chat("Show me the top 5 products by revenue in Q4 2023")

# Use output_type for charts
chart = df.chat("Create a line chart of monthly sales", output_type="plot")
```

### 4. Memory Management:
```python
# Clear memory for new contexts
agent.clear_memory()

# Use appropriate memory size
agent = pai.Agent(df, memory_size=15)
```

### 5. Security:
```python
# Use sandbox for untrusted environments
from pandasai.sandbox import CodeExecutionSandbox
sandbox = CodeExecutionSandbox()
agent = pai.Agent(df, sandbox=sandbox)
```

## Advanced Features

### Custom Response Parser:
```python
from pandasai.response_parser import ResponseParser

class CustomParser(ResponseParser):
    def parse(self, response):
        # Custom parsing logic
        return processed_response

config = pai.Config(response_parser=CustomParser())
```

### Vector Store Integration:
```python
from pandasai.vectorstore import ChromaVectorStore

vectorstore = ChromaVectorStore()
agent = pai.Agent(df, vectorstore=vectorstore)
```

### Middleware:
```python
from pandasai.middleware import Middleware

class LoggingMiddleware(Middleware):
    def before_chat(self, query):
        print(f"Processing: {query}")
    
    def after_chat(self, response):
        print(f"Response: {response}")

agent.add_middleware(LoggingMiddleware())
```

## Troubleshooting

### Common Issues:

1. **LLM Not Found:**
   ```python
   # Ensure LLM is configured
   pai.config.set({"llm": OpenAI(api_token="your-key")})
   ```

2. **Memory Issues:**
   ```python
   # Clear memory if responses become inconsistent
   agent.clear_memory()
   ```

3. **Code Execution Errors:**
   ```python
   # Check last generated code
   print(agent.last_code_generated)
   print(agent.last_error)
   ```

4. **Database Connection:**
   ```python
   # Test connection separately
   connector.test_connection()
   ```

### Debug Mode:
```python
config = pai.Config(
    verbose=True,
    save_logs=True,
    save_logs_path="debug_logs"
)
```

## Migration Guide

### From v2 to v3:
```python
# Old (v2)
from pandasai import SmartDataframe
sdf = SmartDataframe(df, config=config)

# New (v3)
from pandasai import DataFrame
df = DataFrame(df, config=config)
```

### Configuration Changes:
```python
# Old
config = {"llm": llm, "verbose": True}

# New
config = Config(llm=llm, verbose=True)
```

This documentation covers the complete PandasAI v3 API. For the latest updates, visit the [official documentation](https://docs.getpanda.ai/).