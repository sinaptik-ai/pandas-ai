# PandaAI Ollama LLM Extension

This extension integrates Ollama language models with PandaAI. It allows you to use Ollama's LLMs as the backend for generating Python code in response to natural language queries on your dataframes.

## Features

- **Ollama Integration:** Leverage Ollama's powerful language models (e.g., `llama2`, `llama3.2`) within PandaAI.
- **Customizable Base URL:** Easily change the Ollama base URL (default is `http://localhost:11434/v1`) to point to your own Ollama server.
- **Flexible Model Parameters:** Configure model parameters such as temperature, max tokens, top_p, frequency penalty, etc.
- **Chat & Non-Chat Modes:** Supports both conversational (chat) mode and standard completion mode.

## Installation

1. **Clone the Repository** (if you haven't already):
   ```bash
   git clone https://github.com/sinaptik-ai/pandas-ai.git
   cd pandas-ai
   ```

2. **Navigate to the Ollama Extension Directory:**
   ```bash
   cd extensions/llms/ollama
   ```

3. **Install the Extension Dependencies:**
   ```bash
    poetry install

    poetry add pandasai-ollama
   ```

   > **Note:** If you encounter packaging issues, ensure that this directory contains a valid `README.md` file. This README file is required for Poetry to install the project.

## Configuration

### Environment Variables

You can configure the extension by setting the following environment variables:

- **`OLLAMA_API_KEY`**  
  A dummy API key is required by the extension (the key itself is not used).  
  *Default:* `ollama`

- **`OLLAMA_BASE_URL`**  
  Supports a customizable base URL (normalized to end with `/v1`) and flexible model parameters (e.g., temperature, max_tokens)..  
  *Default format to set:* `http://localhost:11434`

### Code-Based Configuration

You can also override configuration options directly in your code when setting up PandaAI:

```python
import pandasai as pai
from extensions.llms.ollama.pandasai_ollama.ollama import OllamaLLM

# For Ollama, we use a dummy API key ("ollama") since it isn’t used.
pai.api_key.set("ollama")

# Set the global configuration to use the Ollama LLM
pai.config.set(
    {
        "llm": OllamaLLM(
            api_key="ollama",
            ollama_base_url="http://localhost:11434", # Custom URL if needed
            model="llama3.2:latest", # Specify the model (can be overridden)
            temperature=0.7,
            max_tokens=150,
        )
    }
)
```

## Usage

Once you have configured the extension, you can use PandaAI’s DataFrame interface to interact with your data. For example:

```python
import pandasai as pai

# Create a sample DataFrame
df = pai.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy"],
    "revenue": [5000, 3200, 2900, 4100, 2300]
})

# Ask a natural language question that expects a Python code answer
response = df.chat("Which are the top 5 countries by sales?")
print("Response from Ollama:", response)
```

The extension sends your prompt (and any conversation context) to the Ollama LLM backend. The LLM is expected to return a valid Python code snippet that, when executed, produces the desired result.


