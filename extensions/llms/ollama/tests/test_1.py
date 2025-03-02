import pandasai as pai
from extensions.llms.ollama.pandasai_ollama.ollama import OllamaLLM

# Create a sample DataFrame
df = pai.DataFrame(
    {
        "country": ["United States", "United Kingdom", "France", "Germany", "Italy"],
        "revenue": [5000, 3200, 2900, 4100, 2300],
    }
)

# For Ollama, we use a dummy API key ("ollama") since it isn’t used.
pai.api_key.set("ollama")

# Set the global configuration to use the Ollama LLM
pai.config.set(
    {
        "llm": OllamaLLM(
            api_key="ollama",
            ollama_base_url="http://localhost:11434",
            model="llama3.2:latest",
            temperature=0.7,
            max_tokens=150,
        )
    }
)

# Ask the question
response = df.chat("Which are the top 5 countries by sales?")
print("Response from Ollama:", response)
