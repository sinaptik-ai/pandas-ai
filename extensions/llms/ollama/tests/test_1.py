import pandasai as pai
from extensions.llms.ollama.pandasai_ollama.ollama import OllamaLLM

# Sample DataFrame
df = pai.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy"],
    "revenue": [5000, 3200, 2900, 4100, 2300]
})

# Set a dummy API key for Ollama
pai.api_key.set("ollama")

# Create a custom prompt that forces the LLM to use execute_sql_query
custom_prompt = (
 "Generate Python code that retrieves the top 5 countries by sales from the given DataFrame. Your code must call the function `execute_sql_query` with a valid SQL SELECT query to obtain the result. IMPORTANT: Enclose the entire code snippet within triple backticks (```), and return nothing else (no explanations, comments, or extra text)."
)

# Override the global configuration with your custom LLM and custom prompt.
pai.config.set({
    "llm": OllamaLLM(
        api_key="ollama",
        ollama_base_url="http://localhost:11434",  # or your custom URL
        model="llama3.2:latest",  # using the new model name if needed
        temperature=0.7,
        max_tokens=150,
    ),
    "custom_prompts": {
        "chat": custom_prompt
    }
})

# Now call df.chat; the agent will use your custom prompt.
response = df.chat("Which are the top 5 countries by sales?")
print("Response from Ollama:", response)
