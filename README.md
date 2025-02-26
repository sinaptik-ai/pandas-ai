# ![PandaAI](assets/logo.png)

[![Release](https://img.shields.io/pypi/v/pandasai?label=Release&style=flat-square)](https://pypi.org/project/pandasai/)
[![CI](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/ci-core.yml/badge.svg)](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/ci-core.yml/badge.svg)
[![CD](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/cd.yml/badge.svg)](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/cd.yml/badge.svg)
[![Coverage](https://codecov.io/gh/sinaptik-ai/pandas-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/sinaptik-ai/pandas-ai)
[![Discord](https://dcbadge.vercel.app/api/server/kF7FqH2FwS?style=flat&compact=true)](https://discord.gg/KYKj9F2FRH)
[![Downloads](https://static.pepy.tech/badge/pandasai)](https://pepy.tech/project/pandasai) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZnO-njhL7TBOYPZaqvMvGtsjckZKrv2E?usp=sharing)

PandaAI is a Python platform that makes it easy to ask questions to your data in natural language. It helps non-technical users to interact with their data in a more natural way, and it helps technical users to save time, and effort when working with data.

# 🔧 Getting started

You can find the full documentation for PandaAI [here](https://pandas-ai.readthedocs.io/en/latest/).

You can either decide to use PandaAI in your Jupyter notebooks, Streamlit apps, or use the client and server architecture from the repo.

## ☁️ Using the platform

The library can be used alongside our powerful data platform, making end-to-end conversational data analytics possible with as little as a few lines of code.

Load your data, save them as a dataframe, and push them to the platform

```python
import pandasai as pai

pai.api_key.set("your-pai-api-key")

file = pai.read_csv("./filepath.csv")

dataset = pai.create(path="your-organization/dataset-name",
    df=file,
    name="dataset-name",
    description="dataset-description")

dataset.push()
```

Your team can now access and query this data using natural language through the platform.

![PandaAI](assets/demo.gif)

## 📚 Using the library

### Python Requirements

Python version `3.8+ <3.12`

### 📦 Installation

You can install the PandaAI library using pip or poetry.

With pip:

```bash
pip install "pandasai>=3.0.0b2"
```

With poetry:

```bash
poetry add "pandasai>=3.0.0b2"
```

### 💻 Usage

#### Ask questions

```python
import pandasai as pai

# Sample DataFrame
df = pai.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "revenue": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://app.pandabi.ai (you can also configure it in your .env file)
pai.api_key.set("your-pai-api-key")

df.chat('Which are the top 5 countries by sales?')
```

```
China, United States, Japan, Germany, Australia
```

---

Or you can ask more complex questions:

```python
df.chat(
    "What is the total sales for the top 3 countries by sales?"
)
```

```
The total sales for the top 3 countries by sales is 16500.
```

#### Visualize charts

You can also ask PandaAI to generate charts for you:

```python
df.chat(
    "Plot the histogram of countries showing for each one the gd. Use different colors for each bar",
)
```

![Chart](assets/histogram-chart.png?raw=true)

#### Multiple DataFrames

You can also pass in multiple dataframes to PandaAI and ask questions relating them.

```python
import pandasai as pai

employees_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],
    'Department': ['HR', 'Sales', 'IT', 'Marketing', 'Finance']
}

salaries_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Salary': [5000, 6000, 4500, 7000, 5500]
}

employees_df = pai.DataFrame(employees_data)
salaries_df = pai.DataFrame(salaries_data)

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://app.pandabi.ai (you can also configure it in your .env file)
pai.api_key.set("your-pai-api-key")

pai.chat("Who gets paid the most?", employees_df, salaries_df)
```

```
Olivia gets paid the most.
```

#### Docker Sandbox

You can run PandaAI in a Docker sandbox, providing a secure, isolated environment to execute code safely and mitigate the risk of malicious attacks.

##### Python Requirements

```bash
pip install "pandasai-docker"
```

##### Usage

```python
import pandasai as pai
from pandasai_docker import DockerSandbox

# Initialize the sandbox
sandbox = DockerSandbox()
sandbox.start()

employees_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],
    'Department': ['HR', 'Sales', 'IT', 'Marketing', 'Finance']
}

salaries_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Salary': [5000, 6000, 4500, 7000, 5500]
}

employees_df = pai.DataFrame(employees_data)
salaries_df = pai.DataFrame(salaries_data)

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://app.pandabi.ai (you can also configure it in your .env file)
pai.api_key.set("your-pai-api-key")

pai.chat("Who gets paid the most?", employees_df, salaries_df, sandbox=sandbox)

# Don't forget to stop the sandbox when done
sandbox.stop()
```

```
Olivia gets paid the most.
```

You can find more examples in the [examples](examples) directory.

## 📜 License

PandaAI is available under the MIT expat license, except for the `pandasai/ee` directory of this repository, which has its [license here](https://github.com/sinaptik-ai/pandas-ai/blob/main/ee/LICENSE).

If you are interested in managed PandaAI Cloud or self-hosted Enterprise Offering, [contact us](https://getpanda.ai/pricing).

## Resources

> **Beta Notice**  
> Release v3 is currently in beta. The following documentation and examples reflect the features and functionality in progress and may change before the final release.

- [Docs](https://pandas-ai.readthedocs.io/en/latest/) for comprehensive documentation
- [Examples](examples) for example notebooks
- [Discord](https://discord.gg/KYKj9F2FRH) for discussion with the community and PandaAI team

## 🤝 Contributing

Contributions are welcome! Please check the outstanding issues and feel free to open a pull request.
For more information, please check out the [contributing guidelines](CONTRIBUTING.md).

### Thank you!

[![Contributors](https://contrib.rocks/image?repo=sinaptik-ai/pandas-ai)](https://github.com/sinaptik-ai/pandas-ai/graphs/contributors)
