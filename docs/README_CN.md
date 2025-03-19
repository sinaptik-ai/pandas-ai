# ![PandaAI](../assets/logo.png)

<p align="center">
[ <a href="../README.md">En</a> |
<b>中</b> |
<a href="README_FR.md">Fr</a> |
<a href="README_JA.md">日</a> ] 
</p>

[![Release](https://img.shields.io/pypi/v/pandasai?label=Release&style=flat-square)](https://pypi.org/project/pandasai/)
[![CI](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/ci-core.yml/badge.svg)](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/ci-core.yml/badge.svg)
[![CD](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/cd.yml/badge.svg)](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/cd.yml/badge.svg)
[![Coverage](https://codecov.io/gh/sinaptik-ai/pandas-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/sinaptik-ai/pandas-ai)
[![Discord](https://dcbadge.vercel.app/api/server/kF7FqH2FwS?style=flat&compact=true)](https://discord.gg/KYKj9F2FRH)
[![Downloads](https://static.pepy.tech/badge/pandasai)](https://pepy.tech/project/pandasai) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZnO-njhL7TBOYPZaqvMvGtsjckZKrv2E?usp=sharing)

PandaAI 是一个 Python 平台，可以轻松地用自然语言向您的数据提问。它帮助非技术用户以更自然的方式与数据交互，并帮助技术用户在处理数据时节省时间和精力。

# 🔧 入门指南

您可以在此处找到 PandaAI 的完整文档 [here](https://pandas-ai.readthedocs.io/en/latest/)。

您可以选择在 Jupyter 笔记本、Streamlit 应用中使用 PandaAI，或者从仓库中使用客户端和服务器架构。

## ☁️ 使用平台

该库可以与我们的强大数据平台一起使用，只需几行代码即可实现端到端的对话式数据分析。

加载您的数据，将其保存为数据框，并将其推送到平台

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

您的团队现在可以通过平台使用自然语言访问和查询这些数据。

![PandaAI](assets/demo.gif)

## 📚 使用库

### Python 要求

Python 版本 `3.8+ <3.12`

### 📦 安装

您可以使用 pip 或 poetry 安装 PandaAI 库。

使用 pip：

```bash
pip install "pandasai>=3.0.0b2"
```

使用 poetry：

```bash
poetry add "pandasai>=3.0.0b2"
```

### 💻 使用

#### 提问

```python
import pandasai as pai

# 示例数据框
df = pai.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "revenue": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})

# 默认情况下，除非您选择其他 LLM，否则它将使用 BambooLLM。
# 您可以在 https://app.pandabi.ai 注册获取免费的 API 密钥（也可以在 .env 文件中配置）
pai.api_key.set("your-pai-api-key")

df.chat('销售额前五的国家是哪些？')
```

```
中国, 美国, 日本, 德国, 澳大利亚
```

---

或者您可以提出更复杂的问题：

```python
df.chat(
    "销售额前三的国家的总销售额是多少？"
)
```

```
销售额前三的国家的总销售额为 16500。
```

#### 可视化图表

您还可以要求 PandaAI 为您生成图表：

```python
df.chat(
    "绘制各国的直方图，显示每个国家的 GDP，并为每个条形使用不同的颜色",
)
```

![Chart](../assets/histogram-chart.png?raw=true)

#### 多个数据框

您还可以将多个数据框传递给 PandaAI，并提出与之相关的问题。

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

# 默认情况下，除非您选择其他 LLM，否则它将使用 BambooLLM。
# 您可以在 https://app.pandabi.ai 注册获取免费的 API 密钥（也可以在 .env 文件中配置）
pai.api_key.set("your-pai-api-key")

pai.chat("谁的收入最高？", employees_df, salaries_df)
```

```
Olivia 的收入最高。
```

#### Docker 沙盒

您可以在 Docker 沙盒中运行 PandaAI，提供一个安全、隔离的环境来安全地执行代码，并降低恶意攻击的风险。

##### Python 要求

```bash
pip install "pandasai-docker"
```

##### 使用

```python
import pandasai as pai
from pandasai_docker import DockerSandbox

# 初始化沙盒
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

# 默认情况下，除非您选择其他 LLM，否则它将使用 BambooLLM。
# 您可以在 https://app.pandabi.ai 注册获取免费的 API 密钥（也可以在 .env 文件中配置）
pai.api_key.set("your-pai-api-key")

pai.chat("谁的收入最高？", employees_df, salaries_df, sandbox=sandbox)

# 完成后别忘了停止沙盒
sandbox.stop()
```

```
Olivia 的收入最高。
```

您可以在 [examples](examples) 目录中找到更多示例。

## 📜 许可证

PandaAI 在 MIT expat 许可证下可用，除了此仓库的 `pandasai/ee` 目录，该目录有其 [许可证](https://github.com/sinaptik-ai/pandas-ai/blob/main/ee/LICENSE)。

如果您对托管 PandaAI 云或自托管企业版感兴趣，请 [联系我们](https://getpanda.ai/pricing)。

## 资源

> **Beta 通知**  
> 版本 v3 目前处于测试阶段。以下文档和示例反映了正在开发中的功能和特性，可能在最终发布前发生变化。

- [文档](https://pandas-ai.readthedocs.io/en/latest/) 提供全面的文档
- [示例](examples) 提供示例笔记本
- [Discord](https://discord.gg/KYKj9F2FRH) 用于与社区和 PandaAI 团队讨论

## 🤝 贡献

欢迎贡献！请查看未解决的问题，并随时提交拉取请求。
有关更多信息，请查看 [贡献指南](CONTRIBUTING.md)。

### 感谢！

[![贡献者](https://contrib.rocks/image?repo=sinaptik-ai/pandas-ai)](https://github.com/sinaptik-ai/pandas-ai/graphs/contributors)