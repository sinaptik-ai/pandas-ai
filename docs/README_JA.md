# ![PandaAI](../assets/logo.png)

<p align="center">
[ <a href="../README.md">En</a> |
<a href="README_CN.md">中</a> |
<a href="README_FR.md">Fr</a> |
<b>日</b> ] 
</p>

[![Release](https://img.shields.io/pypi/v/pandasai?label=Release&style=flat-square)](https://pypi.org/project/pandasai/)
[![CI](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/ci-core.yml/badge.svg)](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/ci-core.yml/badge.svg)
[![CD](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/cd.yml/badge.svg)](https://github.com/sinaptik-ai/pandas-ai/actions/workflows/cd.yml/badge.svg)
[![Coverage](https://codecov.io/gh/sinaptik-ai/pandas-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/sinaptik-ai/pandas-ai)
[![Discord](https://dcbadge.vercel.app/api/server/kF7FqH2FwS?style=flat&compact=true)](https://discord.gg/KYKj9F2FRH)
[![Downloads](https://static.pepy.tech/badge/pandasai)](https://pepy.tech/project/pandasai) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZnO-njhL7TBOYPZaqvMvGtsjckZKrv2E?usp=sharing)

PandaAIは、自然言語でデータに質問することを容易にするPythonプラットフォームです。非技術ユーザーがデータとより自然な方法で対話するのを助け、技術ユーザーがデータを扱う際の時間と労力を節約するのに役立ちます。

# 🔧 はじめに

PandaAIの完全なドキュメントは[こちら](https://pandas-ai.readthedocs.io/en/latest/)で見つけることができます。

JupyterノートブックやStreamlitアプリでPandaAIを使用するか、リポジトリからクライアントとサーバーのアーキテクチャを使用することができます。

## ☁️ プラットフォームの使用

このライブラリは、強力なデータプラットフォームと併用することができ、わずか数行のコードでエンドツーエンドの会話型データ分析を可能にします。

データを読み込み、データフレームとして保存し、プラットフォームにプッシュします。

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

これで、チームはプラットフォームを通じて自然言語でこのデータにアクセスし、クエリを実行できます。

![PandaAI](assets/demo.gif)

## 📚 ライブラリの使用

### Pythonの要件

Pythonバージョン `3.8+ <3.12`

### 📦 インストール

PandaAIライブラリはpipまたはpoetryを使用してインストールできます。

pipを使用する場合:

```bash
pip install "pandasai>=3.0.0b2"
```

poetryを使用する場合:

```bash
poetry add "pandasai>=3.0.0b2"
```

### 💻 使用法

#### 質問する

```python
import pandasai as pai

# サンプルデータフレーム
df = pai.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "revenue": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})

# デフォルトでは、別のLLMを選択しない限り、BambooLLMを使用します。
# https://app.pandabi.ai でサインアップして無料のAPIキーを取得できます（.envファイルで設定することもできます）。
pai.api_key.set("your-pai-api-key")

df.chat('売上上位5か国はどれですか？')
```

```
中国、アメリカ、日本、ドイツ、オーストラリア
```

---

または、より複雑な質問をすることもできます:

```python
df.chat(
    "売上上位3か国の総売上はいくらですか？"
)
```

```
売上上位3か国の総売上は16500です。
```

#### チャートの可視化

PandaAIにチャートを生成させることもできます:

```python
df.chat(
    "各国のGDPを示すヒストグラムをプロットしてください。各バーに異なる色を使用してください",
)
```

<<<<<<< HEAD
![Chart](../assets/histogram-chart.png?raw=true)
=======
![Chart](assets/histogram-chart.png?raw=true)
>>>>>>> 2bfbc53f54f9ce6325ced7611cbb54b8eff721f9

#### 複数のデータフレーム

複数のデータフレームをPandaAIに渡し、それらに関連する質問をすることもできます。

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

# デフォルトでは、別のLLMを選択しない限り、BambooLLMを使用します。
# https://app.pandabi.ai でサインアップして無料のAPIキーを取得できます（.envファイルで設定することもできます）。
pai.api_key.set("your-pai-api-key")

pai.chat("誰が最も多く給与をもらっていますか？", employees_df, salaries_df)
```

```
Oliviaが最も多く給与をもらっています。
```

#### Dockerサンドボックス

PandaAIをDockerサンドボックスで実行し、安全で隔離された環境でコードを実行し、悪意のある攻撃のリスクを軽減することができます。

##### Pythonの要件

```bash
pip install "pandasai-docker"
```

##### 使用法

```python
import pandasai as pai
from pandasai_docker import DockerSandbox

# サンドボックスの初期化
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

# デフォルトでは、別のLLMを選択しない限り、BambooLLMを使用します。
# https://app.pandabi.ai でサインアップして無料のAPIキーを取得できます（.envファイルで設定することもできます）。
pai.api_key.set("your-pai-api-key")

pai.chat("誰が最も多く給与をもらっていますか？", employees_df, salaries_df, sandbox=sandbox)

# 終了時にサンドボックスを停止することを忘れないでください
sandbox.stop()
```

```
Oliviaが最も多く給与をもらっています。
```

その他の例は[examples](examples)ディレクトリで見つけることができます。

## 📜 ライセンス

PandaAIはMIT expatライセンスの下で利用可能ですが、このリポジトリの`pandasai/ee`ディレクトリは[こちら](https://github.com/sinaptik-ai/pandas-ai/blob/main/ee/LICENSE)に独自のライセンスがあります。

管理されたPandaAI Cloudまたはセルフホスト型のエンタープライズオファリングに興味がある場合は、[お問い合わせください](https://getpanda.ai/pricing)。

## リソース

> **ベータ版のお知らせ**  
> リリースv3は現在ベータ版です。以下のドキュメントと例は、進行中の機能と機能を反映しており、最終リリース前に変更される可能性があります。

- [ドキュメント](https://pandas-ai.readthedocs.io/en/latest/) 包括的なドキュメント
- [例](examples) サンプルノートブック
- [Discord](https://discord.gg/KYKj9F2FRH) コミュニティとPandaAIチームとのディスカッション

## 🤝 貢献

貢献は大歓迎です！未解決のイシューを確認し、プルリクエストを自由に開いてください。
詳細については、[貢献ガイドライン](CONTRIBUTING.md)を確認してください。

### ありがとうございます！

[![Contributors](https://contrib.rocks/image?repo=sinaptik-ai/pandas-ai)](https://github.com/sinaptik-ai/pandas-ai/graphs/contributors)