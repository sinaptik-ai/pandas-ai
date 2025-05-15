# 🐼 Contributing to PandasAI

Hi there! We're thrilled that you'd like to contribute to this project. Your help is essential for keeping it great.

## 🤝 How to submit a contribution

To make a contribution, follow the following steps:

1. Fork and clone this repository
2. Do the changes on your fork
3. If you modified the code (new feature or bug-fix), please add tests for it
4. Check the linting [see below](https://github.com/gventuri/pandas-ai/blob/main/CONTRIBUTING.md#-linting)
5. Ensure that all tests pass [see below](https://github.com/gventuri/pandas-ai/blob/main/CONTRIBUTING.md#-testing)
6. Submit a pull request

For more details about pull requests, please read [GitHub's guides](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

### 📦 Package manager

We use `poetry` as our package manager. You can install poetry by following the instructions [here](https://python-poetry.org/docs/#installation).

Please DO NOT use pip or conda to install the dependencies. Instead, use poetry:

```bash
poetry install --all-extras --with dev
```

### 📌 Pre-commit

To ensure our standards, make sure to install pre-commit before starting to contribute.

```bash
pre-commit install
```

### 🧹 Linting

We use `ruff` to lint our code. You can run the linter by running the following command:

```bash
make format_diff
```

Make sure that the linter does not report any errors or warnings before submitting a pull request.

### Code Format with `ruff-format`

We use `ruff` to reformat the code by running the following command:

```bash
make format
```

### Spell check

We usee `codespell` to check the spelling of our code. You can run codespell by running the following command:

```bash
make spell_fix
```


### 🧪 Testing

We use `pytest` to test our code. You can run the tests by running the following command:

```bash
make tests
```

Make sure that all tests pass before submitting a pull request.

## 🚀 Release Process

At the moment, the release process is manual. We try to make frequent releases. Usually, we release a new version when we have a new feature or bugfix. A developer with admin rights to the repository will create a new release on GitHub, and then publish the new version to PyPI.
