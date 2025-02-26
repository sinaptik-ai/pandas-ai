.PHONY: all format format_diff spell_check spell_fix tests tests-coverage integration docs help install_extension_deps test_extensions test_all install_deps test_core setup_python

all: help  ## default target executed when no arguments are given to make

#############################
# UNIT AND INTEGRATION TESTS
#############################

UNIT_TESTS_DIR ?= tests/unit_tests/
INTEGRATION_TESTS_DIR ?= tests/integration_tests/
# setup_python:  ## ensure we're using Python 3.10
# 	@echo "Setting up Python 3.10..."
# 	poetry env use python3.10

install_deps: setup_python  ## install core dependencies
	@echo "Installing core dependencies..."
	poetry install --all-extras --with dev

test_core: install_deps  ## run core tests only
	@echo "Running core tests..."
	poetry run pytest $(UNIT_TESTS_DIR) $(INTEGRATION_TESTS_DIR)

install_extension_deps: setup_python  ## install all extension dependencies
	@echo "Installing LLM extension dependencies..."
	@for dir in extensions/llms/*/; do \
		if [ -f "$$dir/pyproject.toml" ]; then \
			echo "Installing dependencies for $$dir"; \
			cd "$$dir" && poetry install --all-extras --with test && cd - || exit 1; \
		fi \
	done

	@echo "Installing connector extension dependencies..."
	@for dir in extensions/connectors/*/; do \
		if [ -f "$$dir/pyproject.toml" ]; then \
			echo "Installing dependencies for $$dir"; \
			cd "$$dir" && poetry install --all-extras --with test && cd - || exit 1; \
		fi \
	done

	@echo "Installing enterprise extension dependencies..."
	@for dir in extensions/ee/*/*/; do \
		if [ -f "$$dir/pyproject.toml" ]; then \
			echo "Installing dependencies for $$dir"; \
			cd "$$dir" && poetry install --all-extras --with test && cd - || exit 1; \
		fi \
	done

test_extensions: install_extension_deps  ## run all extension tests
	@echo "Running LLM extension tests..."
	@for dir in extensions/llms/*/; do \
		if [ -d "$$dir/tests" ]; then \
			echo "Running tests for $$dir"; \
			cd "$$dir" && poetry run pytest tests/ && cd - || exit 1; \
		fi \
	done

	@echo "Running connector extension tests..."
	@for dir in extensions/connectors/*/; do \
		if [ -d "$$dir/tests" ]; then \
			echo "Running tests for $$dir"; \
			cd "$$dir" && poetry run pytest tests/ && cd - || exit 1; \
		fi \
	done

	@echo "Running enterprise extension tests..."
	@for dir in extensions/ee/*/*/; do \
		if [ -d "$$dir/tests" ]; then \
			echo "Running tests for $$dir"; \
			cd "$$dir" && poetry run pytest tests/ && cd - || exit 1; \
		fi \
	done

test_all: test_core test_extensions  ## run all tests (core and extensions)

tests-coverage: install_deps  ## run unit tests and generate coverage report
	poetry run coverage run --source=pandasai -m pytest $(UNIT_TESTS_DIR) $(INTEGRATION_TESTS_DIR)
	poetry run coverage xml

###########################
# SPELLCHECK AND FORMATTING
###########################

IGNORE_FORMATS ?= "*.csv,*.txt,*.lock,*.log"

format:  ## run code formatters
	poetry run ruff format pandasai examples tests
	poetry run ruff --select I --fix pandasai examples tests

format_diff:  ## run code formatters in diff mode
	poetry run ruff format pandasai examples tests --diff
	poetry run ruff --select I pandasai examples tests

spell_check:  ## run codespell on the project
	poetry run codespell --toml pyproject.toml --ignore-words=ignore-words.txt --skip=$(IGNORE_FORMATS)

spell_fix:  ## run codespell on the project and fix the errors
	poetry run codespell --toml pyproject.toml --ignore-words=ignore-words.txt --skip=$(IGNORE_FORMATS) -w

######################
# DOCS
######################

docs:  ## run docs serving
	mkdocs serve

######################
# HELP
######################

help:  ## Show this help message.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'