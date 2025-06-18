import os
import re

import click

from pandasai import DatasetLoader
from pandasai.data_loader.semantic_layer_schema import (
    SemanticLayerSchema,
    Source,
    SQLConnectionConfig,
)
from pandasai.helpers.path import find_project_root, get_validated_dataset_path


def validate_api_key(api_key: str) -> bool:
    """Validate PandaBI API key format."""
    pattern = r"^PAI-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    return bool(re.match(pattern, api_key))


@click.group()
def cli():
    """🐼 PandasAI CLI - Manage your datasets with ease"""
    pass


@cli.group()
def dataset():
    """📊 Dataset management commands"""
    pass


@dataset.command()
def create():
    """🎨 Create a new dataset through a guided process"""
    click.echo("🚀 Let's create a new dataset!\n")

    # Get organization and dataset name
    while True:
        path = click.prompt("📁 Enter the dataset path (format: organization/dataset)")
        try:
            org_name, dataset_name = get_validated_dataset_path(path)
            break
        except ValueError as e:
            click.echo(f"❌ Error: {str(e)}")

    dataset_directory = os.path.join(
        find_project_root(), "datasets", org_name, dataset_name
    )

    # Check if dataset already exists
    if os.path.exists(dataset_directory):
        schema_path = os.path.join(dataset_directory, "schema.yaml")
        if os.path.exists(schema_path):
            click.echo(f"❌ Error: Dataset already exists at path: {path}")
            return

    # Get dataset metadata
    name = click.prompt("📝 Enter dataset name", default=dataset_name)
    description = click.prompt("📋 Enter dataset description", default="")

    # Get source configuration
    source_type = click.prompt(
        "🔌 Enter source type",
        type=click.Choice(["mysql", "postgres"]),
        default="mysql",
    )

    table_name = click.prompt("📦 Enter table name")

    # Build connection configuration
    connection_config = {
        "host": click.prompt("🌐 Enter host", default="localhost"),
        "port": click.prompt("🔍 Enter port", type=int),
        "database": click.prompt("💾 Enter database name"),
        "user": click.prompt("👤 Enter username"),
        "password": click.prompt("🔑 Enter password", hide_input=True),
    }

    # Create source configuration
    source = {
        "type": source_type,
        "table": table_name,
        "connection": SQLConnectionConfig(**connection_config),
    }

    # Create schema
    schema = SemanticLayerSchema(
        name=name, description=description, source=Source(**source)
    )

    # Create directory and save schema
    os.makedirs(dataset_directory, exist_ok=True)
    schema_path = os.path.join(dataset_directory, "schema.yaml")

    with open(schema_path, "w") as yml_file:
        yml_file.write(schema.to_yaml())

    click.echo(f"\n✨ Dataset created successfully at: {dataset_directory}")


@cli.command()
@click.argument("api_key")
def login(api_key: str):
    """🔑 Authenticate with your PandaBI API key"""
    if not validate_api_key(api_key):
        click.echo(
            "❌ Invalid API key format. Expected format: PAI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        )
        return

    env_path = os.path.join(find_project_root(), ".env")
    env_content = ""
    new_line = f"PANDABI_API_KEY={api_key}\n"

    # Read existing .env if it exists
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
            # Filter out existing PANDABI_API_KEY line if present
            lines = [line for line in lines if not line.startswith("PANDABI_API_KEY=")]
            env_content = "".join(lines)
            if env_content and not env_content.endswith("\n"):
                env_content += "\n"

    # Write updated content
    with open(env_path, "w") as f:
        f.write(env_content + new_line)

    click.echo("✅ Successfully authenticated with PandaBI!")


if __name__ == "__main__":
    cli()
