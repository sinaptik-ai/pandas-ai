from unittest.mock import MagicMock, patch

import duckdb
import pandas as pd
import pytest

from pandasai.data_loader.semantic_layer_schema import SemanticLayerSchema
from pandasai.data_loader.view_loader import ViewDatasetLoader
from pandasai.dataframe.virtual_dataframe import VirtualDataFrame
from pandasai.query_builders import ViewQueryBuilder


class TestViewDatasetLoader:
    @pytest.fixture
    def view_schema(self):
        """Create a test view schema that combines data from two datasets."""
        return SemanticLayerSchema(
            name="sales_overview",
            view=True,
            columns=[
                {"name": "sales.product_id", "type": "string"},
                {"name": "sales.amount", "type": "float"},
                {"name": "products.name", "type": "string"},
                {"name": "products.category", "type": "string"},
            ],
            relations=[
                {
                    "name": "product_relation",
                    "from": "sales.product_id",
                    "to": "products.id",
                }
            ],
        )

    @pytest.fixture
    def view_schema_with_group_by(self):
        """Create a test view schema with group by functionality."""
        return SemanticLayerSchema(
            name="sales_by_category",
            view=True,
            columns=[
                {"name": "products.category", "type": "string"},
                {
                    "name": "sales.amount",
                    "type": "float",
                    "expression": "SUM(sales.amount)",
                },
                {"name": "sales.count", "type": "integer", "expression": "COUNT(*)"},
                {
                    "name": "sales.avg_amount",
                    "type": "float",
                    "expression": "AVG(sales.amount)",
                },
            ],
            relations=[
                {
                    "name": "product_relation",
                    "from": "sales.product_id",
                    "to": "products.id",
                }
            ],
            group_by=["products.category"],
        )

    def create_mock_loader(self, name, source_type="csv"):
        """Helper method to create properly configured mock loaders"""
        mock_loader = MagicMock()
        mock_schema = MagicMock()
        mock_source = MagicMock()

        # Configure the source
        mock_source.type = source_type

        # Configure the schema
        mock_schema.name = name
        mock_schema.source = mock_source

        # Set the schema on the loader
        mock_loader.schema = mock_schema

        return mock_loader

    def test_init(self, view_schema):
        """Test initialization of ViewDatasetLoader."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Create mock loaders for the dependencies
            mock_sales_loader = self.create_mock_loader("sales")
            mock_products_loader = self.create_mock_loader("products")

            # Configure the mock to return different loaders based on the path
            def side_effect(path):
                if "sales" in path:
                    return mock_sales_loader
                elif "products" in path:
                    return mock_products_loader
                raise ValueError(f"Unexpected path: {path}")

            mock_create_loader.side_effect = side_effect

            loader = ViewDatasetLoader(view_schema, "test/sales-overview")

            # Verify dependencies were loaded
            assert "sales" in loader.dependencies_datasets
            assert "products" in loader.dependencies_datasets
            assert len(loader.schema_dependencies_dict) == 2

            # Verify query builder was created
            assert isinstance(loader.query_builder, ViewQueryBuilder)

    def test_get_dependencies_datasets(self, view_schema):
        """Test extraction of dependency dataset names from relations."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Setup mock loaders
            mock_sales_loader = self.create_mock_loader("sales")
            mock_products_loader = self.create_mock_loader("products")

            mock_create_loader.side_effect = (
                lambda path: mock_sales_loader
                if "sales" in path
                else mock_products_loader
            )

            loader = ViewDatasetLoader(view_schema, "test/sales-overview")

            dependencies = loader._get_dependencies_datasets()
            assert "sales" in dependencies
            assert "products" in dependencies
            assert len(dependencies) == 2

    def test_get_dependencies_schemas_missing_dependency(self, view_schema):
        """Test error handling when a dependency is missing."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Make the factory raise FileNotFoundError for a dependency
            mock_create_loader.side_effect = FileNotFoundError("Dataset not found")

            with pytest.raises(FileNotFoundError, match="Missing required dataset"):
                ViewDatasetLoader(view_schema, "test/sales-overview")

    def test_get_dependencies_schemas_incompatible_sources(self, view_schema):
        """Test error handling when sources are incompatible."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Create mock loaders with incompatible sources
            mock_sales_loader = self.create_mock_loader("sales", "csv")
            mock_products_loader = self.create_mock_loader("products", "postgres")

            # Configure the mock to return different loaders
            def side_effect(path):
                if "sales" in path:
                    return mock_sales_loader
                elif "products" in path:
                    return mock_products_loader
                raise ValueError(f"Unexpected path: {path}")

            mock_create_loader.side_effect = side_effect

            # Mock the compatibility check to return False
            with patch(
                "pandasai.query_builders.base_query_builder.BaseQueryBuilder.check_compatible_sources",
                return_value=False,
            ):
                with pytest.raises(ValueError, match="compatible for a view"):
                    ViewDatasetLoader(view_schema, "test/sales-overview")

    def test_load(self, view_schema):
        """Test that load returns a VirtualDataFrame."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Setup mock loaders
            mock_sales_loader = self.create_mock_loader("sales")
            mock_products_loader = self.create_mock_loader("products")

            mock_create_loader.side_effect = (
                lambda path: mock_sales_loader
                if "sales" in path
                else mock_products_loader
            )

            loader = ViewDatasetLoader(view_schema, "test/sales-overview")

            result = loader.load()

            assert isinstance(result, VirtualDataFrame)
            assert result.schema == view_schema
            assert result.path == "test/sales-overview"

    def test_execute_local_query(self, view_schema):
        """Test execution of local queries with DuckDB."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Setup mock loaders
            mock_sales_loader = self.create_mock_loader("sales")
            mock_products_loader = self.create_mock_loader("products")

            mock_create_loader.side_effect = (
                lambda path: mock_sales_loader
                if "sales" in path
                else mock_products_loader
            )

            with patch(
                "pandasai.data_loader.view_loader.DuckDBConnectionManager"
            ) as mock_db_manager_class:
                mock_db_manager = MagicMock()
                mock_db_manager_class.return_value = mock_db_manager

                # Mock result of the query
                mock_sql_result = MagicMock()
                mock_sql_result.df.return_value = pd.DataFrame({"result": [1, 2, 3]})
                mock_db_manager.sql.return_value = mock_sql_result

                loader = ViewDatasetLoader(view_schema, "test/sales-overview")

                # Manually set the loader's schema_dependencies_dict
                loader.schema_dependencies_dict = {
                    "sales": mock_sales_loader,
                    "products": mock_products_loader,
                }

                result = loader.execute_local_query(
                    "SELECT * FROM sales_overview", params=[]
                )

                # Verify the query was executed correctly
                mock_db_manager.sql.assert_called_once()
                assert isinstance(result, pd.DataFrame)

    def test_execute_local_query_error(self, view_schema):
        """Test error handling in execute_local_query."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Setup mock loaders
            mock_sales_loader = self.create_mock_loader("sales")
            mock_products_loader = self.create_mock_loader("products")

            mock_create_loader.side_effect = (
                lambda path: mock_sales_loader
                if "sales" in path
                else mock_products_loader
            )

            with patch(
                "pandasai.data_loader.view_loader.DuckDBConnectionManager"
            ) as mock_db_manager_class:
                mock_db_manager = MagicMock()
                mock_db_manager_class.return_value = mock_db_manager

                # Make the SQL execution raise an error
                mock_db_manager.sql.side_effect = duckdb.Error("Test SQL error")

                loader = ViewDatasetLoader(view_schema, "test/sales-overview")

                # Manually set the loader's schema_dependencies_dict
                loader.schema_dependencies_dict = {
                    "sales": mock_sales_loader,
                    "products": mock_products_loader,
                }

                with pytest.raises(RuntimeError, match="SQL execution failed"):
                    loader.execute_local_query("SELECT * FROM invalid_table")

    def test_execute_query_with_group_by(self, view_schema_with_group_by):
        """Test execution of queries with GROUP BY functionality."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Setup mock loaders
            mock_sales_loader = self.create_mock_loader("sales")
            mock_products_loader = self.create_mock_loader("products")

            # Add LocalDatasetLoader-specific methods
            mock_sales_loader.register_table = MagicMock()
            mock_products_loader.register_table = MagicMock()

            mock_create_loader.side_effect = (
                lambda path: mock_sales_loader
                if "sales" in path
                else mock_products_loader
            )

            with patch(
                "pandasai.data_loader.view_loader.DuckDBConnectionManager"
            ) as mock_db_manager_class:
                mock_db_manager = MagicMock()
                mock_db_manager_class.return_value = mock_db_manager

                # Create expected group by result
                expected_result = pd.DataFrame(
                    {
                        "category": ["Electronics", "Clothing", "Food"],
                        "amount": [1000.0, 500.0, 250.0],
                        "count": [10, 5, 2],
                        "avg_amount": [100.0, 100.0, 125.0],
                    }
                )

                # Mock result of the query
                mock_sql_result = MagicMock()
                mock_sql_result.df.return_value = expected_result
                mock_db_manager.sql.return_value = mock_sql_result

                loader = ViewDatasetLoader(
                    view_schema_with_group_by, "test/sales-by-category"
                )

                # Manually set the loader's schema_dependencies_dict
                loader.schema_dependencies_dict = {
                    "sales": mock_sales_loader,
                    "products": mock_products_loader,
                }

                # Test that the query builder generates the correct SQL with GROUP BY
                with patch.object(
                    loader.query_builder, "build_query"
                ) as mock_build_query:
                    mock_build_query.return_value = """
                    SELECT 
                        products.category,
                        SUM(sales.amount) AS amount,
                        COUNT(*) AS count,
                        AVG(sales.amount) AS avg_amount
                    FROM sales
                    JOIN products ON sales.product_id = products.id
                    GROUP BY products.category
                    """

                    result = loader.execute_local_query(
                        loader.query_builder.build_query()
                    )

                    # Verify the query was built correctly
                    mock_build_query.assert_called_once()

                    # Verify the SQL was executed
                    mock_db_manager.sql.assert_called_once()

                    # Check the result
                    assert isinstance(result, pd.DataFrame)
                    assert result.equals(expected_result)
                    assert list(result.columns) == [
                        "category",
                        "amount",
                        "count",
                        "avg_amount",
                    ]

    def test_execute_query_with_custom_fixtures(
        self, mysql_view_schema, mysql_view_dependencies_dict
    ):
        """Test execution of queries using the provided fixtures."""
        with patch(
            "pandasai.data_loader.loader.DatasetLoader.create_loader_from_path"
        ) as mock_create_loader:
            # Configure the mock to return loaders from the fixture
            def side_effect(path):
                if "parents" in path:
                    return mysql_view_dependencies_dict["parents"]
                elif "children" in path:
                    return mysql_view_dependencies_dict["children"]
                raise ValueError(f"Unexpected path: {path}")

            mock_create_loader.side_effect = side_effect

            with patch(
                "pandasai.query_builders.base_query_builder.BaseQueryBuilder.check_compatible_sources",
                return_value=True,
            ):
                # Convert dataset paths for testing
                dataset_path = f"test/{mysql_view_schema.name}"
                if "_" in dataset_path:
                    dataset_path = dataset_path.replace("_", "-")

                loader = ViewDatasetLoader(mysql_view_schema, dataset_path)

                # Test that the dependencies were correctly loaded
                assert len(loader.dependencies_datasets) > 0
                assert len(loader.schema_dependencies_dict) > 0

                # Mock execution of a query
                with patch.object(loader, "execute_query") as mock_execute_query:
                    mock_execute_query.return_value = pd.DataFrame(
                        {
                            "parents.id": [1, 2, 3],
                            "parents.name": ["Parent1", "Parent2", "Parent3"],
                            "children.name": ["Child1", "Child2", "Child3"],
                        }
                    )

                    result = loader.load()

                    # Verify that the loader created a VirtualDataFrame with the right schema
                    assert isinstance(result, VirtualDataFrame)
                    assert result.schema == mysql_view_schema
