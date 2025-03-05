import unittest
from unittest.mock import MagicMock, patch

from pandasai.data_loader.semantic_layer_schema import (
    Column,
    SemanticLayerSchema,
    Source,
    SQLConnectionConfig,
)
from pandasai.query_builders.base_query_builder import BaseQueryBuilder
from pandasai.query_builders.local_query_builder import LocalQueryBuilder
from pandasai.query_builders.sql_query_builder import SqlQueryBuilder
from pandasai.query_builders.view_query_builder import ViewQueryBuilder


class TestGroupByQueries(unittest.TestCase):
    def setUp(self):
        # Setup common test data
        self.base_schema = SemanticLayerSchema(
            name="sales",
            source=Source(type="csv", path="/path/to/sales.csv"),
            columns=[
                Column(name="category"),
                Column(name="region"),
                Column(name="amount", expression="sum(amount)", alias="total_sales"),
                Column(
                    name="quantity", expression="avg(quantity)", alias="avg_quantity"
                ),
            ],
            group_by=["category", "region"],
        )

        # Setup for SQL query builder
        self.sql_schema = SemanticLayerSchema(
            name="sales",
            source=Source(
                type="mysql",
                connection=SQLConnectionConfig(
                    host="localhost",
                    port=3306,
                    database="test",
                    user="user",
                    password="pass",
                ),
                table="sales",
            ),
            columns=[
                Column(name="category"),
                Column(name="region"),
                Column(name="amount", expression="sum(amount)", alias="total_sales"),
                Column(
                    name="quantity", expression="avg(quantity)", alias="avg_quantity"
                ),
            ],
            group_by=["category", "region"],
        )

        # Setup for view query builder
        self.view_schema = SemanticLayerSchema(
            name="sales_view",
            view=True,
            columns=[
                Column(name="sales.category"),
                Column(name="sales.region"),
                Column(
                    name="sales.amount", expression="sum(amount)", alias="total_sales"
                ),
                Column(
                    name="sales.quantity",
                    expression="avg(quantity)",
                    alias="avg_quantity",
                ),
            ],
            group_by=["sales.category", "sales.region"],
        )

    def test_base_query_builder(self):
        builder = BaseQueryBuilder(self.base_schema)
        query = builder.build_query()

        expected = (
            "SELECT\n"
            '  "category",\n'
            '  "region",\n'
            '  SUM("amount") AS "total_sales",\n'
            '  AVG("quantity") AS "avg_quantity"\n'
            'FROM "sales"\n'
            "GROUP BY\n"
            '  "category",\n'
            '  "region"'
        )
        self.assertEqual(query.strip(), expected.strip())

    def test_local_query_builder(self):
        with patch(
            "pandasai.query_builders.local_query_builder.ConfigManager.get"
        ) as mock_config_get:
            # Mock the return of `ConfigManager.get()`
            mock_config = MagicMock()
            mock_config.file_manager.abs_path.return_value = "/mocked/absolute/path"
            mock_config_get.return_value = mock_config
            builder = LocalQueryBuilder(self.base_schema, "test/test")
            query = builder.build_query()

            expected = (
                "SELECT\n"
                '  "category",\n'
                '  "region",\n'
                '  SUM("amount") AS "total_sales",\n'
                '  AVG("quantity") AS "avg_quantity"\n'
                "FROM READ_CSV('/mocked/absolute/path')\n"
                "GROUP BY\n"
                '  "category",\n'
                '  "region"'
            )
            self.assertEqual(query.strip(), expected.strip())

    def test_sql_query_builder(self):
        builder = SqlQueryBuilder(self.sql_schema)
        query = builder.build_query()

        expected = (
            "SELECT\n"
            '  "category",\n'
            '  "region",\n'
            '  SUM("amount") AS "total_sales",\n'
            '  AVG("quantity") AS "avg_quantity"\n'
            'FROM "sales"\n'
            "GROUP BY\n"
            '  "category",\n'
            '  "region"'
        )
        self.assertEqual(query.strip(), expected.strip())

    def test_invalid_group_by(self):
        # Test when an aggregated column is incorrectly included in group_by
        with self.assertRaises(ValueError) as context:
            SemanticLayerSchema(
                name="sales",
                columns=[
                    Column(name="category"),
                    Column(name="amount", expression="sum"),
                ],
                group_by=["category", "amount"],  # amount should not be in group_by
            )

        self.assertTrue(
            "Column 'amount' cannot be in group_by because it has an aggregation expression"
            in str(context.exception)
        )

        # Test when a non-aggregated column is not in group_by
        with self.assertRaises(ValueError) as context:
            SemanticLayerSchema(
                name="sales",
                columns=[
                    Column(name="category"),
                    Column(name="region"),  # Missing from group_by
                    Column(name="amount", expression="sum"),
                ],
                group_by=["category"],
            )

        self.assertTrue(
            "Column 'region' must either be in group_by or have an aggregation expression"
            in str(context.exception)
        )

    def test_no_group_by(self):
        # Test normal query without group by
        schema = SemanticLayerSchema(
            name="sales",
            source=Source(type="csv", path="/path/to/sales.csv"),
            columns=[
                Column(name="category"),
                Column(name="amount"),
            ],
        )
        builder = BaseQueryBuilder(schema)
        query = builder.build_query()

        expected = 'SELECT\n  "category",\n  "amount"\nFROM "sales"'
        self.assertEqual(query.strip(), expected.strip())
