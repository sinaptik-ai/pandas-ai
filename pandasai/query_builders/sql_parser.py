from typing import List, Optional

import sqlglot
from sqlglot import ParseError, exp, parse_one
from sqlglot.optimizer.qualify_columns import quote_identifiers

from pandasai.exceptions import MaliciousQueryError


class SQLParser:
    @staticmethod
    def replace_table_and_column_names(query, table_mapping):
        """
        Transform a SQL query by replacing table names with either new table names or subqueries.

        Args:
            query (str): Original SQL query
            table_mapping (dict): Dictionary mapping original table names to either:
                           - actual table names (str)
                           - subqueries (str)
        """
        # Pre-parse all subqueries in mapping to avoid repeated parsing
        parsed_mapping = {}
        for key, value in table_mapping.items():
            try:
                parsed_mapping[key] = parse_one(value)
            except ParseError:
                raise ValueError(f"{value} is not a valid SQL expression")

        def transform_node(node):
            # Handle Table nodes
            if isinstance(node, exp.Table):
                original_name = node.name

                if original_name in table_mapping:
                    alias = node.alias or original_name
                    mapped_value = parsed_mapping[original_name]
                    if isinstance(mapped_value, exp.Alias):
                        return exp.Subquery(
                            this=mapped_value.this.this,
                            alias=alias,
                        )
                    elif isinstance(mapped_value, exp.Column):
                        return exp.Table(this=mapped_value.this, alias=alias)
                    return exp.Subquery(this=mapped_value, alias=alias)

            return node

        # Parse the SQL query
        parsed = parse_one(query)

        # Transform the query
        transformed = parsed.transform(transform_node)
        transformed = transformed.transform(quote_identifiers)

        # Convert back to SQL string
        return transformed.sql(pretty=True)

    @staticmethod
    def transpile_sql_dialect(
        query: str, to_dialect: str, from_dialect: Optional[str] = None
    ):
        placeholder = "___PLACEHOLDER___"
        query = query.replace("%s", placeholder)
        query = (
            parse_one(query, read=from_dialect) if from_dialect else parse_one(query)
        )
        result = query.sql(dialect=to_dialect, pretty=True)

        if to_dialect == "duckdb":
            return result.replace(placeholder, "?")

        return result.replace(placeholder, "%s")

    @staticmethod
    def extract_table_names(sql_query: str, dialect: str = "postgres") -> List[str]:
        # Parse the SQL query
        parsed = sqlglot.parse(sql_query, dialect=dialect)
        table_names = []
        cte_names = set()

        for stmt in parsed:
            # Identify and store CTE names
            for cte in stmt.find_all(exp.With):
                for cte_expr in cte.expressions:
                    cte_names.add(cte_expr.alias_or_name)

            # Extract table names, excluding CTEs
            for node in stmt.find_all(exp.Table):
                if node.name not in cte_names:  # Ignore CTE names
                    table_names.append(node.name)

        return table_names
