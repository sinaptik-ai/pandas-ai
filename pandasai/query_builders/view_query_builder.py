import re
from typing import Dict, List

from sqlglot import exp, expressions, parse_one, select
from sqlglot.expressions import Subquery
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from ..data_loader.loader import DatasetLoader
from ..data_loader.semantic_layer_schema import SemanticLayerSchema, Transformation
from ..helpers.sql_sanitizer import sanitize_view_column_name
from .base_query_builder import BaseQueryBuilder
from .sql_transformation_manager import SQLTransformationManager


class ViewQueryBuilder(BaseQueryBuilder):
    def __init__(
        self,
        schema: SemanticLayerSchema,
        schema_dependencies_dict: Dict[str, DatasetLoader],
    ):
        super().__init__(schema)
        self.schema_dependencies_dict = schema_dependencies_dict

    @staticmethod
    def normalize_view_column_name(name: str) -> str:
        return normalize_identifiers(parse_one(sanitize_view_column_name(name))).sql()

    @staticmethod
    def normalize_view_column_alias(name: str) -> str:
        return normalize_identifiers(
            sanitize_view_column_name(name).replace(".", "_")
        ).sql()

    def _get_group_by_columns(self) -> list[str]:
        """Get the group by columns with proper view column aliasing."""
        group_by_cols = []
        for col in self.schema.group_by:
            group_by_cols.append(self.normalize_view_column_alias(col))
        return group_by_cols

    def _get_aliases(self) -> list[str]:
        return [
            col.alias or self.normalize_view_column_alias(col.name)
            for col in self.schema.columns
        ]

    def _get_columns(self) -> list[str]:
        columns = []
        aliases = self._get_aliases()
        for i, col in enumerate(self.schema.columns):
            if col.expression:
                # Pre-process the expression to handle hyphens between letters
                expr = re.sub(r"([a-zA-Z])-([a-zA-Z])", r"\1_\2", col.expression)
                expr = re.sub(r"([a-zA-Z])\.([a-zA-Z])", r"\1_\2", expr)
                column_expr = parse_one(expr).sql()
            else:
                column_expr = self.normalize_view_column_alias(col.name)

            # Apply any transformations defined for this column
            column_expr = SQLTransformationManager.apply_column_transformations(
                column_expr, col.name, self.schema.transformations
            )

            alias = aliases[i]
            column_expr = f"{column_expr} AS {alias}"

            columns.append(column_expr)

        return columns

    def build_query(self) -> str:
        """Build the SQL query with proper group by column aliasing."""
        query = select(*self._get_aliases()).from_(self._get_table_expression())

        if self._check_distinct():
            query = query.distinct()

        if self.schema.order_by:
            query = query.order_by(*self.schema.order_by)
        if self.schema.limit:
            query = query.limit(self.schema.limit)
        return query.sql(pretty=True)

    def get_head_query(self, n=5):
        """Get the head query with proper group by column aliasing."""
        query = select(*self._get_aliases()).from_(self._get_table_expression())

        if self._check_distinct():
            query = query.distinct()

        query = query.limit(n)
        return query.sql(pretty=True)

    def _get_sub_query_from_loader(self, loader: DatasetLoader) -> Subquery:
        sub_query = parse_one(loader.query_builder.build_query())
        return exp.Subquery(this=sub_query, alias=loader.schema.name)

    def _get_table_expression(self) -> str:
        relations = self.schema.relations
        columns = self.schema.columns
        first_dataset = (
            relations[0].from_.split(".")[0]
            if relations
            else columns[0].name.split(".")[0]
        )
        first_loader = self.schema_dependencies_dict[first_dataset]
        first_query = self._get_sub_query_from_loader(first_loader)

        columns = [
            f"{self.normalize_view_column_name(col.name)} AS {self.normalize_view_column_alias(col.name)}"
            for col in self.schema.columns
        ]

        query = select(*columns).from_(first_query)

        # Group relations by target dataset to combine multiple join conditions
        join_conditions = {}
        for relation in relations:
            to_datasets = relation.to.split(".")[0]
            if to_datasets not in join_conditions:
                join_conditions[to_datasets] = []
            join_conditions[to_datasets].append(
                f"{sanitize_view_column_name(relation.from_)} = {sanitize_view_column_name(relation.to)}"
            )

        # Create joins with combined conditions
        for to_datasets, conditions in join_conditions.items():
            loader = self.schema_dependencies_dict[to_datasets]
            subquery = self._get_sub_query_from_loader(loader)
            query = query.join(
                subquery,
                on=" AND ".join(conditions),
                append=True,
            )
        alias = normalize_identifiers(self.schema.name).sql()

        subquery = exp.Subquery(this=query).sql(pretty=True)

        final_query = select(*self._get_columns()).from_(subquery)

        if self.schema.group_by:
            final_query = final_query.group_by(
                *[normalize_identifiers(col) for col in self._get_group_by_columns()]
            )

        return exp.Subquery(this=final_query, alias=alias).sql(pretty=True)
