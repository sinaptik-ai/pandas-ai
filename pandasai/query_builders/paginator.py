import datetime
import json
import uuid
from typing import List, Optional, Tuple

import sqlglot
from pydantic import BaseModel, Field, field_validator

from pandasai.helpers.sql_sanitizer import is_sql_query


class PaginationParams(BaseModel):
    """Parameters for pagination requests"""

    page: int = Field(ge=1, description="Page number, starting from 1")
    page_size: int = Field(
        ge=1, le=100, description="Number of items per page, maximum 100"
    )
    search: Optional[str] = Field(
        None, description="Search term to filter across all fields"
    )
    sort_by: Optional[str] = Field(None, description="Column to sort by")
    sort_order: Optional[str] = Field(
        None, pattern="^(asc|desc)$", description="Sort order (asc or desc)"
    )
    filters: Optional[str] = Field(None, description="Filters to apply to the data")

    @field_validator("search", "filters", "sort_by", "sort_order")
    @classmethod
    def not_sql(cls, field):
        if is_sql_query(str(field)):
            raise ValueError(
                f"SQL queries are not allowed in pagination parameters: {field}"
            )
        return field


class DatasetPaginator:
    @staticmethod
    def is_float(value: str) -> bool:
        try:
            # Try to cast the value to a number
            float(value)
            return True
        except (ValueError, TypeError):
            # If it fails, it's not a number
            return False

    @staticmethod
    def is_valid_boolean(value):
        """Check if the value is a valid boolean."""
        return (
            value.lower() in ["true", "false"]
            if isinstance(value, str)
            else isinstance(value, bool)
        )

    @staticmethod
    def is_valid_uuid(value):
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_valid_datetime(value: str) -> bool:
        try:
            datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
            return True
        except ValueError:
            return False

    @staticmethod
    def apply_pagination(
        query: str,
        columns: List[dict],
        pagination: Optional[PaginationParams],
        target_dialect: str = "postgres",
    ) -> Tuple[str, List]:
        """
        Apply pagination to a SQL query.

        Args:
            query (str): The SQL query to apply pagination to
            columns (List[dict]): A list of dictionaries containing
                information about the columns in the result set. Each
                dictionary should have the following structure:
                    {
                        "name": str,
                        "type": str
                    }
                The type should be one of: "string", "number", "integer", "float",
                "boolean", "datetime"
            pagination (Optional[PaginationParams]): The pagination parameters
                to apply to the query. If None, the query is returned unchanged
            target_dialect (str): The SQL dialect to generate the query for.
                Defaults to "postgres".

        Returns:
            Tuple[str, List]: A tuple containing the modified SQL query and a
                list of parameters to pass to the query.
        """

        params = []

        if not pagination:
            return query, params

        # Convert query from target dialect to postgres to generate standardized pagination query
        query = sqlglot.transpile(query, read=target_dialect, write="postgres")[0]

        filtering_query = f"SELECT * FROM ({query}) AS filtered_data"
        conditions = []

        # Handle search functionality
        if pagination.search:
            search_conditions = []
            for column in columns:
                column_name = column["name"]
                column_type = column["type"]

                if column_type == "string":
                    search_conditions.append(f'"{column_name}" ILIKE %s')
                    params.append(f"%{pagination.search}%")

                elif column_type == "float" and DatasetPaginator.is_float(
                    pagination.search
                ):
                    search_conditions.append(f'"{column_name}" = %s')
                    params.append(pagination.search)

                elif (
                    column_type in ["number", "integer"]
                    and pagination.search.isnumeric()
                ):
                    search_conditions.append(f'"{column_name}" = %s')
                    params.append(pagination.search)

                elif column_type == "datetime" and DatasetPaginator.is_valid_datetime(
                    pagination.search
                ):
                    search_conditions.append(f'"{column_name}" = %s')
                    params.append(
                        datetime.datetime.strptime(
                            pagination.search, "%Y-%m-%d %H:%M:%S"
                        )
                    )

                elif column_type == "boolean" and DatasetPaginator.is_valid_boolean(
                    pagination.search
                ):
                    search_conditions.append(f'"{column_name}" = %s')
                    params.append(pagination.search)

                elif column_type == "uuid" and DatasetPaginator.is_valid_uuid(
                    pagination.search
                ):
                    search_conditions.append(f'"{column_name}"::TEXT = %s')
                    params.append(pagination.search)

            if search_conditions:
                conditions.append(" OR ".join(search_conditions))

        # Handle filters
        if pagination.filters:
            try:
                filters = (
                    json.loads(pagination.filters)
                    if isinstance(pagination.filters, str)
                    else pagination.filters
                )
                for column, values in filters.items():
                    if not isinstance(values, list):
                        values = [values]
                    placeholders = ", ".join(["%s"] * len(values))
                    conditions.append(f'"{column}" IN ({placeholders})')
                    params.extend(values)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid filters format: {e}")

        # Add WHERE clause if conditions exist
        if conditions:
            filtering_query += " WHERE " + " AND ".join(conditions)

        # Handle sorting
        if pagination.sort_by and pagination.sort_order:
            if not any(pagination.sort_by == column["name"] for column in columns):
                raise ValueError(
                    f"Sort column '{pagination.sort_by}' not found in available columns"
                )

            filtering_query += (
                f' ORDER BY "{pagination.sort_by}" {pagination.sort_order.upper()}'
            )

        # Handle page and page_size
        if pagination.page and pagination.page_size:
            filtering_query += " LIMIT %s OFFSET %s"
            params.extend(
                [pagination.page_size, (pagination.page - 1) * pagination.page_size]
            )

        return filtering_query, params
