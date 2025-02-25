from typing import Any, Dict, List, Optional

from pandasai.data_loader.semantic_layer_schema import (
    Transformation,
    TransformationParams,
)


class SQLTransformationManager:
    """Manages SQL-based transformations for query expressions."""

    @staticmethod
    def apply_transformations(expr: str, transformations: List[Transformation]) -> str:
        if not transformations:
            return expr

        transformed_expr = expr
        for transformation in transformations:
            method_name = f"_{transformation.type}"
            if hasattr(SQLTransformationManager, method_name):
                method = getattr(SQLTransformationManager, method_name)
                transformed_expr = method(transformed_expr, transformation.params)

        return transformed_expr

    @staticmethod
    def _anonymize(expr: str, params: TransformationParams) -> str:
        # Basic hashing for anonymization
        return f"MD5({expr})"

    @staticmethod
    def _fill_na(expr: str, params: TransformationParams) -> str:
        return f"COALESCE({expr}, {params.value})"

    @staticmethod
    def _map_values(expr: str, params: TransformationParams) -> str:
        if not params.mapping:
            return expr

        case_stmt = (
            "CASE "
            + " ".join(
                f"WHEN {expr} = '{key}' THEN '{value}'"
                for key, value in params.mapping.items()
            )
            + f" ELSE {expr} END"
        )

        return case_stmt

    @staticmethod
    def _to_lowercase(expr: str, params: TransformationParams) -> str:
        return f"LOWER({expr})"

    @staticmethod
    def _to_uppercase(expr: str, params: TransformationParams) -> str:
        return f"UPPER({expr})"

    @staticmethod
    def _round_numbers(expr: str, params: TransformationParams) -> str:
        decimals = params.decimals or 0
        return f"ROUND({expr}, {decimals})"

    @staticmethod
    def _format_date(expr: str, params: TransformationParams) -> str:
        date_format = params.format or "%Y-%m-%d"
        return f"DATE_FORMAT({expr}, '{date_format}')"

    @staticmethod
    def _truncate(expr: str, params: TransformationParams) -> str:
        length = params.length or 10
        return f"LEFT({expr}, {length})"

    @staticmethod
    def _scale(expr: str, params: TransformationParams) -> str:
        factor = params.factor or 1
        return f"({expr} * {factor})"

    @staticmethod
    def _normalize(expr: str, params: TransformationParams) -> str:
        return f"(({expr} - MIN({expr})) / (MAX({expr}) - MIN({expr})))"

    @staticmethod
    def _standardize(expr: str, params: TransformationParams) -> str:
        return f"(({expr} - AVG({expr})) / STDDEV({expr}))"

    @staticmethod
    def _convert_timezone(expr: str, params: TransformationParams) -> str:
        to_tz = params.to_tz or "UTC"
        from_tz = params.from_tz or "UTC"
        return f"CONVERT_TZ({expr}, '{from_tz}', '{to_tz}')"

    @staticmethod
    def _strip(expr: str, params: TransformationParams) -> str:
        return f"TRIM({expr})"

    @staticmethod
    def _to_numeric(expr: str, params: TransformationParams) -> str:
        return f"CAST({expr} AS DECIMAL)"

    @staticmethod
    def _to_datetime(expr: str, params: TransformationParams) -> str:
        format = params.format or "%Y-%m-%d"
        return f"STR_TO_DATE({expr}, '{format}')"

    @staticmethod
    def _replace(expr: str, params: TransformationParams) -> str:
        old_value = params.old_value
        new_value = params.new_value
        return f"REPLACE({expr}, '{old_value}', '{new_value}')"

    @staticmethod
    def _extract(expr: str, params: TransformationParams) -> str:
        pattern = params.pattern
        return f"REGEXP_SUBSTR({expr}, '{pattern}')"

    @staticmethod
    def _pad(expr: str, params: TransformationParams) -> str:
        width = params.width or 10
        side = params.side or "left"
        pad_char = params.pad_char or " "

        if side.lower() == "left":
            return f"LPAD({expr}, {width}, '{pad_char}')"
        return f"RPAD({expr}, {width}, '{pad_char}')"

    @staticmethod
    def _clip(expr: str, params: TransformationParams) -> str:
        lower = params.lower
        upper = params.upper
        return f"LEAST(GREATEST({expr}, {lower}), {upper})"

    @staticmethod
    def _bin(expr: str, params: TransformationParams) -> str:
        bins = params.bins
        labels = params.labels
        if not bins or not labels or len(bins) != len(labels) + 1:
            return expr

        case_stmt = "CASE "
        for i in range(len(labels)):
            case_stmt += (
                f"WHEN {expr} >= {bins[i]} AND {expr} < {bins[i+1]} THEN '{labels[i]}' "
            )
        case_stmt += f"ELSE {expr} END"

        return case_stmt

    @staticmethod
    def _validate_email(expr: str, params: TransformationParams) -> str:
        # Basic email validation pattern
        pattern = "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        return f"CASE WHEN {expr} REGEXP '{pattern}' THEN {expr} ELSE NULL END"

    @staticmethod
    def _validate_date_range(expr: str, params: TransformationParams) -> str:
        start_date = params.start_date
        end_date = params.end_date
        return f"CASE WHEN {expr} BETWEEN '{start_date}' AND '{end_date}' THEN {expr} ELSE NULL END"

    @staticmethod
    def _normalize_phone(expr: str, params: TransformationParams) -> str:
        country_code = params.country_code or "+1"
        return f"CONCAT('{country_code}', REGEXP_REPLACE({expr}, '[^0-9]', ''))"

    @staticmethod
    def _remove_duplicates(expr: str, params: TransformationParams) -> str:
        return f"DISTINCT {expr}"

    @staticmethod
    def _validate_foreign_key(expr: str, params: TransformationParams) -> str:
        ref_table = params.ref_table
        ref_column = params.ref_column
        return f"CASE WHEN {expr} IN (SELECT {ref_column} FROM {ref_table}) THEN {expr} ELSE NULL END"

    @staticmethod
    def _ensure_positive(expr: str, params: TransformationParams) -> str:
        return f"CASE WHEN {expr} > 0 THEN {expr} ELSE NULL END"

    @staticmethod
    def _standardize_categories(expr: str, params: TransformationParams) -> str:
        if not params.mapping:
            return expr

        case_stmt = (
            "CASE "
            + " ".join(
                f"WHEN LOWER({expr}) = LOWER('{key}') THEN '{value}'"
                for key, value in params.mapping.items()
            )
            + f" ELSE {expr} END"
        )

        return case_stmt

    @staticmethod
    def _rename(expr: str, params: TransformationParams) -> str:
        # Renaming is typically handled at the query level with AS
        new_name = params.new_name
        return f"{expr} AS {new_name}"

    @staticmethod
    def get_column_transformations(
        column_name: str, schema_transformations: List[Transformation]
    ) -> List[Transformation]:
        """Get all transformations that apply to a specific column.

        Args:
            column_name (str): Name of the column
            schema_transformations (List[Transformation]): List of all transformations in the schema

        Returns:
            List[Transformation]: List of transformations that apply to the column
        """
        return (
            [
                t
                for t in schema_transformations
                if t.params and t.params.column.lower() == column_name.lower()
            ]
            if schema_transformations
            else []
        )

    @staticmethod
    def apply_column_transformations(
        expr: str, column_name: str, schema_transformations: List[Transformation]
    ) -> str:
        """Apply all transformations for a specific column to an expression.

        Args:
            expr (str): The SQL expression to transform
            column_name (str): Name of the column
            schema_transformations (List[Transformation]): List of all transformations in the schema

        Returns:
            str: The transformed SQL expression
        """
        transformations = SQLTransformationManager.get_column_transformations(
            column_name, schema_transformations
        )
        return SQLTransformationManager.apply_transformations(expr, transformations)
