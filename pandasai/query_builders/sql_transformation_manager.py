from typing import Any, Dict, List, Optional, Union

from pandasai.data_loader.semantic_layer_schema import (
    Transformation,
    TransformationParams,
)


class SQLTransformationManager:
    """Manages SQL-based transformations for query expressions."""

    @staticmethod
    def _quote_str(value: str) -> str:
        """Quote and escape a string value for SQL."""
        if value is None:
            return "NULL"
        # Replace single quotes with double single quotes for SQL escaping
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def _validate_numeric(
        value: Union[int, float], param_name: str
    ) -> Union[int, float]:
        """Validate that a value is numeric."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Parameter {param_name} must be numeric, got {type(value)}"
                )
        return value

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
            else:
                raise ValueError(f"Unsupported transformation type: {method_name}")

        return transformed_expr

    @staticmethod
    def _anonymize(expr: str, params: TransformationParams) -> str:
        # Basic hashing for anonymization
        return f"MD5({expr})"

    @staticmethod
    def _fill_na(expr: str, params: TransformationParams) -> str:
        if isinstance(params.value, str):
            params.value = SQLTransformationManager._quote_str(params.value)
        else:
            params.value = SQLTransformationManager._validate_numeric(
                params.value, "value"
            )
        return f"COALESCE({expr}, {params.value})"

    @staticmethod
    def _map_values(expr: str, params: TransformationParams) -> str:
        if not params.mapping:
            return expr

        case_stmt = (
            "CASE "
            + " ".join(
                f"WHEN {expr} = {SQLTransformationManager._quote_str(key)} THEN {SQLTransformationManager._quote_str(value)}"
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
        decimals = SQLTransformationManager._validate_numeric(
            params.decimals or 0, "decimals"
        )
        return f"ROUND({expr}, {int(decimals)})"

    @staticmethod
    def _format_date(expr: str, params: TransformationParams) -> str:
        date_format = params.format or "%Y-%m-%d"
        return (
            f"DATE_FORMAT({expr}, {SQLTransformationManager._quote_str(date_format)})"
        )

    @staticmethod
    def _truncate(expr: str, params: TransformationParams) -> str:
        length = SQLTransformationManager._validate_numeric(
            params.length or 10, "length"
        )
        return f"LEFT({expr}, {int(length)})"

    @staticmethod
    def _scale(expr: str, params: TransformationParams) -> str:
        factor = SQLTransformationManager._validate_numeric(
            params.factor or 1, "factor"
        )
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
        return f"CONVERT_TZ({expr}, {SQLTransformationManager._quote_str(from_tz)}, {SQLTransformationManager._quote_str(to_tz)})"

    @staticmethod
    def _strip(expr: str, params: TransformationParams) -> str:
        return f"TRIM({expr})"

    @staticmethod
    def _to_numeric(expr: str, params: TransformationParams) -> str:
        return f"CAST({expr} AS DECIMAL)"

    @staticmethod
    def _to_datetime(expr: str, params: TransformationParams) -> str:
        _format = params.format or "%Y-%m-%d"
        _format = SQLTransformationManager._quote_str(_format)
        return f"STR_TO_DATE({expr}, {_format})"

    @staticmethod
    def _replace(expr: str, params: TransformationParams) -> str:
        old_value = params.old_value
        new_value = params.new_value
        return f"REPLACE({expr}, {SQLTransformationManager._quote_str(old_value)}, {SQLTransformationManager._quote_str(new_value)})"

    @staticmethod
    def _extract(expr: str, params: TransformationParams) -> str:
        pattern = params.pattern
        return f"REGEXP_SUBSTR({expr}, {SQLTransformationManager._quote_str(pattern)})"

    @staticmethod
    def _pad(expr: str, params: TransformationParams) -> str:
        width = SQLTransformationManager._validate_numeric(params.width or 10, "width")
        side = params.side or "left"
        pad_char = params.pad_char or " "

        if side.lower() == "left":
            return f"LPAD({expr}, {int(width)}, {SQLTransformationManager._quote_str(pad_char)})"
        return f"RPAD({expr}, {int(width)}, {SQLTransformationManager._quote_str(pad_char)})"

    @staticmethod
    def _clip(expr: str, params: TransformationParams) -> str:
        lower = SQLTransformationManager._validate_numeric(params.lower, "lower")
        upper = SQLTransformationManager._validate_numeric(params.upper, "upper")
        return f"LEAST(GREATEST({expr}, {lower}), {upper})"

    @staticmethod
    def _bin(expr: str, params: TransformationParams) -> str:
        bins = params.bins
        labels = params.labels
        if not bins or not labels or len(bins) != len(labels) + 1:
            raise ValueError(
                "Bins and labels lengths do not match the expected configuration."
            )

        # Validate all bin values are numeric
        bins = [
            SQLTransformationManager._validate_numeric(b, f"bins[{i}]")
            for i, b in enumerate(bins)
        ]

        case_stmt = "CASE "
        for i in range(len(labels)):
            case_stmt += f"WHEN {expr} >= {bins[i]} AND {expr} < {bins[i+1]} THEN {SQLTransformationManager._quote_str(labels[i])} "
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
        return f"CASE WHEN {expr} BETWEEN {SQLTransformationManager._quote_str(start_date)} AND {SQLTransformationManager._quote_str(end_date)} THEN {expr} ELSE NULL END"

    @staticmethod
    def _normalize_phone(expr: str, params: TransformationParams) -> str:
        country_code = params.country_code or "+1"
        return f"CONCAT({SQLTransformationManager._quote_str(country_code)}, REGEXP_REPLACE({expr}, '[^0-9]', ''))"

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
                f"WHEN LOWER({expr}) = LOWER({SQLTransformationManager._quote_str(key)}) THEN {SQLTransformationManager._quote_str(value)}"
                for key, value in params.mapping.items()
            )
            + f" ELSE {expr} END"
        )

        return case_stmt

    @staticmethod
    def _rename(expr: str, params: TransformationParams) -> str:
        # Renaming is typically handled at the query level with AS
        new_name = SQLTransformationManager._quote_str(params.new_name)
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
