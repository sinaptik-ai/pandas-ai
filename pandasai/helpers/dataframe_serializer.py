import json
import typing

if typing.TYPE_CHECKING:
    from ..dataframe.base import DataFrame


class DataframeSerializer:
    MAX_COLUMN_TEXT_LENGTH = 200

    @classmethod
    def serialize(cls, df: "DataFrame", dialect: str = "postgres") -> str:
        """
        Convert df to a CSV-like format wrapped inside <table> tags, truncating long text values, and serializing only a subset of rows using df.head().

        Args:
            df (pd.DataFrame): Pandas DataFrame
            dialect (str): Database dialect (default is "postgres")

        Returns:
            str: Serialized DataFrame string
        """

        # Start building the table metadata
        dataframe_info = f'<table dialect="{dialect}" table_name="{df.schema.name}"'

        # Add description attribute if available
        if df.schema.description is not None:
            dataframe_info += f' description="{df.schema.description}"'

        if df.schema.columns:
            columns = [column.model_dump() for column in df.schema.columns]
            dataframe_info += f' columns="{json.dumps(columns, ensure_ascii=False)}"'

        dataframe_info += f' dimensions="{df.rows_count}x{df.columns_count}">'

        # Truncate long values
        df_truncated = cls._truncate_dataframe(df.head())

        # Convert to CSV format
        dataframe_info += f"\n{df_truncated.to_csv(index=False)}"

        # Close the table tag
        dataframe_info += "</table>\n"

        return dataframe_info

    @classmethod
    def _truncate_dataframe(cls, df: "DataFrame") -> "DataFrame":
        """Truncates string values exceeding MAX_COLUMN_TEXT_LENGTH, and converts JSON-like values to truncated strings."""

        def truncate_value(value):
            if isinstance(value, (dict, list)):  # Convert JSON-like objects to strings
                value = json.dumps(value, ensure_ascii=False)

            if isinstance(value, str) and len(value) > cls.MAX_COLUMN_TEXT_LENGTH:
                return f"{value[: cls.MAX_COLUMN_TEXT_LENGTH]}…"
            return value

        return df.apply(lambda row: row.apply(truncate_value), axis=1)
