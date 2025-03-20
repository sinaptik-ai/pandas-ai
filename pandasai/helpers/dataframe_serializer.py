import json
import typing

if typing.TYPE_CHECKING:
    from ..dataframe.base import DataFrame


class DataframeSerializer:
    MAX_COLUMN_TEXT_LENGTH = 200

    @classmethod
    def serialize(cls, df: "DataFrame", dialect: str = "postgres") -> str:
        """
        Convert df to a CSV-like format wrapped inside <table> tags, truncating long text values.

        Args:
            df (pd.DataFrame): Pandas DataFrame
            dialect (str): Database dialect (default is "postgres")

        Returns:
            str: Serialized DataFrame string
        """

        # Start building the table metadata
        dataframe_info = f'<table dialect="{dialect}" table_name="{getattr(df.schema, "name", "unknown")}"'

        # Add description attribute if available
        description = getattr(df.schema, "description", None)
        if description:
            dataframe_info += f' description="{description}"'

        dataframe_info += f' dimensions="{getattr(df, "rows_count", len(df))}x{getattr(df, "columns_count", len(df.columns))}">\n'

        # Truncate long values
        df_truncated = cls._truncate_dataframe(df.head())

        # Convert to CSV format
        dataframe_info += df_truncated.to_csv(index=False)

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
                return f"{value[: cls.MAX_COLUMN_TEXT_LENGTH]} …"
            return value

        return df.applymap(truncate_value)
