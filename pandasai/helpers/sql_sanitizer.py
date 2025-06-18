import os
import re

import sqlglot
from sqlglot import parse_one
from sqlglot.optimizer.qualify_columns import quote_identifiers


def sanitize_view_column_name(relation_name: str) -> str:
    return (
        parse_one(
            ".".join(list(map(sanitize_sql_table_name, relation_name.split("."))))
        )
        .transform(quote_identifiers)
        .sql()
    )


def sanitize_sql_table_name(table_name: str) -> str:
    # Replace invalid characters with underscores
    sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)

    # Truncate to a reasonable length (e.g., 64 characters)
    max_length = 64
    sanitized_name = sanitized_name[:max_length]

    return sanitized_name


def sanitize_sql_table_name_lowercase(table_name: str) -> str:
    return sanitize_sql_table_name(table_name).lower()


def sanitize_file_name(filepath: str) -> str:
    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    return sanitize_sql_table_name(file_name).lower()


def is_sql_query_safe(query: str, dialect: str = "postgres") -> bool:
    try:
        # List of infected keywords to block (you can add more)
        infected_keywords = [
            r"\bINSERT\b",
            r"\bUPDATE\b",
            r"\bDELETE\b",
            r"\bDROP\b",
            r"\bEXEC\b",
            r"\bALTER\b",
            r"\bCREATE\b",
            r"\bMERGE\b",
            r"\bREPLACE\b",
            r"\bTRUNCATE\b",
            r"\bLOAD\b",
            r"\bGRANT\b",
            r"\bREVOKE\b",
            r"\bCALL\b",
            r"\bEXECUTE\b",
            r"\bSHOW\b",
            r"\bDESCRIBE\b",
            r"\bEXPLAIN\b",
            r"\bUSE\b",
            r"\bSET\b",
            r"\bDECLARE\b",
            r"\bOPEN\b",
            r"\bFETCH\b",
            r"\bCLOSE\b",
            r"\bSLEEP\b",
            r"\bBENCHMARK\b",
            r"\bDATABASE\b",
            r"\bUSER\b",
            r"\bCURRENT_USER\b",
            r"\bSESSION_USER\b",
            r"\bSYSTEM_USER\b",
            r"\bVERSION\b",
            r"\b@@VERSION\b",
            r"--",
            r"/\*.*\*/",  # Block comments and inline comments
        ]

        placeholder = "___PLACEHOLDER___"  # Temporary placeholder for params

        # Replace '%s' (MySQL, Psycopg2) with a unique placeholder
        temp_query = query.replace("%s", placeholder)

        # Parse the query to extract its structure
        parsed = sqlglot.parse_one(temp_query, dialect=dialect)

        # Ensure the main query is SELECT
        if parsed.key.upper() != "SELECT":
            return False

        # Check for infected keywords in the main query
        if any(
            re.search(keyword, query, re.IGNORECASE) for keyword in infected_keywords
        ):
            return False

        # Check for infected keywords in subqueries
        for subquery in parsed.find_all(sqlglot.exp.Subquery):
            subquery_sql = subquery.sql()  # Get the SQL of the subquery
            if any(
                re.search(keyword, subquery_sql, re.IGNORECASE)
                for keyword in infected_keywords
            ):
                return False

        return True

    except sqlglot.errors.ParseError:
        return False


def is_sql_query(query: str) -> bool:
    # Define SQL patterns with context to avoid standalone keyword matches
    sql_patterns = [
        r"\bSELECT\b.*\bFROM\b",
        r"\bINSERT\b.*\bINTO\b",
        r"\bUPDATE\b.*\bSET\b",
        r"\bDELETE\b.*\bFROM\b",
        r"\bDROP\b.*\b(TABLE|DATABASE)\b",
        r"\bCREATE\b.*\b(DATABASE|TABLE)\b",
        r"\bALTER\b.*\bTABLE\b",
        r"\bJOIN\b.*\bON\b",
        r"\bWHERE\b",
    ]

    # Combine all patterns into a single regex
    sql_regex = re.compile("|".join(sql_patterns), re.IGNORECASE)

    # If the query matches any SQL pattern, it's considered a SQL query
    if sql_regex.search(query):
        return True
    return False
