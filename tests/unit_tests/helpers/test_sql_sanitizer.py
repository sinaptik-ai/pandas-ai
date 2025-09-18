from pandasai.helpers.sql_sanitizer import (
    is_sql_query,
    is_sql_query_safe,
    sanitize_file_name,
    sanitize_view_column_name,
    sanitize_sql_table_name_lowercase,
)


class TestSqlSanitizer:
    def test_sanitize_file_name_valid(self):
        filepath = "/path/to/valid_table.csv"
        expected = "valid_table"
        assert sanitize_file_name(filepath) == expected

    def test_sanitize_file_name_special_characters(self):
        filepath = "/path/to/invalid!@#.csv"
        expected = "invalid___"
        assert sanitize_file_name(filepath) == expected

    def test_sanitize_file_name_long_name(self):
        """Test with a filename exceeding the length limit."""
        filepath = "/path/to/" + "a" * 100 + ".csv"
        expected = "a" * 64
        assert sanitize_file_name(filepath) == expected

    def test_sanitize_table_name_lowercase(self):
        table_name = "My-Table.Name"
        expected = "my_table_name"
        assert sanitize_sql_table_name_lowercase(table_name) == expected

    def test_sanitize_relation_name_valid(self):
        relation = "dataset-name.column"
        expected = '"dataset_name"."column"'
        assert sanitize_view_column_name(relation) == expected

    def test_sanitize_relation_name_three_part(self):
        relation = "schema-name.dataset-name.column-name"
        expected = '"schema_name"."dataset_name"."column_name"'
        assert sanitize_view_column_name(relation) == expected

    def test_safe_select_query(self):
        query = "SELECT * FROM users WHERE username = 'admin';"
        assert is_sql_query_safe(query)

    def test_safe_with_query(self):
        query = "WITH user_data AS (SELECT * FROM users) SELECT * FROM user_data;"
        assert is_sql_query_safe(query)

    def test_unsafe_insert_query(self):
        query = "INSERT INTO users (username, password) VALUES ('admin', 'password');"
        assert not is_sql_query_safe(query)

    def test_unsafe_update_query(self):
        query = "UPDATE users SET password = 'newpassword' WHERE username = 'admin';"
        assert not is_sql_query_safe(query)

    def test_unsafe_delete_query(self):
        query = "DELETE FROM users WHERE username = 'admin';"
        assert not is_sql_query_safe(query)

    def test_unsafe_drop_query(self):
        query = "DROP TABLE users;"
        assert not is_sql_query_safe(query)

    def test_unsafe_alter_query(self):
        query = "ALTER TABLE users ADD COLUMN age INT;"
        assert not is_sql_query_safe(query)

    def test_unsafe_create_query(self):
        query = "CREATE TABLE users (id INT, username VARCHAR(50));"
        assert not is_sql_query_safe(query)

    def test_safe_select_with_comment(self):
        query = "SELECT * FROM users WHERE username = 'admin' -- comment"
        assert not is_sql_query_safe(query)

    def test_safe_select_with_inline_comment(self):
        query = "SELECT * FROM users /* inline comment */ WHERE username = 'admin';"
        assert not is_sql_query_safe(query)

    def test_unsafe_query_with_subquery(self):
        query = "SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);"
        assert is_sql_query_safe(query)

    def test_unsafe_query_with_subquery_insert(self):
        query = (
            "SELECT * FROM users WHERE id IN (INSERT INTO orders (user_id) VALUES (1));"
        )
        assert not is_sql_query_safe(query)

    def test_invalid_sql(self):
        query = "INVALID SQL QUERY"
        assert not is_sql_query_safe(query)

    def test_safe_query_with_multiple_keywords(self):
        query = "SELECT name FROM users WHERE username = 'admin' AND age > 30;"
        assert is_sql_query_safe(query)

    def test_safe_query_with_subquery(self):
        query = "SELECT name FROM users WHERE username IN (SELECT username FROM users WHERE age > 30);"
        assert is_sql_query_safe(
            query
        )

    def test_safe_query_with_query_params(self):
        query = "SELECT * FROM (SELECT * FROM heart_data) AS filtered_data LIMIT %s OFFSET %s"
        assert is_sql_query_safe(query)

    def test_plain_text(self):
        """Test with plain text input that is not a SQL query."""
        assert not is_sql_query("Hello, how are you?")
        assert not is_sql_query("This is just some text.")

    def test_sql_queries(self):
        """Test with typical SQL queries."""
        assert is_sql_query("SELECT * FROM users")
        assert is_sql_query("insert into users values ('john', 25)")
        assert is_sql_query("delete from orders where id=10")
        assert is_sql_query("DROP TABLE users")
        assert is_sql_query("update products set price=100 where id=1")

    def test_case_insensitivity(self):
        """Test with queries in different cases."""
        assert is_sql_query("select id from users")
        assert is_sql_query("SeLeCt id FROM users")
        assert is_sql_query("DROP table orders")
        assert is_sql_query("cReAtE DATABASE testdb")

    def test_edge_cases(self):
        """Test with edge cases like empty strings and special characters."""
        assert not is_sql_query("")
        assert not is_sql_query(" ")
        assert not is_sql_query("1234567890")
        assert not is_sql_query("#$%^&*()")
        assert not is_sql_query("JOIN the party")

    def test_mixed_input(self):
        """Test with mixed input containing SQL keywords in non-SQL contexts."""
        assert not is_sql_query("Let's SELECT a movie to watch")
        assert not is_sql_query("CREATE a new painting")
        assert not is_sql_query("DROP by my house later")
