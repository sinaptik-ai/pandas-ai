import datetime
import json

import pytest
from pydantic import ValidationError

from pandasai.query_builders.paginator import DatasetPaginator, PaginationParams


class TestPaginationParams:
    def test_valid_pagination_params(self):
        """Test creating PaginationParams with valid data"""
        params = PaginationParams(
            page=1,
            page_size=10,
            search="test",
            sort_by="name",
            sort_order="asc",
            filters=json.dumps({"status": ["active", "pending"]}),
        )
        assert params.page == 1
        assert params.page_size == 10
        assert params.search == "test"
        assert params.sort_by == "name"
        assert params.sort_order == "asc"
        assert json.loads(params.filters) == {"status": ["active", "pending"]}

    def test_invalid_page_number(self):
        """Test validation error for invalid page number"""
        with pytest.raises(ValidationError) as exc_info:
            PaginationParams(page=0, page_size=10)
        assert "Input should be greater than or equal to 1" in str(exc_info.value)

    def test_invalid_page_size(self):
        """Test validation error for invalid page size"""
        with pytest.raises(ValidationError) as exc_info:
            PaginationParams(page=1, page_size=101)
        assert "Input should be less than or equal to 100" in str(exc_info.value)

    def test_invalid_sort_order(self):
        """Test validation error for invalid sort order"""
        with pytest.raises(ValidationError) as exc_info:
            PaginationParams(page=1, page_size=10, sort_by="name", sort_order="invalid")
        assert "String should match pattern" in str(exc_info.value)

    def test_sql_injection_prevention(self):
        """Test that SQL injection attempts are caught"""
        with pytest.raises(ValueError) as exc_info:
            PaginationParams(page=1, page_size=10, search="SELECT * FROM users")
        assert "SQL queries are not allowed" in str(exc_info.value)


class TestDatasetPaginator:
    @pytest.fixture
    def sample_query(self):
        return "SELECT id, name, age FROM users"

    @pytest.fixture
    def sample_columns(self):
        return [
            {"name": "id", "type": "integer"},
            {"name": "name", "type": "string"},
            {"name": "age", "type": "integer"},
            {"name": "created_at", "type": "datetime"},
            {"name": "is_active", "type": "boolean"},
            {"name": "score", "type": "float"},
            {"name": "user_id", "type": "uuid"},
        ]

    def test_basic_pagination(self, sample_query, sample_columns):
        """Test basic pagination without search or filters"""
        params = PaginationParams(page=2, page_size=10)
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert "LIMIT %s OFFSET %s" in query
        assert parameters == [10, 10]  # page_size and offset

    def test_search_string_column(self, sample_query, sample_columns):
        """Test search on string column"""
        params = PaginationParams(page=1, page_size=10, search="John")
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert '"name" ILIKE %s' in query
        assert parameters[0] == "%John%"  # First parameter is search term
        assert len(parameters) == 3  # search + LIMIT/OFFSET

    def test_search_numeric_columns(self, sample_query, sample_columns):
        """Test search on numeric columns"""
        params = PaginationParams(page=1, page_size=10, search="25")
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert '"id" = %s' in query
        assert '"age" = %s' in query
        assert parameters.count("25") >= 2  # At least id and age columns
        assert len(parameters) > 2  # search params + LIMIT/OFFSET

    def test_search_datetime(self, sample_query, sample_columns):
        """Test search on datetime column"""
        params = PaginationParams(page=1, page_size=10, search="2023-01-01 12:00:00")
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert '"created_at" = %s' in query
        # Convert the datetime string to expected format
        expected_dt = datetime.datetime.strptime(
            "2023-01-01 12:00:00", "%Y-%m-%d %H:%M:%S"
        )
        assert any(
            isinstance(p, datetime.datetime) and p == expected_dt for p in parameters
        )

    def test_filters(self, sample_query, sample_columns):
        """Test filtering with IN clause"""
        params = PaginationParams(
            page=1, page_size=10, filters=json.dumps({"age": [25, 30, 35]})
        )
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert '"age" IN (%s, %s, %s)' in query
        assert all(
            x in parameters for x in [25, 30, 35]
        )  # Filter values are in parameters
        assert len(parameters) == 5  # 3 filter values + LIMIT/OFFSET

    def test_sorting(self, sample_query, sample_columns):
        """Test sorting functionality"""
        params = PaginationParams(
            page=1, page_size=10, sort_by="age", sort_order="desc"
        )
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert 'ORDER BY "age" DESC' in query

    def test_invalid_sort_column(self, sample_query, sample_columns):
        """Test error on invalid sort column"""
        params = PaginationParams(
            page=1, page_size=10, sort_by="invalid_column", sort_order="asc"
        )
        with pytest.raises(ValueError) as exc_info:
            DatasetPaginator.apply_pagination(sample_query, sample_columns, params)
        assert "not found in available columns" in str(exc_info.value)

    def test_type_validation_methods(self):
        """Test the type validation helper methods"""
        # Test float validation
        assert DatasetPaginator.is_float("123.45")
        assert not DatasetPaginator.is_float("abc")

        # Test boolean validation
        assert DatasetPaginator.is_valid_boolean("true")
        assert DatasetPaginator.is_valid_boolean("false")
        assert not DatasetPaginator.is_valid_boolean("invalid")

        # Test datetime validation
        assert DatasetPaginator.is_valid_datetime("2023-01-01 12:00:00")
        assert not DatasetPaginator.is_valid_datetime("invalid-date")

        # Test UUID validation
        assert DatasetPaginator.is_valid_uuid("123e4567-e89b-12d3-a456-426614174000")
        assert not DatasetPaginator.is_valid_uuid("invalid-uuid")
        try:
            DatasetPaginator.is_valid_uuid(None)
            assert False, "Should raise TypeError"
        except (ValueError, TypeError):
            pass

    def test_no_pagination(self, sample_query, sample_columns):
        """Test that query is returned as-is when pagination is None"""
        query, params = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, None
        )
        assert query == sample_query
        assert params == []

    def test_boolean_search(self, sample_query, sample_columns):
        """Test search on boolean column"""
        params = PaginationParams(page=1, page_size=10, search="true")
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert '"is_active" = %s' in query
        assert "true" in [str(p).lower() for p in parameters]

    def test_uuid_search(self, sample_query, sample_columns):
        """Test search on UUID column"""
        uuid_value = "123e4567-e89b-12d3-a456-426614174000"
        params = PaginationParams(page=1, page_size=10, search=uuid_value)
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert '"user_id"::TEXT = %s' in query
        assert uuid_value in parameters

    def test_filter_single_value(self, sample_query, sample_columns):
        """Test filtering with a single value instead of a list"""
        params = PaginationParams(
            page=1,
            page_size=10,
            filters=json.dumps({"age": 25}),  # Single value instead of list
        )
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )
        assert '"age" IN (%s)' in query
        assert 25 in parameters

    def test_invalid_json_filter(self, sample_query, sample_columns):
        """Test error handling for invalid JSON in filters"""
        params = PaginationParams(page=1, page_size=10, filters="{invalid json")
        with pytest.raises(ValueError) as exc_info:
            DatasetPaginator.apply_pagination(sample_query, sample_columns, params)
        assert "Invalid filters format" in str(exc_info.value)

    def test_combined_functionality(self, sample_query, sample_columns):
        """Test combining multiple pagination features"""
        params = PaginationParams(
            page=2,
            page_size=10,
            search="John",
            sort_by="age",
            sort_order="desc",
            filters=json.dumps({"is_active": [True]}),
        )
        query, parameters = DatasetPaginator.apply_pagination(
            sample_query, sample_columns, params
        )

        # Check all components are present
        assert "WHERE" in query
        assert "ORDER BY" in query
        assert "LIMIT" in query
        assert "OFFSET" in query

        # Check parameters
        assert len(parameters) == 4  # search param + filter value + LIMIT/OFFSET
        assert parameters[0] == "%John%"  # First parameter is search
        assert True in parameters  # Filter value
        assert 10 in parameters  # page_size
        assert parameters[-1] == 10  # offset for page 2
