import pytest
from pydantic import ValidationError

from pandasai.data_loader.semantic_layer_schema import (
    Destination,
    SemanticLayerSchema,
    Transformation,
    is_schema_source_same,
)


class TestSemanticLayerSchema:
    def test_valid_schema(self, raw_sample_schema):
        schema = SemanticLayerSchema(**raw_sample_schema)

        assert schema.name == "users"
        assert schema.update_frequency == "weekly"
        assert len(schema.columns) == 3
        assert schema.order_by == ["created_at DESC"]
        assert schema.limit == 100
        assert schema.source.type == "csv"

    def test_valid_raw_mysql_schema(self, raw_mysql_schema):
        schema = SemanticLayerSchema(**raw_mysql_schema)

        assert schema.name == "users"
        assert schema.update_frequency == "weekly"
        assert len(schema.columns) == 3
        assert schema.order_by == ["created_at DESC"]
        assert schema.limit == 100
        assert schema.source.type == "mysql"

    def test_valid_raw_mysql_view_schema(self, raw_mysql_view_schema):
        schema = SemanticLayerSchema(**raw_mysql_view_schema)

        assert schema.name == "parent_children"
        assert len(schema.columns) == 3
        assert schema.view == True

    def test_invalid_name(self, raw_sample_schema):
        raw_sample_schema["name"] = "invalid-name"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_missing_source_path(self, raw_sample_schema):
        raw_sample_schema["source"].pop("path")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_missing_source_table(self, raw_mysql_schema):
        raw_mysql_schema["source"].pop("table")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_schema)

    def test_missing_mysql_connection(self, raw_mysql_schema):
        raw_mysql_schema["source"].pop("connection")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_schema)

    def test_invalid_schema_missing_name(self, raw_sample_schema):
        raw_sample_schema.pop("name")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_invalid_column_type(self, raw_sample_schema):
        raw_sample_schema["columns"][0]["type"] = "unsupported"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_invalid_source_type(self, raw_sample_schema):
        raw_sample_schema["source"]["type"] = "invalid"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_valid_transformations(self):
        transformation_data = {
            "type": "anonymize",
            "params": {"column": "email"},
        }

        transformation = Transformation(**transformation_data)

        assert transformation.type == "anonymize"
        assert transformation.params.column == "email"

    def test_valid_destination(self):
        destination_data = {
            "type": "local",
            "format": "parquet",
            "path": "output.parquet",
        }

        destination = Destination(**destination_data)

        assert destination.type == "local"
        assert destination.format == "parquet"
        assert destination.path == "output.parquet"

    def test_invalid_destination_format(self):
        destination_data = {
            "type": "local",
            "format": "invalid",
            "path": "output.parquet",
        }

        with pytest.raises(ValidationError):
            Destination(**destination_data)

    def test_invalid_transformation_type(self):
        transformation_data = {
            "type": "unsupported_transformation",
            "params": {"column": "email"},
        }

        with pytest.raises(ValidationError):
            Transformation(**transformation_data)

    def test_is_schema_source_same_true(self, raw_mysql_schema):
        schema1 = SemanticLayerSchema(**raw_mysql_schema)
        schema2 = SemanticLayerSchema(**raw_mysql_schema)

        assert is_schema_source_same(schema1, schema2) is True

    def test_is_schema_source_same_false(self, raw_mysql_schema, raw_sample_schema):
        schema1 = SemanticLayerSchema(**raw_mysql_schema)
        schema2 = SemanticLayerSchema(**raw_sample_schema)

        assert is_schema_source_same(schema1, schema2) is False

    def test_invalid_view_and_source(self, raw_mysql_schema):
        raw_mysql_schema["view"] = True

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_schema)

    def test_invalid_source_missing_view_or_table(self, raw_mysql_schema):
        raw_mysql_schema["source"].pop("table")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_schema)

    def test_invalid_no_relation_for_view(self, raw_mysql_view_schema):
        raw_mysql_view_schema.pop("relations")

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_view_schema)

    def test_invalid_duplicated_columns(self, raw_sample_schema):
        raw_sample_schema["columns"].append(raw_sample_schema["columns"][0])

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_invalid_wrong_column_format_in_view(self, raw_mysql_view_schema):
        raw_mysql_view_schema["columns"][0]["name"] = "parentsid"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_view_schema)

    def test_invalid_wrong_column_format(self, raw_sample_schema):
        raw_sample_schema["columns"][0]["name"] = "parents.id"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_sample_schema)

    def test_invalid_wrong_relation_format_in_view(self, raw_mysql_view_schema):
        raw_mysql_view_schema["relations"][0]["to"] = "parentsid"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_view_schema)

    def test_invalid_uncovered_columns_in_view(self, raw_mysql_view_schema):
        raw_mysql_view_schema["relations"][0]["to"] = "parents.id"

        with pytest.raises(ValidationError):
            SemanticLayerSchema(**raw_mysql_view_schema)
