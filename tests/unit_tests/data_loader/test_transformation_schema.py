import pytest
from pydantic import ValidationError

from pandasai.data_loader.semantic_layer_schema import (
    Column,
    SemanticLayerSchema,
    Source,
    SQLConnectionConfig,
    Transformation,
    TransformationParams,
)


def test_basic_transformation_params():
    """Test basic transformation parameters validation"""
    params = TransformationParams(column="test_column", value=42)
    assert params.column == "test_column"
    assert params.value == 42


def test_transformation_params_value_types():
    """Test that value field accepts different types"""
    valid_values = [
        "string",  # str
        42,  # int
        3.14,  # float
        True,  # bool
    ]
    for value in valid_values:
        params = TransformationParams(value=value)
        assert params.value == value


def test_mapping_transformation():
    """Test mapping dictionary validation"""
    mapping = {
        "A": "Alpha",
        "B": "Beta",
        "C": "Charlie",
    }
    params = TransformationParams(column="test", mapping=mapping)
    assert params.mapping == mapping


def test_invalid_mapping_values():
    """Test that mapping only accepts string values"""
    with pytest.raises(ValidationError):
        TransformationParams(
            column="test",
            mapping={
                "A": 1,  # Should be string
                "B": True,  # Should be string
            },
        )


def test_optional_params_defaults():
    """Test default values for optional parameters"""
    params = TransformationParams()
    assert params.side == "left"
    assert params.pad_char == " "
    assert params.add_ellipsis is True
    assert params.drop_first is True
    assert params.drop_invalid is False
    assert params.country_code == "+1"
    assert params.keep == "first"


def test_numeric_params():
    """Test numeric parameters validation"""
    params = TransformationParams(
        column="test",
        factor=2.5,
        decimals=2,
        lower=0,
        upper=100,
        bins=[0, 25, 50, 75, 100],
    )
    assert params.factor == 2.5
    assert params.decimals == 2
    assert params.lower == 0
    assert params.upper == 100
    assert params.bins == [0, 25, 50, 75, 100]


def test_complete_transformation():
    """Test complete transformation with params"""
    transform = Transformation(
        type="map_values",
        params=TransformationParams(
            column="category",
            mapping={"A": "Alpha", "B": "Beta"},
        ),
    )
    assert transform.type == "map_values"
    assert transform.params.column == "category"
    assert transform.params.mapping == {"A": "Alpha", "B": "Beta"}


def test_schema_with_transformations():
    """Test schema with multiple transformations"""
    schema = SemanticLayerSchema(
        name="test_dataset",
        source={"type": "parquet", "path": "data.parquet", "table": "table"},
        transformations=[
            {
                "type": "fill_na",
                "params": {"column": "col1", "value": 0},
            },
            {
                "type": "map_values",
                "params": {
                    "column": "col2",
                    "mapping": {"Y": "Yes", "N": "No"},
                },
            },
        ],
    )
    assert len(schema.transformations) == 2
    assert schema.transformations[0].type == "fill_na"
    assert schema.transformations[0].params.value == 0
    assert schema.transformations[1].params.mapping == {"Y": "Yes", "N": "No"}


def test_invalid_transformation_type():
    """Test validation of transformation type"""
    with pytest.raises(ValidationError):
        Transformation(
            type="invalid_transform",
            params=TransformationParams(column="test"),
        )


def test_date_range_params():
    """Test date range validation parameters"""
    params = TransformationParams(
        column="date",
        start_date="2023-01-01",
        end_date="2023-12-31",
        drop_invalid=True,
    )
    assert params.start_date == "2023-01-01"
    assert params.end_date == "2023-12-31"
    assert params.drop_invalid is True


def test_complex_transformation_chain():
    """Test a complex chain of transformations in schema"""
    schema = SemanticLayerSchema(
        name="complex_dataset",
        source={"type": "parquet", "path": "data.parquet", "table": "table"},
        transformations=[
            {
                "type": "fill_na",
                "params": {"column": "numeric_col", "value": 0},
            },
            {
                "type": "map_values",
                "params": {
                    "column": "category_col",
                    "mapping": {"A": "Alpha", "B": "Beta"},
                },
            },
            {
                "type": "to_datetime",
                "params": {
                    "column": "date_col",
                    "format": "%Y-%m-%d",
                    "errors": "coerce",
                },
            },
            {
                "type": "clip",
                "params": {
                    "column": "value_col",
                    "lower": 0,
                    "upper": 100,
                },
            },
        ],
    )

    assert len(schema.transformations) == 4
    datetime_transform = schema.transformations[2]
    assert datetime_transform.type == "to_datetime"
    assert datetime_transform.params.format == "%Y-%m-%d"
    assert datetime_transform.params.errors == "coerce"

    clip_transform = schema.transformations[3]
    assert clip_transform.type == "clip"
    assert clip_transform.params.lower == 0
    assert clip_transform.params.upper == 100


def test_rename_transformation():
    """Test rename transformation validation"""
    schema = SemanticLayerSchema(
        name="test_dataset",
        source={"type": "parquet", "path": "data.parquet", "table": "table"},
        transformations=[
            {
                "type": "rename",
                "params": {
                    "column": "old_column",
                    "new_name": "new_column",
                },
            },
        ],
    )
    assert len(schema.transformations) == 1
    assert schema.transformations[0].type == "rename"
    assert schema.transformations[0].params.column == "old_column"
    assert schema.transformations[0].params.new_name == "new_column"


def test_rename_transformation_missing_params():
    """Test rename transformation requires both column and new_name"""
    with pytest.raises(ValueError):
        SemanticLayerSchema(
            name="test_dataset",
            source={"type": "parquet", "path": "data.parquet"},
            transformations=[
                {
                    "type": "rename",
                    "params": {
                        "column": "old_column",
                        # missing new_name
                    },
                },
            ],
        )


def test_column_expression_parse_error():
    with pytest.raises(ValueError):
        Column.is_expression_valid("invalid SELECT FROM sql")


def test_incompatible_source():
    source1 = Source(type="csv", path="path")
    source2 = Source(
        type="postgres",
        connection=SQLConnectionConfig(
            **{
                "host": "example.amazonaws.com",
                "port": 5432,
                "user": "user",
                "password": "password",
                "database": "db",
            }
        ),
        table="table",
    )
    assert not source1.is_compatible_source(source2)


def test_source_or_view_error():
    with pytest.raises(ValidationError):
        SemanticLayerSchema(name="ciao")


def test_column_must_be_defined_for_view():
    with pytest.raises(ValidationError):
        SemanticLayerSchema(name="ciao", view=True)
