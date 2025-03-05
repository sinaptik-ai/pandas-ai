import pydantic_core
import pytest
import sqlglot

from pandasai import SqlQueryBuilder
from pandasai.data_loader.semantic_layer_schema import (
    Column,
    SemanticLayerSchema,
    Source,
    SQLConnectionConfig,
    Transformation,
    TransformationParams,
)
from pandasai.query_builders.sql_transformation_manager import SQLTransformationManager


def validate_sql(sql: str) -> bool:
    """Validate if the SQL is syntactically correct using sqlglot"""
    try:
        sqlglot.parse_one(sql)
        return True
    except Exception:
        return False


def test_anonymize_transformation():
    expr = "user_email"
    transform = Transformation(type="anonymize", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "MD5(user_email)"
    assert validate_sql(result)


def test_fill_na_transformation():
    expr = "salary"
    transform = Transformation(type="fill_na", params=TransformationParams(value=0))
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "COALESCE(salary, 0)"
    assert validate_sql(result)


def test_map_values_transformation():
    expr = "status"
    mapping = {"A": "Active", "I": "Inactive"}
    transform = Transformation(
        type="map_values", params=TransformationParams(mapping=mapping)
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    expected = "CASE WHEN status = 'A' THEN 'Active' WHEN status = 'I' THEN 'Inactive' ELSE status END"
    assert result == expected
    assert validate_sql(result)


def test_to_lowercase_transformation():
    expr = "username"
    transform = Transformation(type="to_lowercase", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "LOWER(username)"
    assert validate_sql(result)


def test_round_numbers_transformation():
    expr = "price"
    transform = Transformation(
        type="round_numbers", params=TransformationParams(decimals=2)
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "ROUND(price, 2)"
    assert validate_sql(result)


def test_format_date_transformation():
    expr = "created_at"
    transform = Transformation(
        type="format_date", params=TransformationParams(format="%Y-%m-%d")
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "DATE_FORMAT(created_at, '%Y-%m-%d')"
    assert validate_sql(result)


def test_normalize_transformation():
    expr = "score"
    transform = Transformation(type="normalize", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "((score - MIN(score)) / (MAX(score) - MIN(score)))"
    assert validate_sql(result)


def test_multiple_transformations():
    expr = "user_data"
    transforms = [
        Transformation(type="to_lowercase", params=TransformationParams()),
        Transformation(type="truncate", params=TransformationParams(length=5)),
    ]
    result = SQLTransformationManager.apply_transformations(expr, transforms)
    assert result == "LEFT(LOWER(user_data), 5)"
    assert validate_sql(result)


def test_no_transformations():
    expr = "column_name"
    result = SQLTransformationManager.apply_transformations(expr, [])
    assert result == "column_name"
    assert validate_sql(result)


def test_invalid_transformation_type():
    with pytest.raises(pydantic_core._pydantic_core.ValidationError):
        Transformation(type="non_existent", params=TransformationParams())


def test_bin_transformation():
    expr = "age"
    bins = [0, 18, 35, 50, 100]
    labels = ["child", "young", "adult", "senior"]
    transform = Transformation(
        type="bin", params=TransformationParams(bins=bins, labels=labels)
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    expected = (
        "CASE WHEN age >= 0 AND age < 18 THEN 'child' "
        "WHEN age >= 18 AND age < 35 THEN 'young' "
        "WHEN age >= 35 AND age < 50 THEN 'adult' "
        "WHEN age >= 50 AND age < 100 THEN 'senior' "
        "ELSE age END"
    )
    assert result == expected
    assert validate_sql(result)


def test_clip_transformation():
    expr = "temperature"
    transform = Transformation(
        type="clip", params=TransformationParams(lower=0, upper=100)
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "LEAST(GREATEST(temperature, 0), 100)"
    assert validate_sql(result)


def test_to_uppercase_transformation():
    expr = "username"
    transform = Transformation(type="to_uppercase", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "UPPER(username)"
    assert validate_sql(result)


def test_truncate_transformation():
    expr = "description"
    transform = Transformation(type="truncate", params=TransformationParams(length=100))
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "LEFT(description, 100)"
    assert validate_sql(result)


def test_scale_transformation():
    expr = "temperature"
    transform = Transformation(type="scale", params=TransformationParams(factor=1.8))
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "(temperature * 1.8)"
    assert validate_sql(result)


def test_standardize_transformation():
    expr = "score"
    transform = Transformation(type="standardize", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "((score - AVG(score)) / STDDEV(score))"
    assert validate_sql(result)


def test_convert_timezone_transformation():
    expr = "event_time"
    transform = Transformation(
        type="convert_timezone",
        params=TransformationParams(from_tz="UTC", to_tz="America/New_York"),
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "CONVERT_TZ(event_time, 'UTC', 'America/New_York')"
    assert validate_sql(result)


def test_strip_transformation():
    expr = "text_field"
    transform = Transformation(type="strip", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "TRIM(text_field)"
    assert validate_sql(result)


def test_to_numeric_transformation():
    expr = "string_number"
    transform = Transformation(type="to_numeric", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "CAST(string_number AS DECIMAL)"
    assert validate_sql(result)


def test_to_datetime_transformation():
    expr = "date_string"
    transform = Transformation(
        type="to_datetime", params=TransformationParams(format="%Y-%m-%d %H:%i:%s")
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "STR_TO_DATE(date_string, '%Y-%m-%d %H:%i:%s')"
    assert validate_sql(result)


def test_replace_transformation():
    expr = "text"
    transform = Transformation(
        type="replace", params=TransformationParams(old_value="old", new_value="new")
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "REPLACE(text, 'old', 'new')"
    assert validate_sql(result)


def test_extract_transformation():
    expr = "text"
    transform = Transformation(
        type="extract", params=TransformationParams(pattern="[0-9]+")
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "REGEXP_SUBSTR(text, '[0-9]+')"
    assert validate_sql(result)


def test_pad_transformation():
    expr = "code"
    transform = Transformation(
        type="pad", params=TransformationParams(width=5, side="left", pad_char="0")
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "LPAD(code, 5, '0')"
    assert validate_sql(result)

    # Test right padding
    transform = Transformation(
        type="pad", params=TransformationParams(width=5, side="right", pad_char=" ")
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "RPAD(code, 5, ' ')"
    assert validate_sql(result)


def test_validate_email_transformation():
    expr = "email"
    transform = Transformation(type="validate_email", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert "REGEXP" in result and "email" in result
    assert validate_sql(result)


def test_validate_date_range_transformation():
    expr = "event_date"
    transform = Transformation(
        type="validate_date_range",
        params=TransformationParams(start_date="2023-01-01", end_date="2023-12-31"),
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert (
        result
        == "CASE WHEN event_date BETWEEN '2023-01-01' AND '2023-12-31' THEN event_date ELSE NULL END"
    )
    assert validate_sql(result)


def test_normalize_phone_transformation():
    expr = "phone"
    transform = Transformation(
        type="normalize_phone", params=TransformationParams(country_code="+44")
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "CONCAT('+44', REGEXP_REPLACE(phone, '[^0-9]', ''))"
    assert validate_sql(result)


def test_remove_duplicates_transformation():
    query_builder = SqlQueryBuilder(
        schema=SemanticLayerSchema(
            name="test_schema",
            source=Source(
                type="postgres",
                table="table_name",
                connection=SQLConnectionConfig(
                    host="-", port=8080, database="-", user="-", password="-"
                ),
            ),
            columns=[Column(name="value")],
            transformations=[Transformation(type="remove_duplicates")],
        )
    )
    head_query = query_builder.get_head_query()
    assert head_query == (
        'SELECT DISTINCT\n  "value" AS "value"\nFROM "table_name"\nLIMIT 5'
    )
    assert validate_sql(head_query)
    build_query = query_builder.build_query()
    assert build_query == 'SELECT DISTINCT\n  "value" AS "value"\nFROM "table_name"'
    assert validate_sql(build_query)


def test_validate_foreign_key_transformation():
    expr = "user_id"
    transform = Transformation(
        type="validate_foreign_key",
        params=TransformationParams(ref_table="users", ref_column="id"),
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert (
        result
        == "CASE WHEN user_id IN (SELECT id FROM users) THEN user_id ELSE NULL END"
    )
    assert validate_sql(result)


def test_ensure_positive_transformation():
    expr = "quantity"
    transform = Transformation(type="ensure_positive", params=TransformationParams())
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "CASE WHEN quantity > 0 THEN quantity ELSE NULL END"
    assert validate_sql(result)


def test_standardize_categories_transformation():
    expr = "category"
    mapping = {"cat": "Category", "prod": "Product"}
    transform = Transformation(
        type="standardize_categories", params=TransformationParams(mapping=mapping)
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    expected = "CASE WHEN LOWER(category) = LOWER('cat') THEN 'Category' WHEN LOWER(category) = LOWER('prod') THEN 'Product' ELSE category END"
    assert result == expected
    assert validate_sql(result)


def test_rename_transformation():
    expr = "old_name"
    transform = Transformation(
        type="rename", params=TransformationParams(new_name="new_name")
    )
    result = SQLTransformationManager.apply_transformations(expr, [transform])
    assert result == "old_name AS 'new_name'"
    assert validate_sql(result)
