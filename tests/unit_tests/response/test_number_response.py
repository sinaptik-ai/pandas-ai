from pandasai.core.response.number import NumberResponse


def test_number_response_initialization():
    response = NumberResponse(42, "test_code")
    assert response.type == "number"
    assert response.value == 42
    assert response.last_code_executed == "test_code"


def test_number_response_minimal():
    response = NumberResponse(0)
    assert response.type == "number"
    assert response.value == 0
    assert response.last_code_executed is None


def test_number_response_with_float():
    response = NumberResponse(3.14, "test_code")
    assert response.type == "number"
    assert response.value == 3.14
    assert response.last_code_executed == "test_code"


def test_number_response_with_string_number():
    response = NumberResponse("123", "test_code")
    assert response.type == "number"
    assert response.value == "123"


def test_number_response_string_formatting():
    """Test string formatting with f-strings for NumberResponse."""
    response = NumberResponse(3.14159)
    
    assert f"{response:.2f}" == "3.14"
    assert f"{response:.3f}" == "3.142"
    assert f"{response:+.2f}" == "+3.14"
    assert f"{response:>8.2f}" == "    3.14"
    assert f"{response:08.2f}" == "00003.14"


def test_number_response_string_formatting_with_integer():
    """Test string formatting with integer values."""
    response = NumberResponse(42)
    
    assert f"{response:04d}" == "0042"
    assert f"{response:+d}" == "+42"
    assert f"{response:>6d}" == "    42"
    assert f"{response:06d}" == "000042"


def test_number_response_string_formatting_with_negative():
    """Test string formatting with negative numbers."""
    response = NumberResponse(-3.14159)
    
    assert f"{response:.2f}" == "-3.14"
    assert f"{response:+.2f}" == "-3.14"
    assert f"{response:>8.2f}" == "   -3.14"


def test_number_response_string_formatting_with_large_number():
    """Test string formatting with large numbers."""
    response = NumberResponse(1234567.89)
    
    assert f"{response:,.2f}" == "1,234,567.89"
    assert f"{response:.0f}" == "1234568"
    assert f"{response:e}" == "1.234568e+06"


def test_base_response_string_formatting():
    """Test string formatting works for all BaseResponse subclasses."""
    from pandasai.core.response.string import StringResponse
    from pandasai.core.response.dataframe import DataFrameResponse
    
    string_response = StringResponse("hello")
    assert f"{string_response:>10}" == "     hello"
    assert f"{string_response:^10}" == "  hello   "
    
    df_response = DataFrameResponse({"a": [1, 2, 3]})
    formatted = f"{df_response}"
    assert isinstance(formatted, str)
