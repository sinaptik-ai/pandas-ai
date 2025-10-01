from pandasai.core.response.number import NumberResponse


def test_number_response_initialization():
    response = NumberResponse(42, "test_code")
    assert response.type == "number"
    assert response.value == 42
    assert response.last_code_executed == "test_code"


def test_number_response_minimal():
    response = NumberResponse(0)  # Zero instead of None
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
    assert response.value == "123"  # Value remains as string


def test_number_response_format_decimal():
    """Test __format__ with decimal places"""
    response = NumberResponse(3.14159, "test_code")
    assert f"{response:.2f}" == "3.14"
    assert f"{response:.4f}" == "3.1416"


def test_number_response_format_with_fstring():
    """Test __format__ in f-string context"""
    response = NumberResponse(123.456, "test_code")
    result = f"Value: {response:.2f}"
    assert result == "Value: 123.46"


def test_number_response_format_function():
    """Test __format__ with format() function"""
    response = NumberResponse(42.123, "test_code")
    assert format(response, ".1f") == "42.1"


def test_number_response_format_scientific():
    """Test __format__ with scientific notation"""
    response = NumberResponse(1234.5, "test_code")
    assert f"{response:e}" == "1.234500e+03"


def test_number_response_format_percentage():
    """Test __format__ with percentage"""
    response = NumberResponse(0.875, "test_code")
    assert f"{response:.1%}" == "87.5%"


def test_number_response_format_padding():
    """Test __format__ with padding"""
    response = NumberResponse(42, "test_code")
    assert f"{response:05d}" == "00042"
    assert f"{response:>10}" == "        42"


def test_number_response_format_integer():
    """Test __format__ with integer formatting"""
    response = NumberResponse(42, "test_code")
    assert f"{response:d}" == "42"


def test_number_response_format_with_str_format():
    """Test __format__ with string .format() method"""
    response = NumberResponse(99.9, "test_code")
    result = "Price: ${:.2f}".format(response)
    assert result == "Price: $99.90"
