from pandasai.core.response.string import StringResponse


def test_string_response_initialization():
    response = StringResponse("test value", "test_code")
    assert response.type == "string"
    assert response.value == "test value"
    assert response.last_code_executed == "test_code"


def test_string_response_minimal():
    response = StringResponse("")
    assert response.type == "string"
    assert response.value == ""
    assert response.last_code_executed is None


def test_string_response_with_non_string_value():
    response = StringResponse(123, "test_code")
    assert response.type == "string"
    assert response.value == 123
    assert response.last_code_executed == "test_code"


def test_string_response_format_alignment():
    """Test __format__ with string alignment"""
    response = StringResponse("hello", "test_code")
    assert f"{response:>10}" == "     hello"  # Right align
    assert f"{response:<10}" == "hello     "  # Left align
    assert f"{response:^10}" == "  hello   "  # Center align


def test_string_response_format_with_fstring():
    """Test __format__ in f-string context"""
    response = StringResponse("world", "test_code")
    result = f"Hello {response:>10}!"
    assert result == "Hello      world!"


def test_string_response_format_function():
    """Test __format__ with format() function"""
    response = StringResponse("test", "test_code")
    assert format(response, ">8") == "    test"


def test_string_response_format_truncate():
    """Test __format__ with truncation"""
    response = StringResponse("hello world", "test_code")
    assert f"{response:.5}" == "hello"


def test_string_response_format_with_str_format():
    """Test __format__ with string .format() method"""
    response = StringResponse("Python", "test_code")
    result = "Language: {:>10}".format(response)
    assert result == "Language:     Python"
