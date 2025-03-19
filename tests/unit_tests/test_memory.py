from pandasai.helpers.memory import Memory


def test_to_json_empty_memory():
    memory = Memory()
    assert memory.to_json() == []


def test_to_json_with_messages():
    memory = Memory()

    # Add test messages
    memory.add("Hello", is_user=True)
    memory.add("Hi there!", is_user=False)
    memory.add("How are you?", is_user=True)

    expected_json = [
        {"role": "user", "message": "Hello"},
        {"role": "assistant", "message": "Hi there!"},
        {"role": "user", "message": "How are you?"},
    ]

    assert memory.to_json() == expected_json


def test_to_json_message_order():
    memory = Memory()

    # Add messages in specific order
    messages = [("Message 1", True), ("Message 2", False), ("Message 3", True)]

    for msg, is_user in messages:
        memory.add(msg, is_user=is_user)

    result = memory.to_json()

    # Verify order is preserved
    assert len(result) == 3
    assert result[0]["message"] == "Message 1"
    assert result[1]["message"] == "Message 2"
    assert result[2]["message"] == "Message 3"


def test_to_openai_messages_empty():
    memory = Memory()
    assert memory.to_openai_messages() == []


def test_to_openai_messages_with_agent_description():
    memory = Memory(agent_description="I am a helpful assistant")
    memory.add("Hello", is_user=True)
    memory.add("Hi there!", is_user=False)

    expected_messages = [
        {"role": "system", "content": "I am a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    assert memory.to_openai_messages() == expected_messages


def test_to_openai_messages_without_agent_description():
    memory = Memory()
    memory.add("Hello", is_user=True)
    memory.add("Hi there!", is_user=False)
    memory.add("How are you?", is_user=True)

    expected_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
    ]

    assert memory.to_openai_messages() == expected_messages
