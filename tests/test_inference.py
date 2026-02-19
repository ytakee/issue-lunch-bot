from unittest.mock import MagicMock

from src.inference import generate


def test_generate_returns_content():
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": "カレーライス！暑い日にはスパイスで元気を出しましょう。"
                }
            }
        ]
    }

    result = generate(
        mock_llm, "あなたはアドバイザーです", "今日は暑い", max_tokens=128
    )

    assert result == "カレーライス！暑い日にはスパイスで元気を出しましょう。"
    mock_llm.create_chat_completion.assert_called_once_with(
        messages=[
            {"role": "system", "content": "あなたはアドバイザーです"},
            {"role": "user", "content": "今日は暑い"},
        ],
        max_tokens=128,
    )


def test_generate_uses_default_max_tokens():
    mock_llm = MagicMock()
    mock_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "おすすめです"}}]
    }

    generate(mock_llm, "system", "user")

    call_kwargs = mock_llm.create_chat_completion.call_args
    assert call_kwargs.kwargs["max_tokens"] == 256
