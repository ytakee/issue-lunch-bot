import pytest
from unittest.mock import MagicMock, patch

from src.main import main


@patch("src.main.generate", return_value="ラーメンがおすすめです！")
@patch("src.main.load_model")
def test_main_prints_result(mock_load_model, mock_generate, capsys):
    mock_llm = MagicMock()
    mock_load_model.return_value = mock_llm

    test_args = [
        "main.py",
        "--model-path",
        "./models/test.gguf",
        "--system-prompt",
        "テスト用プロンプト",
        "--max-tokens",
        "128",
        "--n-ctx",
        "256",
        "--n-threads",
        "2",
        "--n-batch",
        "64",
    ]

    with patch("sys.argv", test_args), patch("sys.stdin") as mock_stdin:
        mock_stdin.read.return_value = "お腹すいた"
        main()

    mock_load_model.assert_called_once_with(
        "./models/test.gguf", n_ctx=256, n_threads=2, n_batch=64
    )
    mock_generate.assert_called_once_with(
        mock_llm, "テスト用プロンプト", "お腹すいた", 128
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == "ラーメンがおすすめです！"


def test_main_exits_on_empty_stdin():
    test_args = [
        "main.py",
        "--model-path",
        "./models/test.gguf",
        "--system-prompt",
        "テスト用プロンプト",
    ]

    with (
        patch("sys.argv", test_args),
        patch("sys.stdin") as mock_stdin,
        pytest.raises(SystemExit, match="1"),
    ):
        mock_stdin.read.return_value = ""
        main()


def test_main_exits_on_whitespace_only_stdin():
    test_args = [
        "main.py",
        "--model-path",
        "./models/test.gguf",
        "--system-prompt",
        "テスト用プロンプト",
    ]

    with (
        patch("sys.argv", test_args),
        patch("sys.stdin") as mock_stdin,
        pytest.raises(SystemExit, match="1"),
    ):
        mock_stdin.read.return_value = "  \n\t\n  "
        main()
