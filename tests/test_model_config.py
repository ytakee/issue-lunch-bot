from src.model_config import ModelSettings, resolve_model_settings


def test_resolve_model_settings_returns_defaults_when_missing(tmp_path):
    missing = tmp_path / "missing.toml"

    settings = resolve_model_settings("models/any.gguf", str(missing))

    assert settings == ModelSettings()


def test_resolve_model_settings_overrides_model(tmp_path):
    config = tmp_path / "models.toml"
    config.write_text(
        "\n".join(
            [
                "[defaults]",
                'chat_format = "chatml"',
                "supports_system = true",
                "",
                '[models."gemma-2-2b-it-Q4_K_M.gguf"]',
                'chat_format = "gemma"',
                "supports_system = false",
            ]
        ),
        encoding="utf-8",
    )

    settings = resolve_model_settings(
        "models/gemma-2-2b-it-Q4_K_M.gguf", str(config)
    )

    assert settings.supports_system is False
    assert settings.chat_format == "gemma"
