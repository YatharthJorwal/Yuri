"""Secret resolution: env var, then secrets.local.yaml, then .env.

The point of these tests is that a missing key returns None instead of a
placeholder that would 401 mid-conversation, and that a key in any of the three
places is found in the documented order.
"""

import config


def test_env_var_wins(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text("airouter:\n  api_key: from-file\n", encoding="utf-8")
    monkeypatch.setenv("AIROUTER_API_KEY", "from-env")
    assert config.resolve_api_key("AIROUTER_API_KEY", secrets_path=secrets) == "from-env"


def test_secrets_yaml_used_when_no_env(tmp_path, monkeypatch):
    monkeypatch.delenv("AIROUTER_API_KEY", raising=False)
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text("airouter:\n  api_key: from-file\n", encoding="utf-8")
    assert config.resolve_api_key("AIROUTER_API_KEY", secrets_path=secrets) == "from-file"


def test_flat_secrets_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("AIROUTER_API_KEY", raising=False)
    secrets = tmp_path / "secrets.yaml"
    secrets.write_text("AIROUTER_API_KEY: flat-key\n", encoding="utf-8")
    assert config.resolve_api_key("AIROUTER_API_KEY", secrets_path=secrets) == "flat-key"


def test_dotenv_used_last(tmp_path, monkeypatch):
    monkeypatch.delenv("AIROUTER_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("# comment\nAIROUTER_API_KEY=from-dotenv\n", encoding="utf-8")
    assert (
        config.resolve_api_key(
            "AIROUTER_API_KEY", secrets_path=tmp_path / "missing.yaml", dotenv_path=dotenv
        )
        == "from-dotenv"
    )


def test_dotenv_quotes_are_stripped(tmp_path, monkeypatch):
    monkeypatch.delenv("AIROUTER_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("AIROUTER_API_KEY='sk-quoted'\n", encoding="utf-8")
    assert (
        config.resolve_api_key(
            "AIROUTER_API_KEY", secrets_path=tmp_path / "missing.yaml", dotenv_path=dotenv
        )
        == "sk-quoted"
    )


def test_missing_everywhere_is_none(tmp_path, monkeypatch):
    monkeypatch.delenv("AIROUTER_API_KEY", raising=False)
    assert (
        config.resolve_api_key(
            "AIROUTER_API_KEY",
            secrets_path=tmp_path / "missing.yaml",
            dotenv_path=tmp_path / "missing.env",
        )
        is None
    )


def test_empty_env_value_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "   ")
    dotenv = tmp_path / ".env"
    dotenv.write_text("AIROUTER_API_KEY=real-key\n", encoding="utf-8")
    assert (
        config.resolve_api_key(
            "AIROUTER_API_KEY", secrets_path=tmp_path / "missing.yaml", dotenv_path=dotenv
        )
        == "real-key"
    )


def test_no_env_name_is_none(monkeypatch):
    monkeypatch.setenv("AIROUTER_API_KEY", "x")
    assert config.resolve_api_key("") is None


def test_environment_overrides_dotenv(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text("AIROUTER_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setenv("AIROUTER_API_KEY", "from-env")
    assert (
        config.resolve_api_key(
            "AIROUTER_API_KEY", secrets_path=tmp_path / "missing.yaml", dotenv_path=dotenv
        )
        == "from-env"
    )
