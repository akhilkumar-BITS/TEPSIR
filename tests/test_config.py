from app.core.config import Settings


def test_default_embedding_model_uses_new_env_name(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("SENTENCE_TRANSFORMER_MODEL", raising=False)
    monkeypatch.setenv("DEFAULT_EMBEDDING_MODEL", "custom/new-model")

    settings = Settings(_env_file=None)

    assert settings.default_embedding_model == "custom/new-model"


def test_old_embedding_model_env_name_is_still_accepted(monkeypatch):
    monkeypatch.delenv("DEFAULT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("SENTENCE_TRANSFORMER_MODEL", raising=False)
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/legacy-model")

    settings = Settings(_env_file=None)

    assert settings.default_embedding_model == "custom/legacy-model"


def test_sentence_transformer_model_alias_is_still_accepted(monkeypatch):
    monkeypatch.delenv("DEFAULT_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("SENTENCE_TRANSFORMER_MODEL", "custom/interim-model")

    settings = Settings(_env_file=None)

    assert settings.default_embedding_model == "custom/interim-model"
