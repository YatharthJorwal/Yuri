"""Runtime settings: the API-model switch, its persistence and its application."""

import dataclasses
import json

import config
import settings


class FakeRouter:
    def __init__(self):
        self.use_api_model_override = None

    def set_use_api_model(self, enabled):
        self.use_api_model_override = enabled


def test_no_file_means_no_choice_yet(tmp_path):
    assert settings.load(tmp_path / "settings.json") is None


def test_roundtrip(tmp_path):
    path = tmp_path / "settings.json"
    settings.save(settings.RuntimeSettings(use_api_model=True), path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"use_api_model": True}
    assert settings.load(path) == settings.RuntimeSettings(use_api_model=True)


def test_corrupt_file_is_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("not json", encoding="utf-8")
    assert settings.load(path) is None


def test_initial_falls_back_to_config_default(tmp_path):
    shadow = dataclasses.replace(config.CONFIG, routing=dataclasses.replace(
        config.CONFIG.routing, mode="shadow"
    ))
    assert settings.initial(shadow, tmp_path / "nothing.json") == settings.RuntimeSettings(
        use_api_model=False
    )


def test_initial_follows_enforce_config(tmp_path):
    enforced = dataclasses.replace(
        config.CONFIG,
        routing=dataclasses.replace(
            config.CONFIG.routing, mode="enforce", enforce_roles=("mind",)
        ),
    )
    assert settings.initial(enforced, tmp_path / "nothing.json").use_api_model is True


def test_saved_choice_beats_config_default(tmp_path):
    path = tmp_path / "settings.json"
    settings.save(settings.RuntimeSettings(use_api_model=False), path)
    enforced = dataclasses.replace(
        config.CONFIG,
        routing=dataclasses.replace(
            config.CONFIG.routing, mode="enforce", enforce_roles=("mind",)
        ),
    )
    assert settings.initial(enforced, path).use_api_model is False


def test_apply_sets_the_router_and_persists(tmp_path):
    path = tmp_path / "settings.json"
    router = FakeRouter()
    settings.apply(True, router, path=path)

    assert router.use_api_model_override is True
    assert settings.load(path).use_api_model is True


def test_describe_reports_unavailable_without_key(tmp_path):
    router = FakeRouter()
    router.state = type("State", (), {"cloud_available": False})()
    settings.apply(True, router, path=tmp_path / "settings.json")

    described = settings.describe(router, config.CONFIG)
    assert described["use_api_model"] is True
    assert described["cloud_available"] is False
    assert described["effective"] is False        # on, but it cannot actually run


def test_describe_reports_effective_when_available(tmp_path):
    router = FakeRouter()
    router.state = type("State", (), {"cloud_available": True})()
    settings.apply(True, router, path=tmp_path / "settings.json")

    described = settings.describe(router, config.CONFIG)
    assert described["effective"] is True
