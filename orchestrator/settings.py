"""
Runtime settings: the switches a user flips while the app is running.

These are deliberately *not* config defaults. `config/config.yaml` is the shape of
the machine; this file is the current mood of the user. The one setting today is
whether cloud (API) models may be used for the `mind` role.

Persisted as JSON under `data/` (gitignored) so a toggle survives a restart. If
the file is absent, the initial value follows the config: `routing.mode:
enforce` with `mind` in `enforce_roles` means on, otherwise off.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

from config import CONFIG, Config
from router.discovery import repo_path

DEFAULT_PATH = repo_path("data/settings.json")


@dataclass(frozen=True)
class RuntimeSettings:
    # Explicit user choice from the tray/quick-action toggle. False is the safe
    # default; True still obeys every privacy and budget rule in the router.
    use_api_model: bool = False


def config_default(config: Config = CONFIG) -> bool:
    """What the toggle should read before the user has ever touched it."""
    return config.routing.mode == "enforce" and any(
        role for role in config.routing.enforce_roles if role in config.roles
    )


def load(path: str | pathlib.Path | None = None) -> RuntimeSettings | None:
    """The saved choice, or None when the user has never toggled it."""
    file = pathlib.Path(path) if path else DEFAULT_PATH
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return RuntimeSettings(use_api_model=bool(raw.get("use_api_model", False)))


def save(settings: RuntimeSettings, path: str | pathlib.Path | None = None) -> None:
    file = pathlib.Path(path) if path else DEFAULT_PATH
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(asdict(settings), indent=1), encoding="utf-8")


def initial(config: Config = CONFIG, path: str | pathlib.Path | None = None) -> RuntimeSettings:
    """The effective setting at startup: saved choice, else the config default."""
    stored = load(path)
    return stored if stored is not None else RuntimeSettings(config_default(config))


def apply(enabled: bool, router, *, path: str | pathlib.Path | None = None) -> RuntimeSettings:
    """Flip the switch live and persist it. Returns the settings that were saved."""
    settings = RuntimeSettings(use_api_model=enabled)
    router.set_use_api_model(enabled)
    save(settings, path)
    return settings


def describe(router, config: Config = CONFIG) -> dict:
    """What the shell shows about this switch."""
    cloud_available = bool(getattr(router.state, "cloud_available", False))
    return {
        "use_api_model": bool(router.use_api_model_override),
        # A user can turn it on with no key; the toggle then says "no key"
        # instead of silently doing nothing.
        "cloud_available": cloud_available,
        "effective": bool(router.use_api_model_override) and cloud_available,
        "mode": config.routing.mode,
    }
