"""Model catalog discovery for the cloud provider.

`GET /v1/models` is public and returns context length, pricing, capability tags
and a `zdr` (zero-data-retention) flag per model. The router reads prices from
here instead of carrying a table that silently drifts.

Failure policy: use a cached catalog if one exists, otherwise the provider is
unavailable. Never guess a price, never invent a model id.
"""

from __future__ import annotations

import difflib
import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import httpx

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def repo_path(value: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class ModelInfo:
    id: str
    context_length: int = 0
    max_tokens: int = 0
    price_in: float = 0.0      # USD per 1M prompt tokens
    price_out: float = 0.0     # USD per 1M completion tokens
    tags: tuple[str, ...] = ()
    zdr: bool = False
    kind: str = "language"

    def supports(self, tag: str) -> bool:
        return tag in self.tags


class ModelCatalog:
    def __init__(
        self,
        base_url: str,
        cache_path: str | pathlib.Path,
        *,
        ttl_hours: float = 24.0,
        transport: httpx.AsyncBaseTransport | None = None,
        models_url: str | None = None,
        clock=time.time,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.models_url = (models_url or f"{self.base_url}/models").rstrip("/")
        self.cache_path = repo_path(cache_path)
        self.ttl_seconds = ttl_hours * 3600.0
        self._transport = transport
        self._clock = clock
        self._models: dict[str, ModelInfo] = {}
        self._fetched_at: float = 0.0
        self.last_error: str | None = None

    # -- state ---------------------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(self._models)

    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def info(self, model_id: str) -> ModelInfo | None:
        return self._models.get(model_id)

    def price(self, model_id: str) -> tuple[float, float]:
        info = self.info(model_id)
        return (info.price_in, info.price_out) if info else (0.0, 0.0)

    def context_length(self, model_id: str) -> int:
        info = self.info(model_id)
        return info.context_length if info else 0

    def closest(self, model_id: str, n: int = 3) -> list[str]:
        """Nearest ids, for the log line when a configured model vanished."""
        return difflib.get_close_matches(model_id, self.model_ids(), n=n, cutoff=0.4)

    def is_stale(self) -> bool:
        return (self._clock() - self._fetched_at) > self.ttl_seconds

    # -- persistence ---------------------------------------------------------

    def load_cache(self) -> bool:
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        models = raw.get("models") if isinstance(raw, dict) else None
        if not isinstance(models, dict) or not models:
            return False
        self._models = {str(k): _to_info(str(k), v) for k, v in models.items() if isinstance(v, dict)}
        self._fetched_at = _as_float(raw.get("fetched_at", 0.0))
        return bool(self._models)

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": self._fetched_at,
            "base_url": self.base_url,
            "models": {
                model_id: {
                    "context_length": info.context_length,
                    "max_tokens": info.max_tokens,
                    "pricing": {"prompt": info.price_in, "completion": info.price_out},
                    "tags": list(info.tags),
                    "zdr": info.zdr,
                    "type": info.kind,
                }
                for model_id, info in self._models.items()
            },
        }
        self.cache_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # -- network -------------------------------------------------------------

    async def refresh(self, *, force: bool = False) -> bool:
        """Load the catalog. Returns True when models are usable afterwards."""
        if not self._models:
            self.load_cache()
        if self._models and not force and not self.is_stale():
            return True

        url = self.models_url
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=15.0) as client:
                response = await client.get(url)
            if response.status_code >= 400:
                self.last_error = f"{response.status_code} from {url}"
                return self.available
            payload = response.json()
        except (httpx.RequestError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return self.available

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            self.last_error = "unexpected /models payload (no data list)"
            return self.available

        parsed: dict[str, ModelInfo] = {}
        for entry in data:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            model_id = str(entry["id"])
            parsed[model_id] = _to_info(model_id, entry)
        if not parsed:
            self.last_error = "catalog was empty"
            return self.available

        self._models = parsed
        self._fetched_at = self._clock()
        self.last_error = None
        try:
            self._save_cache()
        except OSError as exc:  # a read-only repo must not kill the app
            self.last_error = f"cache not written: {exc}"
        return True


def _to_info(model_id: str, entry: dict[str, Any]) -> ModelInfo:
    pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
    tags = entry.get("tags") if isinstance(entry.get("tags"), list) else []
    return ModelInfo(
        id=model_id,
        context_length=int(_as_float(entry.get("context_length"))),
        max_tokens=int(_as_float(entry.get("max_tokens"))),
        price_in=_as_float(pricing.get("prompt")),
        price_out=_as_float(pricing.get("completion")),
        tags=tuple(str(tag) for tag in tags),
        zdr=bool(entry.get("zdr", False)),
        kind=str(entry.get("type") or "language"),
    )
