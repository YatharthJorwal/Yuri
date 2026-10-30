"""Router error types.

The local provider deliberately lets `llm.py`'s exceptions through untouched, so
every existing `except llm.LLMUnreachableError` keeps working. Only the cloud
path introduces new errors, and those subclass the same family so a caller that
already handles "model unreachable" does not need to learn a second hierarchy.
"""

from __future__ import annotations

import llm


class RouterError(llm.LLMUnreachableError):
    """Any router-level failure: no usable candidate, provider error, budget
    bookkeeping failure severe enough to stop the call."""


class ProviderError(RouterError):
    """A provider was reached but failed (HTTP error, malformed stream, 429)."""


class DiscoveryError(RouterError):
    """The model catalog could not be read and there was no usable cache."""
