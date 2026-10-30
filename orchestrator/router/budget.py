"""Spend accounting for cloud calls.

Prices come from the provider catalog (never a hardcoded table), caps are
checked *before* the call, and the ledger is SQLite so a restart does not reset
today's spending.

A cap of 0 means "never spend". A negative cap means unlimited. That sounds
backwards, but it reads correctly in a config file: `hourly_usd: 0` is how you
say "cloud off until I trust this".
"""

from __future__ import annotations

import pathlib
import sqlite3
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    ts                REAL NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          REAL NOT NULL,
    estimated         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS spend_ts ON spend(ts);
"""

HOUR = 3600.0
DAY = 86400.0
MONTH = 30 * DAY


def estimate_cost(
    prompt_tokens: int,
    max_completion_tokens: int,
    price_in_per_million: float,
    price_out_per_million: float,
) -> float:
    """Worst-case estimate for the cap check: the reply is assumed to run to its
    cap, so a call can never overshoot by more than one request's estimate."""
    return (
        prompt_tokens * price_in_per_million + max_completion_tokens * price_out_per_million
    ) / 1_000_000.0


def actual_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    price_in_per_million: float,
    price_out_per_million: float,
) -> float:
    return (
        (prompt_tokens or 0) * price_in_per_million
        + (completion_tokens or 0) * price_out_per_million
    ) / 1_000_000.0


class SpendLedger:
    def __init__(self, path: str | pathlib.Path) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def record(
        self,
        provider: str,
        model: str,
        cost_usd: float,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        estimated: bool = False,
        ts: float | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO spend (ts, provider, model, prompt_tokens, completion_tokens, cost_usd, estimated)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ts if ts is not None else time.time(),
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    float(cost_usd),
                    1 if estimated else 0,
                ),
            )

    def totals(self, now: float | None = None) -> tuple[float, float, float]:
        """(last hour, last day, last 30 days) in USD."""
        now = time.time() if now is None else now
        with self._connect() as conn:
            row = conn.execute(
                "SELECT"
                " COALESCE(SUM(CASE WHEN ts > ? THEN cost_usd END), 0),"
                " COALESCE(SUM(CASE WHEN ts > ? THEN cost_usd END), 0),"
                " COALESCE(SUM(CASE WHEN ts > ? THEN cost_usd END), 0)"
                " FROM spend",
                (now - HOUR, now - DAY, now - MONTH),
            ).fetchone()
        return float(row[0]), float(row[1]), float(row[2])
