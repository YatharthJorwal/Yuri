"""
Config and router check -- no app window, no model calls.

    python orchestrator/check.py            # cached catalog, no network
    python orchestrator/check.py --refresh  # fetch /v1/models now
    python orchestrator/check.py --check    # same as no args

Prints the resolved chain for every role, whether the AI Router key resolved,
whether the model catalog is usable, and the spend against each cap. This is the
thing to run when "why is she not using the cloud" is the question.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

# Make `import config` / `import router` work no matter which cwd the user ran
# this from (repo root and orchestrator/ are both common).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import config  # noqa: E402
import router  # noqa: E402


async def _run(refresh: bool) -> int:
    cfg = config.CONFIG
    print(config.print_summary(cfg))
    print()

    rt = router.Router(cfg)
    cloud_cfg = cfg.providers.get("airouter")
    key = config.resolve_api_key(cloud_cfg.api_key_env) if cloud_cfg else None
    print(f"api key ({cloud_cfg.api_key_env if cloud_cfg else '-'}): {'resolved' if key else 'missing'}")

    if refresh:
        await rt.startup()
    else:
        rt.catalog.load_cache()
        rt.state.cloud_available = bool(
            cloud_cfg
            and cloud_cfg.enabled
            and key
            and rt.catalog.available
        )
    print(router.summarize(rt))

    if not cfg.routing.enforce_roles:
        print("\nnote: no role is enforced -- the router is recording decisions only (shadow).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Yuri's config and show router state.")
    parser.add_argument("--refresh", action="store_true", help="fetch the cloud model catalog now")
    parser.add_argument("--check", action="store_true", help="alias for the default behavior")
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.refresh))


if __name__ == "__main__":
    raise SystemExit(main())
