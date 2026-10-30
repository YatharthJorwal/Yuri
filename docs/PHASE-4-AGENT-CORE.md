# Phase 4 — Agent core (tools, tiers, Mind jobs)

The phase where Yuri stops narrating and starts doing, without becoming the
thing that deletes a folder because a web page told it to.

Today `orchestrator/tools/` has four tools, a schema list, and a dispatch
function. `app.py` runs a bounded tool loop around one turn. There is no tier,
no confirmation, no audit, and no way to run a multi-step job in the background.

---

## 0. Goal

1. Every tool is a registry entry with a schema, a tier, and an audit shape.
2. The effective tier is decided at call time from the *context*, not just the
   tool (untrusted content raises it; Tier 4 cannot be reached at all).
3. Tier 2 requires a real confirmation bound to the exact action.
4. Mind can run a job of several steps, cancellable, each step audited.
5. One Tier 3 channel: reach the owner. One recipient, off by default.

Dependencies: P1 (router, for Mind's model calls), P3 (memory tools use the
store). Does not depend on P2.

---

## 1. Tool registry

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict                    # JSON Schema for arguments
    tier: int                       # declared tier, 0..4
    scopes: tuple[str, ...]         # "screen", "files:<dir>", "network", "message:<channel>"
    timeout_s: float = 30.0
    idempotent: bool = False
    audit_fields: tuple[str, ...] = ()   # argument keys safe to log
    handler: Callable[..., Awaitable[str]]
```

`registry.py` holds the specs and exposes `get(name)`, `offered(role, context)`,
and `dispatch(name, args, ctx)`. `offered()` is where Luna's hard-won behavior
lives: after the look gate has already captured a screen, the look tools are
withheld this turn; a model cannot start a task; a tool whose result is already
in context is not re-offered.

Tiers (from `SPEC-v0.1.md` §4, made executable):

| Tier | Meaning | Examples |
| --- | --- | --- |
| 0 | observe | `capture_screen`, `read_clipboard`, `memory_search`, `read_file` (allowed dirs) |
| 1 | draft | `draft_message`, `propose_edit`, `draft_command` (returns text, executes nothing) |
| 2 | ask first | `write_file`, `run_command`, `open_app`, `send_message` (not Tier 3) |
| 3 | autonomous allowlist | `ping_owner` when promoted, `write_file` inside a promoted folder |
| 4 | never | delete outside sandbox, spend money, credentials, third-party messages |

### Effective tier

```
effective = max(spec.tier, context_floor)
context_floor = 2 if turn.trust == untrusted        # injected content in play
              = 1 if turn.came_from_perception        # screen/file text influenced it
              = 0 otherwise
effective = 4 if spec.name in HARD_DENY               # code list, unreachable
effective = min(effective, config_grant)              # a grant can only lower to 2
```

Rule that matters: **untrusted content can never produce a Tier 0/1 auto-run.**
It raises everything to "confirm" or blocks it. This is the injection defense
that a prompt instruction cannot provide.

### New tools

| Tool | Tier | Notes |
| --- | --- | --- |
| `read_file(path)` | 0 | only under `tools.allowed_dirs`; path resolved, symlinks refused |
| `write_file(path, text)` | 2 | confirmed; refuses outside allowed dirs; writes via temp + rename |
| `run_command(cmd, cwd)` | 2 | confirmed, allowlisted binaries, no shell interpolation, captured output, timeout |
| `open_app(name)` | 2 | confirmed; resolves against a known-app map, never a raw command |
| `memory_search(query)` | 0 | P3 store |
| `memory_remember(text)` | 1 | drafts a fact; the writer pass commits it (so a bad draft cannot corrupt memory) |
| `draft_message(text)` | 1 | returns a draft to the panel, sends nothing |
| `ping_owner(text)` | 3 | see §5 |

Keep `capture_screen`, `read_clipboard`, `capture_camera`, `set_active_task` as
they are today, with tiers 0/0/0/1 and the existing withholding rules.

---

## 2. Confirmation

Tier 2 is a hard gate, not a dialog that can be waived by text.

- Flow: `tool.confirm.request` event -> shell renders the exact tool name and
  the exact arguments (formatted, never a summary) -> user clicks allow once /
  allow for this session / deny -> `tool.confirm.response` with the nonce.
- The request carries `action_hash = sha256(name + canonical_json(args) +
  turn_id)`. The response must return the same hash; args changed after asking
  means a new confirmation.
- 60 s expiry, deny on timeout, deny on any malformed response.
- A confirmation never survives its turn. "Allow for this session" is a Tier 3
  grant, written to the allowlist with the same scope and an expiry.
- While a confirmation is pending, the turn is `blocked`; voice input still
  works but only to cancel ("no", "cancel", F9).
- Injected text in the same turn is shown to the user in the confirmation, so
  "the model wants to run rm -rf on your behalf" is visible.

---

## 3. Audit

One JSONL row per call in `logs/audit.jsonl` (the same file the router uses,
with `event: "tool.call"`):

```json
{"ts": "...", "event": "tool.call", "trace_id": "...", "turn_id": "...",
 "tool": "write_file", "tier": 2, "effective_tier": 2, "trust": "untrusted",
 "args_hash": "ab12...", "args_preview": {"path": "notes/todo.md"},
 "decision": "confirmed", "confirm_nonce": "9f...", "duration_ms": 41,
 "result": "ok", "result_bytes": 0, "job_id": null}
```

Never log file contents or clipboard text; `args_preview` is limited to
`audit_fields`. Tier 4 attempts are logged even though they cannot execute.

---

## 4. Mind job runner

```python
@dataclass
class Job:
    id: str
    goal: str
    steps: int = 0
    max_steps: int = 12
    deadline_s: float = 300
    tokens_used: int = 0
    state: str  # queued|running|blocked|done|failed|cancelled
```

- Started by `mind.start` (from a reflex handoff or a trigger); runs on the bus
  in the background; Reflex narrates progress in character ("hold on").
- Each step: one `mind` model call (through the router, P1), zero or more tool
  calls (through the policy above), and an `events` row.
- Cancellation: user speech, the Stop control, or the deadline. A cancelled job
  leaves a `cancelled` row and stops mid-step; no tool runs after that.
- Tool results are returned to the model truncated to a token cap; large output
  goes to `data/job-<id>/` and the model sees a path plus a head.
- A job cannot call `mind.start` again (no recursion) and cannot exceed its
  step budget. Both are enforced in the runner, not the prompt.

---

## 5. The one Tier 3 channel

`ping_owner(text)` exists so "notice I am gone and tell me" is possible without
opening a contact surface.

- One channel adapter, chosen by config; **ntfy** is the recommendation
  (`https://ntfy.sh/<topic>` or self-hosted). Telegram/Discord are not
  implemented.
- Off by default. Enabling requires a config flag, a topic, and a Tier 3 grant.
- Runtime limits: one ping per absence event, minimum 30 minutes between pings,
  max 280 chars, no attachments, no links unless they were in the user's own
  message.
- The ping body is composed locally and templated; the model may choose among
  three fixed phrasings, it may not compose arbitrary text with tool output in
  it. This is deliberate: the exfiltration surface of a free-text outbound tool
  is exactly the thing an agent should not have.
- Every send is audited and appears in the shell log panel.

This is separate from cloud egress. Enabling one never enables the other.

---

## 6. Config

```yaml
tools:
  allowed_dirs: ['C:\Users\User\Documents\Yuri-work']
  command_allowlist: ['git', 'python', 'npm', 'node', 'dir', 'type']
  confirm_timeout_s: 60
  hard_deny: []                 # code list is authoritative; this can only add

allowlist: []                   # Tier 3 grants: {tool, scope, expires, reason}

mind:
  max_steps: 12
  deadline_s: 300
  max_tokens: 60000

owner_ping:
  enabled: false
  channel: 'ntfy'
  ntfy_topic: ''
  min_interval_minutes: 30
  max_chars: 280
```

Every key defaulted in code. A missing `owner_ping` section means off.

---

## 7. Tests

- `test_registry.py` — every tool has a schema, a tier, and a handler; names
  unique; no handler reachable outside dispatch.
- `test_effective_tier.py` — table-driven: (declared tier, trust, source) ->
  effective tier; hard-deny unreachable even with a config grant.
- `test_confirmation.py` — wrong hash rejected, expired rejected, args swapped
  rejected, timeout denies, session grant applies only to its scope.
- `test_tool_safety.py` — the P4 gate: every tier attempted from an
  injected-content context is blocked or confirmed (no skips, no xfail).
- `test_job_runner.py` — step cap, deadline, cancel mid-step, no recursion,
  audit row per step, partial result on cancel.
- `test_ping_owner.py` — off by default, rate limit, once per absence, template
  only, audit row.
- `test_paths.py` — `..` traversal, symlink, UNC path, long path, case-only
  difference on NTFS.

---

## 8. Risks

1. **Confirmation fatigue.** A tool that asks too often gets clicked through.
   Mitigation: Tier 0/1 are free, Tier 2 is rare by construction, and a session
   grant exists for a scoped repeat.
2. **Path sandboxing on Windows.** Junctions, UNC, `\\?\` prefixes, case. Tests
   are the mitigation; `resolve()` + a strict prefix check on the canonical path.
3. **Command tool as an escape hatch.** Allowlisted binaries can still do damage
   (`python -c ...`). Mitigation: `run_command` is Tier 2 with the full command
   in the confirmation; it is not a sandbox and it says so in the UI.
4. **Jobs that never end.** Deadline plus step cap plus cancel; a job row left
   `running` after a crash is marked `failed` on next start.
5. **Ping as an exfiltration channel.** Fixed templates and no model text in the
   body; the only variable is which of three phrasings.

---

## 9. Gate

`PHASES.md` P4 gate, plus: no tool call in the audit log without an effective
tier, a decision, and a matching `turn_id`.
