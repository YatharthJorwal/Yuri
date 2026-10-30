# Phase 6 — Proactivity and the owner ping

She already says things unprompted: Task Guide chides on drift, Continuous OCR
remarks on the screen. Today both are ad-hoc intervals inside `app.py`. P6 turns
"unprompted" into a governed capability: triggers, a budget, a shared mouth, and
one carefully limited way to reach you when you are not there.

Dependencies: P2 (the arbiter), P4 (tiers, jobs), P5 (affect makes an unprompted
line feel motivated rather than scripted).

---

## 0. Goal

1. Triggers are a registry with cooldowns, not special cases in the turn loop.
2. A proactivity budget is enforced in code and visible in logs.
3. Every unprompted line goes through the same arbiter as voice.
4. Task Guide becomes one trigger-driven behavior, as v0.1 §8 asked.
5. The owner ping exists, is off by default, and cannot become a notification
   cannon.

---

## 1. Signals (perception, local only)

| Signal | Source | Cost |
| --- | --- | --- |
| idle time | `GetLastInputInfo` via ctypes | free |
| foreground window title/process | `GetForegroundWindow` + `GetWindowThreadProcessId` | free |
| screen changed | perceptual hash of a downscaled capture | one cheap capture per interval |
| time of day | clock | free |
| absence | idle > threshold while the app runs | free |
| task drift | Task Guide's state machine | one vision call when due |

`perception/windows.py` wraps the two Win32 calls with tests against a fake
window provider. Window titles are `secret` class: they never go to the cloud,
and they are not persisted beyond the current session's event log.

No new Python dependency: `ctypes` plus `user32.dll` is enough, and it keeps the
onboarding story to "Python + Ollama".

---

## 2. Triggers

```python
@dataclass(frozen=True)
class Trigger:
    name: str
    when: Callable[[Signals, State, float], bool]   # now
    cooldown_s: float
    cost: int            # budget units
    salience: float      # used to pick one when several fire
    kind: str            # "remark" | "chide" | "ping" | "job"
```

| Trigger | Fires when | Cooldown |
| --- | --- | --- |
| `idle_remark` | idle 45-90 min, not speaking, not listening | 45 min |
| `time_of_day` | first interaction in a new 4 h block, or "it's late" after 01:00 | 4 h |
| `window_change` | foreground app changes to something task-relevant | 10 min |
| `task_drift` | Task Guide check says drifted | Task Guide's own interval |
| `long_absence` | idle > `absence_minutes` (120) while the app is open | once per absence |
| `comeback` | first input after `long_absence` | once |
| `ambient_screen` | OCR watch decides a screen is worth a remark (existing) | 4 min, floor 2 min |

One trigger wins per tick: highest `salience` among those that pass their
cooldown and the budget. No stacking, no two remarks in a row.

The scheduler ticks every 10 s, is pure over `(signals, state, now)`, and is
unit-testable with a fake clock. This is the local equivalent of a cron, not a
new framework.

---

## 3. Proactivity budget

Token bucket, persisted, in `proactivity:` config:

- Capacity `max_per_hour` (default 2), refill 1 per `3600 / max_per_hour` s.
- Each fired trigger costs `trigger.cost` units (default 1).
- **Backoff on being ignored**: if an unprompted line gets no user response
  within 5 minutes, multiply the refill interval by `backoff_factor` (2), up to
  `backoff_max` (8). Any user turn resets it to 1.
- A hard floor per hour (`hard_min_interval_s`, 10 min) regardless of bucket.
- Quiet hours (`quiet_hours: ["23:30", "07:30"]`): only `long_absence`/ping may
  fire, and the ping only if it is the promoted Tier 3 case.
- All of it visible: one `proactivity.decided` event per tick with
  `{trigger, allowed, reason, budget}` so "why did she say that" has an answer.

---

## 4. The shared mouth

`ambient.proposed` events go to the arbiter from P2. Rules:

- `listening` or `transcribing`: drop silently (never interrupt the user).
- `thinking`: queue, deliver only if it is still relevant when the turn ends.
- `speaking`: drop.
- `idle`: speak.
- Voice replies always preempt ambient; ambient never preempts anything.
- A dropped line is logged with the reason, not retried.

This replaces the three-way collision risk between Task Guide, OCR watch, and
voice that exists in Luna today.

---

## 5. Task Guide as a trigger

Today `app.py` runs `_run_task_guide_check` on its own interval while a task is
active. P6 folds it into the scheduler:

- `task_drift` trigger consults `task_guide.py`'s existing `due_for_check` and
  screen-hash state, so it fires only when the screen actually changed since
  the last look. No change, no model call.
- The check still uses the structured-output call Luna built, still one call per
  check, still the same drift verdict shape.
- The chide is delivered as an `ambient.proposed` line with `kind: "chide"`,
  which makes it subject to the arbiter and the budget like everything else.
- `set_active_task` stays withheld from the model (the classifier owns task
  state). That decision is unchanged by this refactor.

Same for Continuous OCR: its state machine stays, its delivery path becomes the
arbiter, its comment decision keeps the similarity dedupe that stops repeated
remarks about an unchanged screen.

---

## 6. The owner ping

One trigger, one consent, one recipient (P4 §5):

- `long_absence` with `owner_ping.enabled` and a Tier 3 grant -> compose from
  three fixed templates -> send once -> rate-limit 30 min -> never repeat for
  the same absence.
- A ping is only sent while the app is running and Ollama answers; no watchdog
  service, no background daemon. If Yuri is closed, nothing pings.
- The shell shows the sent text in the log panel. A ping is always visible.

---

## 7. Config

```yaml
proactivity:
  enabled: true
  max_per_hour: 2
  hard_min_interval_s: 600
  backoff_factor: 2
  backoff_max: 8
  ignore_window_s: 300
  quiet_hours: ['23:30', '07:30']
  absence_minutes: 120
  triggers:
    idle_remark: {enabled: true, cooldown_s: 2700}
    time_of_day: {enabled: true, cooldown_s: 14400}
    window_change: {enabled: true, cooldown_s: 600}
    task_drift: {enabled: true}
    ambient_screen: {enabled: true, cooldown_s: 240}
```

Not all triggers are on at first. Ship with `idle_remark`, `task_drift` and
`ambient_screen`; leave `window_change` off until its false-positive rate is
measured.

---

## 8. Tests

- `test_scheduler.py` — each trigger's predicate against fake signals; cooldowns;
  one winner per tick; salience tie-break.
- `test_budget.py` — bucket refill with fake clock; backoff after ignored lines;
  reset on user turn; hard floor; quiet hours; `max_per_hour` never exceeded
  over a simulated 72 h.
- `test_arbiter_gate.py` — never in `listening`/`speaking`; queued then dropped
  when stale; voice always wins.
- `test_task_guide_trigger.py` — no model call when the screen hash is unchanged;
  one call when changed; chide is budgeted.
- `test_ping.py` — off by default, one per absence, rate limit, template only,
  not sent when the app is closed.
- `test_signals.py` — fake window provider; idle math; `secret` class on titles.

Gate: `PHASES.md` P6 gate (72 h soak with a simulated clock plus a real 8 h
overnight run with the app open and no false pings).

---

## 9. Risks

1. **Notification cannon.** The budget is the mitigation; the failure mode is
   tuning it up "because she is quiet". The config comment says why not.
2. **Idle detection while watching a video.** `GetLastInputInfo` counts mouse and
   keyboard only, so a 90-minute film looks like absence. Mitigation: fullscreen
   foreground window suppresses `idle_remark`; `long_absence` uses a longer
   threshold and still respects the ping rate limit.
3. **Window titles as a privacy leak.** Local-only class, never persisted, never
   in a prompt that can reach the cloud (§1).
4. **Two schedulers.** Task Guide's own loop and the new scheduler can both fire.
   Mitigation: delete the old loops in the same change; one owner per behavior.
5. **Unprompted lines on a busy machine.** The arbiter queues them; a queue that
   grows is a bug, so it is bounded (max 2) and logged when dropped.
