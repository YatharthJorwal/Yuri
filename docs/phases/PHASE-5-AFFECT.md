# Phase 5 — Affect and avatar life

She should look like she has an inner state, not like an animation player.
Luna already built the body: `idle-motion.ts` (breathing, sway, head drift),
`eyes.ts` (blink variety, gaze, micro-saccades), `expression-life.ts` (mood holds
then fades), and `lipsync.ts`. All three are driven by time and constants today.
P5 gives them one state to be driven by.

---

## 0. Goal

One small, persisted affect vector updated by real events, decaying to a
baseline, feeding three places in this order of importance: the prompt, the VRM,
the voice. Nothing here is random.

Dependencies: P2 (speech events: who spoke, barge-in, silence) and P1 (the
router, for the prompt line's budget). Can be built after P2.

---

## 1. State

```python
@dataclass
class Affect:
    mood: float       # -1 (sour) .. +1 (warm), baseline 0.1
    energy: float     #  0 (drained) .. 1 (alert), baseline 0.55
    affection: float  #  0 .. 1, baseline 0.4
    irritation: float #  0 .. 1, baseline 0.0
    updated_at: float
```

Update rule, per event:

```
target = clamp(baseline + sum(contributions))
value += (target - value) * (1 - exp(-dt / tau_event))
```

Decay toward baseline uses a slower tau (`tau_decay`, default 1800 s for mood and
irritation, 3600 s for energy, none for affection), applied lazily whenever the
state is read, so a sleeping app costs nothing and does not "wake up angry".

Bounded per event: no single event moves a value by more than `max_step` (0.25),
so one bad tool call cannot flip her whole mood.

Persisted in SQLite (`affect` table, one row) every write, so a restart keeps the
mood she had. Nothing is sent anywhere.

---

## 2. Events that move it

| Event | Mood | Energy | Affection | Irritation |
| --- | --- | --- | --- | --- |
| user says thanks / compliments | +0.15 | +0.02 | +0.06 | -0.10 |
| user insults her (in play or not) | -0.08 | 0.0 | -0.02 | +0.12 |
| user ignores an unprompted line | -0.05 | -0.02 | -0.01 | +0.06 |
| long absence (> 2 h) then return | +0.10 | +0.05 | +0.03 | -0.05 |
| task completed successfully | +0.10 | -0.05 | +0.02 | -0.05 |
| tool failed / model unreachable | -0.10 | -0.03 | 0.0 | +0.10 |
| barge-in (she was interrupted) | -0.04 | +0.02 | 0.0 | +0.08 |
| idle for > 1 h, no interaction | 0.0 | -0.10 | 0.0 | 0.0 |

Contributions are a config table, not code, so tuning does not mean a rebuild.

---

## 3. To the prompt

One line, appended to the persona block, <= 40 tokens:

```
Right now: mood slightly warm, energy low, a little irritated (she has been
interrupted twice). Do not announce this; let it color the reply.
```

Rules:

- Only qualitative words, never numbers, never "your affection is 0.42".
- The model still chooses the `[emotion]` tag. Affect biases by *tie-break*: if
  the model's tag contradicts a strong state (e.g. `[happy]` while irritation is
  0.9), the shell applies the softer blend rather than overriding the tag.
- The line is `local_only`: it never travels with a cloud Mind call unless the
  turn already qualifies for cloud, in which case it is `internal` and capped.

---

## 4. To the VRM

The shell receives `affect.updated` events and maps them onto the modules that
already exist:

| Value | Drives |
| --- | --- |
| energy | `idleOffsets(t, intensity)` amplitude and breath rate bias; blink interval scale; saccade rate |
| mood | expression baseline offset (a held `relaxed`/`sad`/`happy` blend at 30-50% weight) |
| affection | gaze preference toward the user's side, glance length |
| irritation | fidget/tempo: slightly faster head drift, `angry` micro-offset |

Constraint: the emotion tag still wins for the duration of a reply. Affect only
sets the *resting* state and the intensity. That keeps her responsive without
fighting the reply.

`expression-life.ts` already holds and fades a mood; P5 changes its input from a
timer to the vector, and caps how often the resting expression may change (no
flicker: minimum 20 s between baseline changes).

---

## 5. To the voice

Honest scope: GPT-SoVITS `api_v2` exposes sampling and a speed factor, not
emotion conditioning. So affect can only do two things safely:

1. `speed_factor` within a narrow band (0.95-1.08) from energy.
2. Clause pause length from mood (a longer breath after a downbeat line).

If the server rejects `speed_factor`, it is dropped and logged once; no retries
per clause. Emotion in the *words* stays the persona's job; this phase does not
pretend to do voice acting it cannot do.

---

## 6. Events and ownership

| Event | Emitted by | Payload |
| --- | --- | --- |
| `affect.event` | voice, tools, scheduler | `{kind, magnitude}` from the table |
| `affect.updated` | affect service | full vector + a one-word mood label |
| `affect.state` | on shell connect | current vector, so a reconnect does not reset her face |

One writer only: the affect service. Everything else emits contributions.

---

## 7. Tests

- `test_affect.py` — the 20-event scripted scenario, asserting the trajectory
  against expected sign and magnitude (not exact floats; tolerance ±0.05).
- `test_affect_decay.py` — after 3 h of simulated time, mood returns to within
  0.05 of baseline; energy halves; affection does not decay.
- `test_affect_bounds.py` — no single event exceeds `max_step`; values clamp.
- `test_affect_prompt.py` — the line is <= 40 tokens, contains no numbers, and
  is absent when the state is at baseline.
- `test_affect_persistence.py` — restart restores the vector.
- Frontend: `test:fe` covers `idleOffsets` intensity bounds, blink interval
  scaling, and that a baseline change respects the 20 s floor.

Gate: three screenshots (irritated, warm, drained) rendered from the same model
under the same lighting, visibly different; a 60 s idle capture with no visible
loop seam; the prompt line round-trips.

---

## 8. Risks

1. **Uncanny acting.** Too much expression reads as cartoonish. Mitigation:
   everything is a weight, defaults are small, and `expression-life` already
   fades rather than snaps.
2. **Mood whiplash.** Events can stack in one turn. Mitigation: `max_step` per
   event, and one affect update per turn (batched).
3. **Prompt cost.** 40 tokens per turn on the reflex path is real. Mitigation:
   the line is omitted at baseline and cached with the system message.
4. **Fighting the emotion tag.** Mitigation: the tag wins during a reply; affect
   only sets the resting state (§4).
5. **Energy driven by CPU load instead of events.** Tempting, and wrong: a slow
   model is not "she is tired". Energy comes from interaction cadence, not
   thermals.
