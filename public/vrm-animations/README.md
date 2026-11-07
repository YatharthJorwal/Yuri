# public/vrm-animations/

Two sources of `.vrma` clips live here side by side:

- **Lowercase filenames** (`world-walk.vrma`, `idle.vrma`, `happy.vrma`,
  `angry.vrma`, `relaxed.vrma`, etc., ~113 files) — the "Hanami" VRMA
  pack, sourced from Overte (Apache-2.0) and Quaternius (CC0). See
  `NOTICE.md` for the full per-file attribution and `PACK-README.md`
  for the pack's own engineering documentation (seam measurements,
  phase contracts, the `world.json`/`transitions.json` schema). The
  `LICENSES/` folder holds the full license texts required for
  redistribution. The Microsoft Rocketbox (MIT) family and the `extra/`
  folder from the original pack were **not** copied in — they're an
  alternate, unused animation style and a set of clips the pack's own
  authors set aside, respectively (see `NOTICE.md` §5 if you want them
  later; `NOTICE.md` itself documents them even though the files aren't
  here).
- **Capitalized filenames** (`Surprised.vrma`, `Clapping.vrma`, etc.) —
  clips sourced separately, filling roles the pack above doesn't cover
  (it has no "surprised" clip, and no "teasing" role either).

**Why the split by case matters:** two of the original capitalized
clips (`Angry.vrma`, `Sad.vrma`) were removed in the same round the pack
was added, because they collided case-insensitively with the pack's own
`angry.vrma`/`sad.vrma` — harmless on Linux/most git hosting, but a real
silent-overwrite risk on Windows/macOS's case-insensitive filesystems.
Both roles are now filled by the pack's fitted versions instead (see
`docs/DECISIONS.md`'s "go big" entry). If you add more of your own
clips here, avoid names that differ from an existing pack file only by
case.

See `src/sandbox.ts` for what actually gets loaded (`GESTURE_CLIP_FILES`,
`IDLE_BASE_FILES`, `IDLE_TALKING_FILES`, and the `WORLD_WALK_*`
constants) — loading is best-effort throughout, so a file listed there
but missing from this folder just means that one clip/pose never
registers, not a crash.
