# VRMA animation pack

Drop your `.vrma` clips here. The shell looks for clips by name (idle, nod,
think, happy, angry, sad, surprised, and the `world-*` set), so keep the
Luna-era filenames unless you also update `src/main.ts`.

If this directory is empty, the avatar still loads but has no clip-based
animation: the procedural idle (breathing, sway, blink, gaze) in
`src/idle-motion.ts`, `src/eyes.ts` and `src/expression-life.ts` keeps working
because it is generated in code, not loaded from files.

Nothing in this directory is committed (see `.gitignore`); the packs are large
and may be licensed per-source.
