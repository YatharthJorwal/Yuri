# Avatar model

Drop your VRM here as **yuri.vrm**:

```
public/vrm/yuri.vrm
```

Yuri loads `/vrm/yuri.vrm` at startup. Any VRM 0.x or 1.0 export works; VRM 1.0
is preferred (the shell rotates 0.x models automatically).

To use a different filename without editing source, set a global before the
bundle loads:

```html
<script>window.__YURI_MODEL_PATH__ = "/vrm/your-model.vrm";</script>
```

If this directory has no model, the window opens a short instruction panel
instead of the avatar. Nothing else in the repo depends on the model file, and
it is deliberately not committed (see `.gitignore`).
