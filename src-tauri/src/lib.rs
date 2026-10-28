// NOTE: this sandbox has no Rust toolchain, so this file was originally
// written against the Tauri v2 API from memory and unverified. Confirmed
// building and running successfully on the user's real machine as of a
// launch that included spawn_backend_processes, the tray icon/camera
// indicator code, and everything else through that point -- the file is
// no longer "unverified from memory," but new additions since then still
// start in that same unverified state until their own first real build.
// Newest additions, not yet through a real build: the Job Object
// (install_kill_on_close_job), the startup sweep (kill_stale_listeners /
// is_python_process), and the `gpt_sovits_still_alive` try_wait check in
// spawn_backend_processes.

use std::collections::HashMap;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{
    image::Image,
    menu::{Menu, MenuItem},
    tray::{TrayIcon, TrayIconBuilder},
    Emitter, Manager, PhysicalPosition, Position, WindowEvent,
};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Shortcut, ShortcutState};

/// Embedded at compile time rather than read from disk at runtime -- same
/// "don't depend on a working-directory-relative path being right" reasoning
/// as everything else that's been machine-specific trouble in this file.
/// Swapped onto the tray via set_camera_indicator (see its own doc comment)
/// whenever the frontend's camera toggle is armed; swapped back to
/// `app.default_window_icon()` when it isn't. Needs the `image-png` Cargo
/// feature (Cargo.toml) for `Image::from_bytes` below to even compile.
const CAMERA_ACTIVE_TRAY_ICON: &[u8] = include_bytes!("../icons/tray-camera-active.png");

/// Handle to the tray icon itself, stored as Tauri-managed state (same
/// pattern as `ManagedChildren`) so `set_camera_indicator` can swap its
/// icon later -- `build_tray()`'s own `.build(app)?` call used to just
/// discard this.
struct TrayIconHandle(TrayIcon);

/// No visible console window for a spawned child process. Same numeric
/// flag Windows' own CreateProcess API uses -- there's no named constant
/// for it in std, so it's a bare literal (0x08000000, CREATE_NO_WINDOW,
/// per the Win32 docs).
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Handles to every child process Luna spawned itself (GPT-SoVITS, the
/// orchestrator), stored as Tauri-managed state so the tray menu's "quit"
/// handler can shut them down before actually exiting -- spawning them
/// hidden and then never cleaning them up would leave them running
/// invisibly in the background forever, which is worse than the old
/// visible-terminal setup, not better.
///
/// Each entry is labeled (not just a bare `Child`) so the quit handler can
/// tell which one is the orchestrator -- it's the only one with a
/// graceful-shutdown HTTP endpoint to try first (see
/// `graceful_shutdown_then_kill()`); GPT-SoVITS has no such thing and is
/// just killed outright, same as before.
struct ManagedChildren(Arc<Mutex<Vec<(String, Child)>>>);

/// Exposed to the frontend for later phases (e.g. a keyboard shortcut or a
/// HUD button) even though only the tray menu drives it in Phase 1.
#[tauri::command]
fn toggle_click_through(window: tauri::WebviewWindow, ignore: bool) -> Result<(), String> {
    window
        .set_ignore_cursor_events(ignore)
        .map_err(|e| e.to_string())
}

/// **UNVERIFIED -- no Rust toolchain in the sandbox this was written in.**
/// Phase 5's camera indicator, per `docs/ARCHITECTURE.md`'s own spec for
/// `capture_camera()`: "a visible 'camera active' indicator (tray icon
/// state) whenever it's used, even though nothing is displayed back to the
/// user." Called from the frontend's camera module (src/camera.ts) the
/// instant `getUserMedia` actually grants a live stream (active=true) and
/// the instant that stream is stopped (active=false) -- tracks "is the
/// camera stream open," not literally "is a frame being read from it this
/// millisecond," which is both simpler and more honest: the OS's own
/// camera light works the same way (on for the whole time a stream is
/// open, not just per-frame).
#[tauri::command]
fn set_camera_indicator(
    app: tauri::AppHandle,
    tray: tauri::State<TrayIconHandle>,
    active: bool,
) -> Result<(), String> {
    let icon = if active {
        Image::from_bytes(CAMERA_ACTIVE_TRAY_ICON).map_err(|e| e.to_string())?
    } else {
        app.default_window_icon()
            .cloned()
            .ok_or_else(|| "no default window icon configured".to_string())?
    };
    tray.0.set_icon(Some(icon)).map_err(|e| e.to_string())
}

/// Puts this process (and therefore every child it later spawns --
/// GPT-SoVITS, the orchestrator and the real interpreter the venv's
/// python.exe launcher stub spawns underneath it, WebView2's helpers) into
/// a Windows Job Object flagged KILL_ON_JOB_CLOSE. The OS itself then kills
/// every process in the job the instant this one dies, *for any reason*:
/// tray Quit, End Task from the taskbar, Ctrl+C on `npm run tauri dev`, a
/// crash, Task Manager. That is the whole point -- every fix before this one
/// (graceful_shutdown_then_kill, the orchestrator's PID-file takeover) only
/// helps on paths where this process gets to run its own code before dying,
/// and End Task / Ctrl+C are exactly the paths where it doesn't, which is
/// what left orphaned GPT-SoVITS instances squatting port 9880 across
/// launches on the user's real machine.
///
/// Same pattern the Tauri project itself uses in tauri-driver. Kept alive
/// with mem::forget rather than stored anywhere: the job's handle must stay
/// open for the whole life of the process (dropping it would *trigger* the
/// kill early), and the OS closing it at process exit is exactly the
/// behavior wanted. A failure here is logged and otherwise ignored -- the
/// graceful-quit path and the PID-file takeover still exist as before, this
/// is an additional safety net, not a replacement.
///
/// **UNVERIFIED on a real build** -- API confirmed against the win32job
/// docs and tauri-driver's own usage, but never compiled here.
#[cfg(windows)]
fn install_kill_on_close_job() {
    let result = (|| -> Result<win32job::Job, Box<dyn std::error::Error>> {
        let job = win32job::Job::create()?;
        let mut info = job.query_extended_limit_info()?;
        info.limit_kill_on_job_close();
        job.set_extended_limit_info(&mut info)?;
        job.assign_current_process()?;
        Ok(job)
    })();
    match result {
        Ok(job) => {
            eprintln!(
                "[luna] kill-on-close job object ACTIVE -- Windows will kill \
                 GPT-SoVITS and the orchestrator if Luna dies for any reason \
                 (End Task, Ctrl+C, crash, tray Quit)."
            );
            std::mem::forget(job);
        }
        Err(e) => eprintln!(
            "[luna] WARNING: couldn't set up the kill-on-close job object ({e}); \
             children may be orphaned if Luna is closed via End Task or Ctrl+C \
             instead of tray Quit."
        ),
    }
}

pub fn run() {
    // Must happen before anything spawns a child -- processes only inherit
    // job membership from a parent that's already in the job.
    #[cfg(windows)]
    install_kill_on_close_job();

    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            toggle_click_through,
            set_camera_indicator
        ])
        .setup(|app| {
            let window = app
                .get_webview_window("main")
                .expect("main window must exist -- check the label in tauri.conf.json");

            anchor_bottom_right(&window);
            build_tray(app)?;
            register_push_to_talk_hotkey(app)?;

            let managed_children: Arc<Mutex<Vec<(String, Child)>>> = Arc::new(Mutex::new(Vec::new()));
            spawn_backend_processes(managed_children.clone());
            app.manage(ManagedChildren(managed_children));

            Ok(())
        })
        .on_window_event(|window, event| {
            // Luna is meant to live in the tray, not get closed by accident.
            // The only real "quit" is the tray menu's Quit item.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Luna");
}

/// Places the window in the bottom-right corner of the primary monitor on
/// first launch, with a small margin, instead of wherever Tauri defaults to.
fn anchor_bottom_right(window: &tauri::WebviewWindow) {
    let Ok(Some(monitor)) = window.primary_monitor() else {
        return;
    };
    let Ok(win_size) = window.outer_size() else {
        return;
    };

    let screen = monitor.size();
    let margin: i32 = 24;
    let x = screen.width as i32 - win_size.width as i32 - margin;
    let y = screen.height as i32 - win_size.height as i32 - margin;

    let _ = window.set_position(Position::Physical(PhysicalPosition { x, y }));
}

/// F9 push-to-talk: registered as an OS-level global shortcut, so it works
/// regardless of which window has focus (a game, a browser, whatever) --
/// unlike a plain keydown listener in the frontend, which only fires while
/// Luna's own window is focused, defeating the point of an always-on-top
/// desktop companion you talk to while doing something else. Emits a
/// "hotkey-talk" event with "pressed"/"released" as the payload; the
/// frontend (src/mic.ts, wired up in src/main.ts) does the actual recording
/// start/stop, same as a mic-button click -- this function's only job is
/// turning a raw key event into that event.
fn register_push_to_talk_hotkey(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let window = app
        .get_webview_window("main")
        .expect("main window must exist -- check the label in tauri.conf.json");

    let push_to_talk = Shortcut::new(None, Code::F9);
    let shortcut_for_handler = push_to_talk.clone();

    app.handle().plugin(
        tauri_plugin_global_shortcut::Builder::new()
            .with_handler(move |_app, scut, event| {
                if scut != &shortcut_for_handler {
                    return;
                }
                let state = match event.state() {
                    ShortcutState::Pressed => "pressed",
                    ShortcutState::Released => "released",
                };
                let _ = window.emit("hotkey-talk", state);
            })
            .build(),
    )?;

    app.global_shortcut().register(push_to_talk)?;

    Ok(())
}

/// Reads src-tauri/launcher.local.txt (gitignored -- see
/// launcher.local.txt.example next to it) for machine-specific paths this
/// repo can't hardcode, the same reason start-luna.bat itself stayed
/// gitignored -- one "KEY=value" per line, "#" comments and blank lines
/// ignored. Returns an empty map, not an error, if the file doesn't exist
/// yet -- GPT-SoVITS auto-start is just skipped in that case (tts.py's
/// existing pyttsx3 fallback still covers voice output either way), same
/// non-fatal spirit as the old .bat file simply not being filled in.
fn read_launcher_config() -> HashMap<String, String> {
    let mut map = HashMap::new();
    let Ok(contents) = std::fs::read_to_string("launcher.local.txt") else {
        return map;
    };
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            map.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    map
}

/// Spawns GPT-SoVITS's API server and Luna's own orchestrator as hidden
/// background processes -- `npm run tauri dev` (or the packaged .exe
/// later) becomes the one thing you run, replacing the old
/// start-luna.bat's three separate visible terminal windows. Output goes
/// to logs/*.log instead of a console, since a hidden process obviously
/// can't show one -- open those files if something's not responding, same
/// information the old terminals gave you, just not in a popup window.
///
/// Working directories below are relative to src-tauri/ (this binary's
/// cwd in `cargo run`/`tauri dev`) -- ../orchestrator and ../logs both
/// resolve to the repo root's orchestrator/ and logs/ as intended from
/// there. Not verified for a packaged release build, where the cwd may
/// differ -- worth rechecking when packaging is actually on the table.
fn spawn_backend_processes(children: Arc<Mutex<Vec<(String, Child)>>>) {
    let logs_dir = PathBuf::from("../logs");
    let _ = std::fs::create_dir_all(&logs_dir);

    let config = read_launcher_config();

    // Startup sweep -- see kill_stale_listeners' own doc comment. Must run
    // before anything below spawns, so the fresh processes bind cleanly.
    if config.contains_key("GPT_SOVITS_DIR") {
        kill_stale_listeners(9880, "GPT-SoVITS");
    }
    kill_stale_listeners(8765, "orchestrator");

    if let Some(sovits_dir) = config.get("GPT_SOVITS_DIR") {
        // Spawning runtime\python.exe directly, not through `cmd /C`, is
        // deliberate -- see the quit handler's comment on why that matters
        // for being able to actually kill it later.
        let python_exe = Path::new(sovits_dir.as_str()).join("runtime").join("python.exe");
        let mut cmd = Command::new(&python_exe);
        cmd.arg("api_v2.py")
            .current_dir(sovits_dir)
            // Windows' default console codepage (cp1252) doesn't cover
            // whatever non-ASCII output api_v2.py produces, which raised
            // UnicodeEncodeError and killed the process outright once its
            // stdout/stderr were redirected to a log file instead of a
            // real console (spawn_logged() below) -- found on the user's
            // actual machine, not predicted in advance.
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            // Without this, Python fully buffers stdout/stderr once
            // they're not attached to a real terminal (exactly what
            // spawn_logged()'s file redirection below does) -- meaning
            // every print() call, including the diagnostic ones tts.py
            // relies on for "check the log to see what actually
            // happened," can sit unwritten for a long time instead of
            // showing up when it actually occurred. Found by a real
            // debugging session where gpt_sovits.log clearly showed a
            // completed request but the corresponding print in the log
            // that should have accompanied it never appeared -- not
            // theoretical.
            .env("PYTHONUNBUFFERED", "1");
        spawn_logged("GPT-SoVITS", &mut cmd, &logs_dir.join("gpt_sovits.log"), &children);
    } else {
        eprintln!(
            "[luna] GPT_SOVITS_DIR not set in src-tauri/launcher.local.txt -- \
             skipping GPT-SoVITS auto-start, voice will fall back to pyttsx3 \
             (see launcher.local.txt.example)."
        );
    }

    // The orchestrator waits for GPT-SoVITS's port to actually be
    // listening before starting -- same problem the old .bat file's blind
    // `timeout /t 8` was working around, except this polls for the real
    // signal instead of guessing a fixed delay, and still gives up after a
    // bounded wait rather than blocking forever if GPT-SoVITS never comes
    // up (tts.py's own per-turn pyttsx3 fallback covers that case, so
    // starting the orchestrator anyway afterward is still the right call,
    // not a hard failure). Runs on its own thread so it doesn't hold up
    // window creation -- the frontend's own WebSocket reconnect loop
    // already covers "the orchestrator isn't up yet," so there's nothing
    // else here that needs to wait on it either.
    let orchestrator_children = children;
    std::thread::spawn(move || {
        wait_for_port(9880, Duration::from_secs(60));

        // Something answered on 9880 -- but is it actually the GPT-SoVITS
        // process this launch just spawned, or a stale leftover from a
        // previous session that was already squatting the port before
        // this one even tried to bind (the exact scenario found on the
        // user's real machine: a fresh spawn attempt failing with
        // WinError 10048/"only one usage of each socket address" while
        // an old orphaned instance keeps answering requests)? Checked
        // against the real Child handle's own exit status, not just
        // trusting the port -- `children` already holds it, no new
        // dependency or PID-lookup shelling-out needed.
        let gpt_sovits_still_alive = {
            let mut guard = orchestrator_children.lock().unwrap();
            match guard.iter_mut().find(|entry| entry.0 == "GPT-SoVITS") {
                Some(entry) => matches!(entry.1.try_wait(), Ok(None)),
                // GPT_SOVITS_DIR wasn't set, so nothing was ever spawned
                // -- not this warning's problem, the "skipping GPT-SoVITS
                // auto-start" message above already covers that case.
                None => true,
            }
        };
        if !gpt_sovits_still_alive {
            eprintln!(
                "[luna] WARNING: port 9880 answered, but the GPT-SoVITS \
                 process this launch just started has already exited -- \
                 something ELSE is squatting that port, almost certainly \
                 a leftover process from a previous session that never \
                 shut down cleanly. Voice will likely come from that \
                 stale process instead of this launch's own instance, \
                 which may be fine or may be broken/outdated -- not \
                 predictable either way. Fix: run `netstat -ano | findstr \
                 :9880` in PowerShell, find the PID in the last column, \
                 `taskkill /PID <pid> /F` it, then relaunch Luna for a \
                 clean start. See README.md's troubleshooting section for \
                 the same steps."
            );
        }

        let python = PathBuf::from("../orchestrator/venv/Scripts/python.exe");
        if !python.exists() {
            eprintln!(
                "[luna] {} not found -- orchestrator venv not set up yet? \
                 Run `pip install -r requirements.txt` in orchestrator/venv \
                 first (see README).",
                python.display()
            );
            return;
        }

        let mut cmd = Command::new(&python);
        cmd.arg("app.py")
            .current_dir("../orchestrator")
            // Same reasoning as GPT-SoVITS's spawn above -- app.py's own
            // print() calls (including ones with flush=True already, but
            // no reason to depend on every call site remembering that)
            // need this to reliably show up in orchestrator.log in real
            // time rather than whenever Python's internal buffer happens
            // to fill.
            .env("PYTHONUNBUFFERED", "1");
        spawn_logged(
            "orchestrator",
            &mut cmd,
            &logs_dir.join("orchestrator.log"),
            &orchestrator_children,
        );
    });
}

/// True only for Python interpreters -- the one kind of process the startup
/// sweep below is willing to kill. Luna's own children (GPT-SoVITS's embedded
/// python.exe, the orchestrator's venv python.exe) are all Python, so this
/// is enough to clean up Luna's own orphans without ever touching, say, a
/// browser or a game that happens to be sitting on the same port.
fn is_python_process(pid: u32) -> bool {
    let out = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
    match out {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout).to_lowercase();
            text.contains("\"python.exe\"") || text.contains("\"pythonw.exe\"")
        }
        Err(_) => false,
    }
}

/// **UNVERIFIED -- no Rust toolchain in the sandbox this was written in.**
/// Startup sweep: before spawning anything, kill any stale Python process
/// still LISTENING on `port` from a previous Luna run. This is the safety
/// net behind the Job Object (install_kill_on_close_job): the Job Object
/// stops *new* orphans, but this also clears ones left by older builds, by
/// a Job Object that failed to install, or by anything else -- so a stale
/// GPT-SoVITS can never again silently squat port 9880 and make a fresh
/// launch fail to bind. Same manual steps the user had to do by hand
/// (`netstat -ano | findstr :9880`, `taskkill /PID <pid> /F`), automated,
/// with `/T` so a launcher stub's real-interpreter child dies with it.
/// Deliberately refuses to kill non-Python holders (see is_python_process)
/// and says so instead of failing silently.
fn kill_stale_listeners(port: u16, what: &str) {
    let output = match Command::new("netstat")
        .args(["-ano", "-p", "TCP"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "[luna] startup sweep: couldn't run netstat ({e}); not checking port {port} ({what})"
            );
            return;
        }
    };
    let text = String::from_utf8_lossy(&output.stdout);
    let suffix = format!(":{port}");
    let mut pids: Vec<u32> = Vec::new();
    for line in text.lines() {
        // Proto  Local Address  Foreign Address  State  PID
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() >= 5 && cols[0] == "TCP" && cols[1].ends_with(&suffix) && cols[3] == "LISTENING"
        {
            if let Ok(pid) = cols[4].parse::<u32>() {
                if !pids.contains(&pid) {
                    pids.push(pid);
                }
            }
        }
    }

    let mut killed_any = false;
    for pid in pids {
        if pid == std::process::id() {
            continue;
        }
        if !is_python_process(pid) {
            eprintln!(
                "[luna] startup sweep: port {port} ({what}) is held by pid {pid}, which isn't a \
                 Python process -- leaving it alone. Free that port yourself or the {what} \
                 won't be able to start."
            );
            continue;
        }
        let result = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F", "/T"])
            .creation_flags(CREATE_NO_WINDOW)
            .output();
        match result {
            Ok(o) if o.status.success() => {
                killed_any = true;
                eprintln!(
                    "[luna] startup sweep: killed stale python process (pid {pid}) still \
                     listening on port {port} ({what}) from a previous run."
                );
            }
            _ => eprintln!(
                "[luna] startup sweep: tried to kill stale python pid {pid} on port {port} \
                 ({what}) but taskkill failed -- kill it manually."
            ),
        }
    }
    if killed_any {
        // Give the OS a moment to actually release the socket so the fresh
        // process's bind doesn't race the dying one's.
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Polls 127.0.0.1:<port> until something's listening, or gives up after
/// `max_wait`. A bare TCP connect, not an HTTP-level readiness check --
/// GPT-SoVITS's api_v2.py might bind the port slightly before it's
/// actually finished loading the model onto the GPU, so this is a
/// reasonable proxy for "probably ready," not a guarantee. Better than a
/// blind fixed timeout either way; upgrade to an actual HTTP health check
/// later if this proves too eager in practice.
fn wait_for_port(port: u16, max_wait: Duration) {
    let addr: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .expect("hardcoded loopback address is always valid");
    let start = Instant::now();
    while start.elapsed() < max_wait {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    eprintln!(
        "[luna] nothing answered on port {port} within {}s -- starting the \
         orchestrator anyway.",
        max_wait.as_secs()
    );
}

/// Spawns `command` hidden (no console window), with both stdout and
/// stderr going to `log_path`, and records the resulting Child (alongside
/// `label`, so the quit handler can tell processes apart -- see
/// ManagedChildren's doc comment) in `children` so it can be shut down on
/// quit. `command` should invoke the real interpreter directly, not
/// through a shell -- see the quit handler's comment for why.
fn spawn_logged(
    label: &str,
    command: &mut Command,
    log_path: &std::path::Path,
    children: &Arc<Mutex<Vec<(String, Child)>>>,
) {
    let stdout_file = match std::fs::File::create(log_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[luna] couldn't open {}: {e}", log_path.display());
            return;
        }
    };
    let stderr_file = match stdout_file.try_clone() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[luna] couldn't duplicate the log handle for {label}: {e}");
            return;
        }
    };

    match command
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
    {
        Ok(child) => {
            if let Ok(mut children) = children.lock() {
                children.push((label.to_string(), child));
            }
        }
        Err(e) => eprintln!("[luna] failed to start {label}: {e}"),
    }
}

/// Tray icon + menu: Show/Hide, Toggle Click-through, Quit.
fn build_tray(app: &tauri::App) -> tauri::Result<()> {
    let show_hide = MenuItem::with_id(app, "show_hide", "Show/Hide Luna", true, None::<&str>)?;
    let click_through = MenuItem::with_id(
        app,
        "click_through",
        "Toggle Click-through",
        true,
        None::<&str>,
    )?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show_hide, &click_through, &quit])?;

    // Interior mutability instead of a mutable capture: on_menu_event's
    // closure is `Fn`, not `FnMut` (it may be invoked from more than one
    // call site), so a plain `let mut` captured by the closure won't compile.
    let click_through_state = Arc::new(AtomicBool::new(false));

    let tray = TrayIconBuilder::new()
        .icon(app.default_window_icon().unwrap().clone())
        .menu(&menu)
        .on_menu_event(move |app, event| {
            let Some(window) = app.get_webview_window("main") else {
                return;
            };
            match event.id().as_ref() {
                "show_hide" => {
                    let visible = window.is_visible().unwrap_or(true);
                    let _ = if visible { window.hide() } else { window.show() };
                }
                "click_through" => {
                    let now_ignoring = !click_through_state.load(Ordering::Relaxed);
                    click_through_state.store(now_ignoring, Ordering::Relaxed);
                    let _ = window.set_ignore_cursor_events(now_ignoring);
                }
                "quit" => {
                    // Try a graceful shutdown first, falling back to a
                    // hard kill for anything still alive after a bounded
                    // wait -- see graceful_shutdown_then_kill()'s own doc
                    // comment for why child.kill() alone isn't enough
                    // anymore (it was silently losing Phase 3 memory
                    // consolidation on every quit, found while answering
                    // the user's own "what's the right way to close this"
                    // question -- see docs/DECISIONS.md).
                    if let Some(state) = app.try_state::<ManagedChildren>() {
                        graceful_shutdown_then_kill(&state.0);
                    }
                    app.exit(0)
                }
                _ => {}
            }
        })
        .build(app)?;

    app.manage(TrayIconHandle(tray));

    Ok(())
}

/// **UNVERIFIED -- no Rust toolchain in the sandbox this was written in.**
/// Written as carefully as the rest of this file, but treat with the same
/// suspicion the global-shortcut block started in before its one real
/// E0277 fix on first compile (see docs/DECISIONS.md) -- this is new code
/// of a similar kind, and there's a real chance something here needs a
/// small fix too on your first `cargo build`.
///
/// Tries a graceful shutdown of the orchestrator first -- a fire-and-
/// forget POST to its own `/shutdown` endpoint (see
/// orchestrator/app.py's `shutdown_endpoint()` docstring) -- because
/// `child.kill()` alone (`TerminateProcess` on Windows) gives Python's
/// own `finally` block, and therefore Phase 3's `consolidate_session()`,
/// no chance to run at all. Every quit was silently losing that
/// session's memory before this existed.
///
/// Falls back to a hard `.kill()` for anything still alive after a
/// bounded wait, regardless of label -- GPT-SoVITS has no graceful
/// shutdown path of its own (nothing here asks it for one), and the
/// orchestrator itself falls back to this too if it doesn't finish
/// tearing down in time -- so quitting never hangs waiting on either one,
/// and never leaves an orphaned process behind either way.
fn graceful_shutdown_then_kill(children: &Arc<Mutex<Vec<(String, Child)>>>) {
    request_orchestrator_shutdown();

    // Matched to orchestrator/app.py's own ~5s bounded wait for its
    // connections to finish tearing down -- if that window's ever too
    // short for a slow consolidation LLM call, it's a mismatch to widen
    // on both sides together, not just one.
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let all_exited = match children.lock() {
            Ok(mut children) => children
                .iter_mut()
                .all(|(_label, child)| matches!(child.try_wait(), Ok(Some(_)))),
            Err(_) => break,
        };
        if all_exited || Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }

    // Whatever's still alive at this point (graceful shutdown didn't
    // finish in time, or it's GPT-SoVITS which never had a graceful path
    // to begin with) gets hard-killed here, same as the old unconditional
    // behavior. Calling .kill() on a process that already exited on its
    // own is a harmless no-op error on Windows, so there's no need to
    // re-check try_wait() here first -- just always attempt it.
    if let Ok(mut children) = children.lock() {
        for (_label, child) in children.iter_mut() {
            let _ = child.kill();
        }
    }
}

/// Fire-and-forget raw HTTP/1.1 POST to the orchestrator's `/shutdown`
/// endpoint -- a plain `TcpStream` write, not a new HTTP client
/// dependency (see Cargo.toml -- nothing like `reqwest`/`ureq` is pulled
/// in anywhere else in this project either, and one fire-and-forget POST
/// with an empty body isn't worth adding one for). Doesn't wait for or
/// read the response -- graceful_shutdown_then_kill()'s own poll loop is
/// what actually waits on the *result* of this; this function only needs
/// the request to actually reach the server.
///
/// A connect failure (orchestrator already dead, port not open yet, or
/// launcher.local.txt/venv not set up so it never started at all) is
/// expected and handled fine by the poll-then-kill fallback above either
/// way -- silently returning here on any connect error is deliberate, not
/// an oversight.
fn request_orchestrator_shutdown() {
    let addr: SocketAddr = "127.0.0.1:8765"
        .parse()
        .expect("hardcoded loopback address is always valid");
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else {
        return;
    };
    let _ = stream.write_all(
        b"POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1:8765\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    );
}
