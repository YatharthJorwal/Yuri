// Prevents an extra console window from popping up alongside Luna on
// Windows in release builds. Do not remove.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    luna_lib::run();
}
