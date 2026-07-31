mod camera;
mod detector;
mod pose;
mod posture;
mod settings;
mod sitting;

use camera::{CameraInfo, CaptureState, ProbeResult};
use posture::PostureTracker;
use settings::Settings;
use sitting::{SessionStats, SessionTracker};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager, State};
use std::time::Instant;

/// List available cameras.
#[tauri::command]
fn list_cameras() -> Vec<CameraInfo> {
    camera::list_cameras()
}

/// Probe a camera: open, warm up, verify non-black frames.
#[tauri::command]
fn probe_camera(index: u32) -> ProbeResult {
    camera::probe_camera(index, 3)
}

/// Start streaming preview frames as `camera-frame` events.
#[tauri::command]
fn start_preview(
    app: tauri::AppHandle,
    state: State<'_, CaptureState>,
    tracker: State<'_, Arc<Mutex<PostureTracker>>>,
    session: State<'_, Arc<Mutex<SessionTracker>>>,
    settings_state: State<'_, Arc<Mutex<Settings>>>,
    index: u32,
) -> Result<(), String> {
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("preview already running".into());
    }
    *state.camera_index.lock().unwrap() = Some(index);

    let running = state.running.clone();
    let tracker = tracker.inner().clone();
    let session = session.inner().clone();
    let settings = settings_state.inner().clone();
    std::thread::spawn(move || {
        if let Err(e) =
            camera::capture_loop(app, index, running.clone(), tracker, session, settings)
        {
            log::error!("capture loop ended with error: {e}");
        }
        running.store(false, Ordering::SeqCst);
    });
    Ok(())
}

/// Stop the preview loop.
#[tauri::command]
fn stop_preview(
    app: tauri::AppHandle,
    state: State<'_, CaptureState>,
    settings_state: State<'_, Arc<Mutex<Settings>>>,
    tracker: State<'_, Arc<Mutex<PostureTracker>>>,
) {
    state.running.store(false, Ordering::SeqCst);
    // Persist any calibration captured this session.
    if let (Ok(mut s), Ok(t)) = (settings_state.inner().lock(), tracker.inner().lock()) {
        s.baseline = t.baseline;
        let _ = settings::save(&app, &s);
    }
}

/// Force the posture baseline to re-snapshot on the next frame with a face.
#[tauri::command]
fn calibrate_baseline(tracker: State<'_, Arc<Mutex<PostureTracker>>>) {
    if let Ok(mut t) = tracker.inner().lock() {
        t.force_calibrate_next();
    }
}

/// Return the current persisted settings (baseline, thresholds, reminder copy).
#[tauri::command]
fn get_settings(state: State<'_, Arc<Mutex<Settings>>>) -> Settings {
    state.inner().lock().unwrap().clone()
}

/// Update settings: thresholds + reminder copy. The live baseline (which may
/// have been auto-calibrated) is preserved; thresholds propagate to both the
/// posture and session trackers and are persisted to disk.
#[tauri::command]
fn update_settings(
    app: tauri::AppHandle,
    state: State<'_, Arc<Mutex<Settings>>>,
    tracker: State<'_, Arc<Mutex<PostureTracker>>>,
    session: State<'_, Arc<Mutex<SessionTracker>>>,
    new: Settings,
) -> Result<(), String> {
    let mut st = state.inner().lock().unwrap();
    let mut tr = tracker.inner().lock().unwrap();
    st.baseline = tr.baseline; // keep live calibration
    st.thresholds = new.thresholds;
    st.reminder_title = new.reminder_title;
    st.reminder_body = new.reminder_body;
    st.posture_reminder_title = new.posture_reminder_title;
    st.posture_reminder_body = new.posture_reminder_body;
    tr.thresholds = new.thresholds;
    session.inner().lock().unwrap().set_thresholds(new.thresholds);
    settings::save(&app, &st)
}

/// Return the current session statistics for the report page.
#[tauri::command]
fn get_session_stats(session: State<'_, Arc<Mutex<SessionTracker>>>) -> SessionStats {
    session.inner().lock().unwrap().stats()
}

/// Snooze reminders for `minutes` (e.g. user clicked "稍后提醒").
#[tauri::command]
fn pause_session(session: State<'_, Arc<Mutex<SessionTracker>>>, minutes: f64) {
    session.inner().lock().unwrap().pause(minutes, Instant::now());
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            let handle = app.app_handle();
            let s = settings::load(&handle);
            app.manage(CaptureState::default());
            app.manage(Arc::new(Mutex::new(PostureTracker::with_baseline(
                s.baseline,
                s.thresholds,
            ))));
            app.manage(Arc::new(Mutex::new(SessionTracker::new(s.thresholds))));
            app.manage(Arc::new(Mutex::new(s)));

            // System tray with a context menu (show/hide, start/stop, quit).
            #[cfg(desktop)]
            {
                use tauri::menu::{Menu, MenuItem};
                use tauri::tray::{
                    MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent,
                };

                let show =
                    MenuItem::with_id(app, "show", "显示窗口", true, None::<&str>).unwrap();
                let hide =
                    MenuItem::with_id(app, "hide", "隐藏窗口", true, None::<&str>).unwrap();
                let start =
                    MenuItem::with_id(app, "start", "开始监控", true, None::<&str>).unwrap();
                let stop =
                    MenuItem::with_id(app, "stop", "停止监控", true, None::<&str>).unwrap();
                let quit =
                    MenuItem::with_id(app, "quit", "退出", true, None::<&str>).unwrap();
                let menu = Menu::with_items(app, &[&show, &hide, &start, &stop, &quit]).unwrap();

                TrayIconBuilder::new()
                    .icon(app.default_window_icon().unwrap().clone())
                    .tooltip("SitGuard")
                    .menu(&menu)
                    .on_menu_event(|app, event| match event.id().as_ref() {
                        "show" => {
                            if let Some(w) = app.get_webview_window("main") {
                                let _ = w.show();
                                let _ = w.set_focus();
                            }
                        }
                        "hide" => {
                            if let Some(w) = app.get_webview_window("main") {
                                let _ = w.hide();
                            }
                        }
                        "start" => {
                            let _ = app.emit("tray-command", "start");
                        }
                        "stop" => {
                            let _ = app.emit("tray-command", "stop");
                        }
                        "quit" => app.exit(0),
                        _ => {}
                    })
                    .on_tray_icon_event(|tray, event| {
                        if let TrayIconEvent::Click {
                            button: MouseButton::Left,
                            button_state: MouseButtonState::Up,
                            ..
                        } = event
                        {
                            if let Some(w) = tray.app_handle().get_webview_window("main") {
                                let _ = w.show();
                                let _ = w.set_focus();
                            }
                        }
                    })
                    .build(app)
                    .expect("failed to build tray icon");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_cameras,
            probe_camera,
            start_preview,
            stop_preview,
            calibrate_baseline,
            get_settings,
            update_settings,
            get_session_stats,
            pause_session
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
