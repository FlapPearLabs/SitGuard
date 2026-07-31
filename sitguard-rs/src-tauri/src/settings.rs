//! Settings persistence.
//!
//! Loads/saves `sitguard.json` from the OS app-config directory. Holds the
//! user calibration [`Baseline`], the tunable [`Thresholds`], and the reminder
//! copy. The file is always written in full by the app, so `load` simply falls
//! back to [`Settings::default`] on any parse error.

use crate::posture::{Baseline, Thresholds};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::Manager;

/// Persisted application settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// User calibration captured when sitting properly.
    pub baseline: Baseline,
    /// Tunable judgment + reminder thresholds.
    pub thresholds: Thresholds,
    /// Break reminder title.
    pub reminder_title: String,
    /// Break reminder body; supports `{minutes}` and `{break}` placeholders.
    pub reminder_body: String,
    /// Posture-correction reminder title.
    pub posture_reminder_title: String,
    /// Posture-correction reminder body.
    pub posture_reminder_body: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            baseline: Baseline::default(),
            thresholds: Thresholds::default(),
            reminder_title: "SitGuard 提醒".to_string(),
            // Matches the original CLI's `.format(minutes=..., break=...)` copy.
            reminder_body:
                "你已经连续坐了 {minutes} 分钟，起来活动 {break} 分钟吧！"
                    .to_string(),
            posture_reminder_title: "SitGuard 姿态提醒".to_string(),
            posture_reminder_body: "注意保持良好坐姿，调整背部与头部位置。".to_string(),
        }
    }
}

fn config_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join("sitguard.json"))
}

/// Load settings; falls back to defaults if the file is missing or unreadable.
pub fn load(app: &tauri::AppHandle) -> Settings {
    if let Some(p) = config_path(app) {
        if let Ok(s) = std::fs::read_to_string(&p) {
            match serde_json::from_str::<Settings>(&s) {
                Ok(cfg) => return cfg,
                Err(e) => log::warn!("sitguard.json parse failed ({e}); using defaults"),
            }
        }
    }
    Settings::default()
}

/// Persist settings to disk, creating the parent directory if needed.
pub fn save(app: &tauri::AppHandle, s: &Settings) -> Result<(), String> {
    let p = config_path(app).ok_or("cannot resolve app config dir")?;
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let json = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    std::fs::write(&p, json).map_err(|e| e.to_string())?;
    Ok(())
}
