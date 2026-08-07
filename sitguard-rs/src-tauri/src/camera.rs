//! Camera capture module.
//!
//! Lessons learned from the Python version's black-screen incident:
//! - Never trust `open()` success alone — MSMF can "open" a camera and feed
//!   pure-black frames. Always warm up and verify frame brightness.
//! - Log every step so failures are diagnosable from user machines.

use base64::Engine;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType,
};
use nokhwa::Camera;
use crate::pose::{BodyPose, PoseDetector};
use crate::posture::{FaceObs, PostureInfo, PostureTracker};
use crate::settings::Settings;
use crate::sitting::SessionTracker;
use serde::Serialize;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri_plugin_notification::NotificationExt;

#[derive(Debug, Clone, Serialize)]
pub struct CameraInfo {
    pub index: u32,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProbeResult {
    pub index: u32,
    pub opened: bool,
    pub width: u32,
    pub height: u32,
    pub mean_brightness: f64,
    pub black_frame: bool,
    pub message: String,
}

/// List all cameras visible to the OS (MediaFoundation enumeration on Windows).
pub fn list_cameras() -> Vec<CameraInfo> {
    match nokhwa::query(ApiBackend::Auto) {
        Ok(devices) => devices
            .into_iter()
            .map(|d| CameraInfo {
                index: match d.index() {
                    CameraIndex::Index(i) => *i,
                    CameraIndex::String(_) => 0,
                },
                name: d.human_name(),
                description: d.description().to_string(),
            })
            .collect(),
        Err(e) => {
            log::warn!("camera query failed: {e}");
            Vec::new()
        }
    }
}

/// Open camera, read warm-up frames, verify the frames are not pure black.
/// Returns probe diagnostics — mirrors the v4 Python patch behaviour.
pub fn probe_camera(index: u32, warmup: usize) -> ProbeResult {
    let idx = CameraIndex::Index(index);
    let fmt = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    let mut cam = match Camera::new(idx, fmt) {
        Ok(c) => c,
        Err(e) => {
            return ProbeResult {
                index,
                opened: false,
                width: 0,
                height: 0,
                mean_brightness: 0.0,
                black_frame: false,
                message: format!("open failed: {e}"),
            }
        }
    };

    if let Err(e) = cam.open_stream() {
        return ProbeResult {
            index,
            opened: false,
            width: 0,
            height: 0,
            mean_brightness: 0.0,
            black_frame: false,
            message: format!("open_stream failed: {e}"),
        };
    }

    let res = cam.resolution();
    let mut last_mean = 0.0f64;
    for i in 0..warmup.max(1) {
        match cam.frame() {
            Ok(frame) => {
                if let Ok(decoded) = frame.decode_image::<RgbFormat>() {
                    let sum: u64 = decoded.as_raw().iter().map(|&b| b as u64).sum();
                    last_mean = sum as f64 / decoded.as_raw().len().max(1) as f64;
                    if last_mean > 1.0 {
                        // Non-black frame seen — camera genuinely works.
                        let _ = cam.stop_stream();
                        return ProbeResult {
                            index,
                            opened: true,
                            width: res.width(),
                            height: res.height(),
                            mean_brightness: last_mean,
                            black_frame: false,
                            message: format!("ok after {} warmup frame(s)", i + 1),
                        };
                    }
                }
            }
            Err(e) => {
                log::warn!("warmup frame {i} failed: {e}");
            }
        }
    }
    let _ = cam.stop_stream();

    ProbeResult {
        index,
        opened: true,
        width: res.width(),
        height: res.height(),
        mean_brightness: last_mean,
        black_frame: last_mean <= 1.0,
        message: if last_mean <= 1.0 {
            "camera opened but all warmup frames are black (firmware/privacy shutter?)".into()
        } else {
            "ok".into()
        },
    }
}

/// Shared capture state for the preview loop.
pub struct CaptureState {
    pub running: Arc<AtomicBool>,
    pub camera_index: Mutex<Option<u32>>,
}

impl Default for CaptureState {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            camera_index: Mutex::new(None),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FramePayload {
    pub jpeg_base64: String,
    pub width: u32,
    pub height: u32,
    pub mean_brightness: f64,
    pub seq: u64,
    /// YuNet detections in frame coords (mirrored/selfie space).
    pub faces: Vec<crate::detector::FaceDet>,
    /// Inference time of the last detection pass, milliseconds.
    pub detect_ms: f64,
    /// Presence + posture judgment for this frame.
    pub posture: PostureInfo,
    /// Full-body pose (MoveNet 17 keypoints) in frame coords, or None when the
    /// model is unavailable / not yet attempted.
    pub body: Option<BodyPose>,
}

/// Blocking capture loop; emits `camera-frame` events until `running` is false.
pub fn capture_loop(
    app: tauri::AppHandle,
    index: u32,
    running: Arc<AtomicBool>,
    tracker: Arc<Mutex<PostureTracker>>,
    session: Arc<Mutex<SessionTracker>>,
    settings: Arc<Mutex<Settings>>,
) -> Result<(), String> {
    use tauri::Emitter;

    // Fresh session: reset calibration/state (mirrors a fresh Python SittingDetector).
    if let Ok(mut t) = tracker.lock() {
        let th = t.thresholds;
        *t = PostureTracker::new(th);
    }

    let idx = CameraIndex::Index(index);
    let fmt = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    let mut cam = Camera::new(idx, fmt).map_err(|e| format!("open failed: {e}"))?;
    cam.open_stream().map_err(|e| format!("open_stream failed: {e}"))?;

    let mut seq: u64 = 0;
    // Built lazily on the first decoded frame (needs actual frame dims,
    // which may differ from cam.resolution()).
    let mut detector: Option<crate::detector::FaceDetector> = None;
    // MoveNet body pose (optional). Attempted once; if the model is missing
    // or fails to load we leave it None and keep using YuNet + posture logic.
    let mut pose_det: Option<PoseDetector> = None;
    let mut pose_tried: bool = false;

    while running.load(Ordering::Relaxed) {
        let frame = match cam.frame() {
            Ok(f) => f,
            Err(e) => {
                log::warn!("frame read failed: {e}");
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
        };
        let decoded = match frame.decode_image::<RgbFormat>() {
            Ok(d) => d,
            Err(e) => {
                log::warn!("decode failed: {e}");
                continue;
            }
        };
        // Mirror horizontally — selfie-style, matches the Python version
        // (cv2.flip(frame, 1)) so on-screen movement feels natural.
        let decoded = image::imageops::flip_horizontal(&decoded);

        let (w, h) = (decoded.width(), decoded.height());
        let sum: u64 = decoded.as_raw().iter().map(|&b| b as u64).sum();
        let mean = sum as f64 / decoded.as_raw().len().max(1) as f64;

        // Face detection (YuNet). Detector is shape-specialized; rebuild if
        // the camera ever changes resolution mid-stream.
        if detector.is_none() {
            match crate::detector::FaceDetector::new(w, h) {
                Ok(d) => detector = Some(d),
                Err(e) => log::error!("YuNet init failed: {e}"),
            }
        }
        let mut faces = Vec::new();
        let mut detect_ms = 0.0f64;
        if let Some(det) = detector.as_ref() {
            let t0 = std::time::Instant::now();
            match det.detect(&decoded) {
                Ok(f) => faces = f,
                Err(e) => log::warn!("detect failed: {e}"),
            }
            detect_ms = t0.elapsed().as_secs_f64() * 1000.0;
        }

        // Body pose (MoveNet), optional. Lazy one-shot init; a missing model
        // disables the skeleton overlay without affecting face/posture logic.
        if !pose_tried {
            pose_tried = true;
            match crate::pose::find_model("movenet_lightning.onnx") {
                Some(p) => match PoseDetector::try_new(&p, w, h) {
                    Ok(d) => pose_det = Some(d),
                    Err(e) => log::warn!("MoveNet unavailable, body pose disabled: {e}"),
                },
                None => log::info!("MoveNet model not found; body pose overlay disabled"),
            }
        }
        let body: Option<BodyPose> = if let Some(pd) = pose_det.as_ref() {
            match pd.detect(&decoded) {
                Ok(b) => Some(b),
                Err(e) => {
                    log::warn!("pose detect failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Posture / presence judgment. Pick the highest-confidence face and
        // build a normalized observation; `None` when no face was detected.
        let obs: Option<FaceObs> = faces
            .iter()
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|f| {
                let fw = w as f32;
                let fh = h as f32;
                FaceObs {
                    center_x_norm: (f.x + f.w / 2.0) / fw,
                    center_y_norm: (f.y + f.h / 2.0) / fh,
                    size_ratio: (f.w * f.h) / (fw * fh),
                    yaw_deg: f.yaw_deg,
                    pitch_deg: f.pitch_deg,
                    score: f.score,
                }
            });
        let posture_info = tracker
            .lock()
            .map(|mut t| t.process(obs.as_ref(), body.as_ref()))
            .unwrap_or(PostureInfo {
                presence: crate::posture::PresenceState::NotPresent,
                posture: crate::posture::PostureState::Unknown,
                posture_score: 0.0,
                baseline_calibrated: false,
                focused_since: 0.0,
                body_present: false,
                shoulder_tilt_deg: 0.0,
                spine_lateral_deg: 0.0,
                head_drop_ratio: 0.0,
                torso_compress_ratio: 0.0,
            });

        // Sitting-session timing + reminders (M4). Tick the session with the
        // per-frame presence/posture, then fire OS notifications when either
        // the continuous-sitting or bad-posture gate opens.
        let now = Instant::now();
        // NOTE: lock ordering must be `settings` -> `session` here to match
        // `update_settings` (settings -> tracker -> session) in lib.rs. A
        // reversed order (session -> settings) would be a lock-ordering
        // inversion and could deadlock the preview thread against the main
        // thread when settings are saved mid-preview.
        if let (Ok(s), Ok(mut sess)) = (settings.lock(), session.lock()) {
            sess.tick(posture_info.presence, posture_info.posture, now);
            let th = sess.thresholds();
            if sess.should_remind_break(now) {
                let body = s
                    .reminder_body
                    .replace("{minutes}", &format!("{:.0}", th.sitting_threshold_minutes))
                    .replace("{break}", &format!("{:.0}", th.break_minutes));
                let _ = app
                    .notification()
                    .builder()
                    .title(&s.reminder_title)
                    .body(body)
                    .show();
                sess.mark_break_reminded(now);
            }
            if sess.should_remind_posture(now) {
                let _ = app
                    .notification()
                    .builder()
                    .title(&s.posture_reminder_title)
                    .body(s.posture_reminder_body.clone())
                    .show();
                sess.mark_posture_reminded(now);
            }
        }

        // Encode to JPEG (quality 70) for transport to the webview.
        let mut buf = Cursor::new(Vec::new());
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 70);
        if enc
            .encode(decoded.as_raw(), w, h, image::ExtendedColorType::Rgb8)
            .is_err()
        {
            continue;
        }

        seq += 1;
        let payload = FramePayload {
            jpeg_base64: base64::engine::general_purpose::STANDARD.encode(buf.into_inner()),
            width: w,
            height: h,
            mean_brightness: (mean * 10.0).round() / 10.0,
            seq,
            faces,
            detect_ms: (detect_ms * 10.0).round() / 10.0,
            posture: posture_info,
            body,
        };
        let _ = app.emit("camera-frame", &payload);

        // ~15 fps cap to keep the event channel light.
        std::thread::sleep(std::time::Duration::from_millis(66));
    }

    let _ = cam.stop_stream();
    Ok(())
}
