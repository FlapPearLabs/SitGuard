//! Sitting-posture judgment: presence state machine + face-geometry fusion +
//! **full-body skeleton fusion** + time smoothing.
//!
//! Two independent perception streams feed the classifier:
//!   1. YuNet face geometry (centre drift, size = distance, yaw/pitch) — the
//!      original M2–M4 signal.
//!   2. MoveNet full-body keypoints (17 COCO points) — converted into
//!      body-angle metrics (shoulder tilt, spine lateral bend, head drop,
//!      torso compression). This is the M5 upgrade that, until now, only fed
//!      the on-screen skeleton overlay and never the actual posture judgment.
//!
//! Both streams are fused via "deviation from a self-calibrated baseline", so
//! the same classifier adapts to different body shapes / camera angles. When
//! the body model is unavailable the skeleton branch degrades to neutral and
//! the face-geometry branch alone drives the verdict (backward compatible).
//!
//! Time smoothing (the other half of this module's job):
//!   - A `POSTURE_HOLD_SECS` hysteresis on the discrete posture *label* absorbs
//!     the per-frame jitter of YuNet's bounding box, which used to make the
//!     posture tag flip every frame.
//!   - EMA smoothing on the continuous `posture_score` and the four body-angle
//!     metrics so the numbers the UI shows do not flicker.
//!
//! Ported/extended from the recovered Python reference `recovered_reference/detector.py`.

use crate::pose::BodyPose;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// Seconds a candidate posture label must persist before it becomes the
/// displayed label. Absorbs YuNet bbox jitter (a box wobble must not flip the
/// verdict every frame). The M4 reminder layer has its own 15s dwell on top.
const POSTURE_HOLD_SECS: f64 = 4.0;
/// EMA factor for the continuous posture score (higher = more responsive).
const SCORE_ALPHA: f32 = 0.3;
/// EMA factor for the body-angle metrics shown in the UI.
const BODY_ALPHA: f32 = 0.25;

/// Deviation-from-baseline thresholds for the skeleton branch. Units match the
/// metrics: ratios are in shoulder-widths, angles in degrees.
const HEAD_DROP_DEV: f32 = 0.25; // head sank >25% of shoulder width -> HeadDown
const TORSO_DEV: f32 = 0.20; // torso compressed -> Slouching (hunched)
const SPINE_LAT_DEV_DEG: f32 = 10.0; // lateral spine bend -> Slouching/Leaning
const SHOULDER_TILT_DEV_DEG: f32 = 10.0; // shoulder line tilt -> Leaning

/// One frame's face observation, already normalised to 0..1 by the caller.
/// `None` (no face) is represented by passing `Option::None` to [`PostureTracker::process`].
#[derive(Clone, Copy, Debug)]
pub struct FaceObs {
    /// Face centre x in frame, 0..1.
    pub center_x_norm: f32,
    /// Face centre y in frame, 0..1.
    pub center_y_norm: f32,
    /// bbox area / frame area.
    pub size_ratio: f32,
    /// Head yaw in degrees (simplified geometric estimate from M2 detector).
    pub yaw_deg: f32,
    /// Head pitch in degrees.
    pub pitch_deg: f32,
    /// Detection confidence 0..1 (carried for completeness; signals use geometry).
    #[allow(dead_code)]
    pub score: f32,
}

/// Where the user is, relative to "sitting at the desk, being guarded".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PresenceState {
    NotPresent,
    PresentIdle,
    PresentFocused,
}

/// Sitting-posture classification, computed every frame from face geometry and
/// (when available) full-body skeleton angles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PostureState {
    Unknown,
    Good,
    HeadDown,
    Slouching,
    LeanBack,
    Leaning,
}

/// Severity ranking for fusing the face and body branches — the worse (more
/// harmful) verdict wins, so one bad signal is enough to flag the posture.
fn severity(p: PostureState) -> u8 {
    match p {
        PostureState::Unknown => 0,
        PostureState::Good => 1,
        PostureState::Leaning => 2,
        PostureState::LeanBack => 3,
        PostureState::Slouching => 4,
        PostureState::HeadDown => 5,
    }
}

/// Return the more severe of two posture states.
fn worse(a: PostureState, b: PostureState) -> PostureState {
    if severity(a) >= severity(b) {
        a
    } else {
        b
    }
}

/// Per-user calibration captured when the user sits properly.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Baseline {
    pub face_center_x: f32,
    pub face_center_y: f32,
    pub face_size_ratio: f32,
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    /// Unix seconds when calibrated; 0.0 means "not calibrated yet".
    pub calibrated_at: f64,
    /// Self-calibrated skeleton metrics (filled only when a confident body
    /// pose was visible during calibration). Lets the body branch judge by
    /// deviation from the user's own neutral pose rather than absolute values.
    #[serde(default)]
    pub body_shoulder_tilt_deg: Option<f32>,
    #[serde(default)]
    pub body_spine_lateral_deg: Option<f32>,
    #[serde(default)]
    pub body_head_drop_ratio: Option<f32>,
    #[serde(default)]
    pub body_torso_compress_ratio: Option<f32>,
}

impl Default for Baseline {
    fn default() -> Self {
        Baseline {
            face_center_x: 0.5,
            face_center_y: 0.45,
            face_size_ratio: 0.25,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            calibrated_at: 0.0,
            body_shoulder_tilt_deg: None,
            body_spine_lateral_deg: None,
            body_head_drop_ratio: None,
            body_torso_compress_ratio: None,
        }
    }
}

impl Baseline {
    pub fn is_calibrated(&self) -> bool {
        self.calibrated_at > 0.0
    }
}

/// Tunable thresholds (defaults verbatim from `sitguard/config.pyc`).
// Some fields (sitting/break/reminder/posture-dwell/cooldown, position_tolerance)
// are consumed by the M4 reminder/notification layer, not the M3 classifier.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Thresholds {
    pub sitting_threshold_minutes: f64,
    pub break_minutes: f64,
    pub reminder_cooldown_minutes: f64,
    pub position_tolerance: f32,
    pub size_tolerance: f32,
    pub yaw_tolerance_deg: f32,
    pub pitch_tolerance_deg: f32,
    pub present_dwell_seconds: f64,
    pub focused_dwell_seconds: f64,
    pub absent_timeout_seconds: f64,
    pub leave_reset_seconds: f64,
    pub posture_y_drift: f32,
    pub posture_x_drift: f32,
    pub posture_size_close: f32,
    pub posture_pitch_down: f32,
    pub posture_yaw_turned: f32,
    pub posture_dwell_seconds: f64,
    pub posture_cooldown_minutes: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            sitting_threshold_minutes: 45.0,
            break_minutes: 5.0,
            reminder_cooldown_minutes: 5.0,
            position_tolerance: 0.15,
            size_tolerance: 0.4,
            yaw_tolerance_deg: 30.0,
            pitch_tolerance_deg: 25.0,
            present_dwell_seconds: 2.0,
            focused_dwell_seconds: 3.0,
            absent_timeout_seconds: 10.0,
            leave_reset_seconds: 300.0,
            posture_y_drift: 0.08,
            posture_x_drift: 0.12,
            posture_size_close: 0.15,
            posture_pitch_down: 20.0,
            posture_yaw_turned: 25.0,
            posture_dwell_seconds: 15.0,
            posture_cooldown_minutes: 5.0,
        }
    }
}

/// Weighted signals for the fusion vote (mirrors `PresenceSignals`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PresenceSignals {
    pub face_present: f32,
    pub face_size_ok: f32,
    pub face_stable: f32,
    pub head_facing: f32,
    pub recent_motion: f32,
}

impl PresenceSignals {
    /// Weighted sum; weights sum to 1.0 (0.30/0.25/0.15/0.20/0.10).
    pub fn total(&self) -> f32 {
        0.30 * self.face_present
            + 0.25 * self.face_size_ok
            + 0.15 * self.face_stable
            + 0.20 * self.head_facing
            + 0.10 * self.recent_motion
    }
}

/// Body-angle metrics derived from the 17 MoveNet COCO keypoints, in frame
/// pixel space (y points down). `confident` is false when the spine keypoints
/// (nose, both shoulders, both hips) are not all reliably detected, in which
/// case the body branch is skipped.
#[derive(Clone, Copy, Debug, Default)]
pub struct BodyAngles {
    /// Shoulder-line tilt from horizontal (deg). 0 = level. Positive = right
    /// shoulder lower (image is selfie-mirrored, sign is cosmetic).
    pub shoulder_tilt_deg: f32,
    /// Lateral spine bend: horizontal offset of the hip midpoint from the
    /// shoulder midpoint, relative to the vertical drop (deg). 0 = vertical.
    pub spine_lateral_deg: f32,
    /// Head drop: how far the nose sits above the shoulder midline, normalised
    /// by shoulder width. Larger = head higher (upright). Small/negative =
    /// head sunk toward the shoulders (slouching / looking down).
    pub head_drop_ratio: f32,
    /// Torso compression: vertical shoulder→hip distance normalised by shoulder
    /// width. Smaller than baseline = chest collapsed forward (hunched).
    pub torso_compress_ratio: f32,
    pub confident: bool,
}

fn midpoint(a: [f32; 3], b: [f32; 3]) -> [f32; 2] {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
}

/// Derive body-angle metrics from 17 keypoints `[x_px, y_px, score]`.
/// Returns `None` unless nose + both shoulders + both hips all clear the
/// keypoint threshold, because every metric here depends on at least the spine.
pub fn compute_body_angles(kpts: &[[f32; 3]]) -> Option<BodyAngles> {
    if kpts.len() < 17 {
        return None;
    }
    let nose = kpts[0];
    let lsh = kpts[5];
    let rsh = kpts[6];
    let lhip = kpts[11];
    let rhip = kpts[12];
    let confident = nose[2] > 0.3
        && lsh[2] > 0.3
        && rsh[2] > 0.3
        && lhip[2] > 0.3
        && rhip[2] > 0.3;
    if !confident {
        return None;
    }

    let sh_mid = midpoint(lsh, rsh);
    let hip_mid = midpoint(lhip, rhip);
    let shoulder_width = (rsh[0] - lsh[0]).abs().max(1.0);

    // Shoulder line tilt from horizontal.
    let shoulder_tilt_deg =
        (rsh[1] - lsh[1]).atan2((rsh[0] - lsh[0]).abs()) * 180.0 / std::f32::consts::PI;

    // Lateral spine bend: how far the hips drift sideways from the shoulders.
    let dx = hip_mid[0] - sh_mid[0];
    let dy = (hip_mid[1] - sh_mid[1]).abs().max(1.0);
    let spine_lateral_deg = dx.atan2(dy) * 180.0 / std::f32::consts::PI;

    // Head drop: nose height above the shoulder midline (positive = above).
    let head_drop_ratio = (sh_mid[1] - nose[1]) / shoulder_width;

    // Torso compression: shoulder→hip vertical distance in shoulder widths.
    let torso_compress_ratio = (hip_mid[1] - sh_mid[1]) / shoulder_width;

    Some(BodyAngles {
        shoulder_tilt_deg,
        spine_lateral_deg,
        head_drop_ratio,
        torso_compress_ratio,
        confident: true,
    })
}

/// Result returned to the caller each frame.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PostureInfo {
    pub presence: PresenceState,
    pub posture: PostureState,
    pub posture_score: f32,
    pub baseline_calibrated: bool,
    /// Unix seconds the current focused streak started, or 0.0 if not focused.
    pub focused_since: f64,
    /// Whether a usable full-body skeleton was available this frame.
    #[serde(default)]
    pub body_present: bool,
    /// Smoothed skeleton metrics (degrees / shoulder-width ratios).
    #[serde(default)]
    pub shoulder_tilt_deg: f32,
    #[serde(default)]
    pub spine_lateral_deg: f32,
    #[serde(default)]
    pub head_drop_ratio: f32,
    #[serde(default)]
    pub torso_compress_ratio: f32,
}

/// Running tracker: holds calibration, thresholds and the presence state machine.
pub struct PostureTracker {
    pub baseline: Baseline,
    pub thresholds: Thresholds,

    recent_centers: VecDeque<(f32, f32)>, // ~2s at 15fps
    last_face_seen: f64,
    last_motion_time: f64,

    state: PresenceState,
    state_entered_at: f64,
    focused_started_at: Option<f64>,

    last_process_at: f64,
    calib_accum_seconds: f32,
    force_calibrate_next: bool,

    // --- time smoothing state (the fix for per-frame label flicker) ---
    /// Smoothed posture label actually reported to the UI.
    label: PostureState,
    /// Candidate label during the hysteresis dwell window.
    pending_label: PostureState,
    pending_since: f64,
    /// EMA of the continuous score.
    ema_score: Option<f32>,
    /// EMA of the four body-angle metrics.
    ema_shoulder_tilt: Option<f32>,
    ema_spine_lateral: Option<f32>,
    ema_head_drop: Option<f32>,
    ema_torso_compress: Option<f32>,
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Pure hysteresis step: decide the smoothed label given a fresh candidate.
/// Extracted so it can be unit-tested without a wall clock.
fn decide_label(
    current: PostureState,
    pending: PostureState,
    pending_since: f64,
    candidate: PostureState,
    now: f64,
    hold: f64,
) -> (PostureState, PostureState, f64) {
    if candidate != current {
        // First real verdict: `current` is still the uninitialised Unknown,
        // so adopt the candidate immediately rather than waiting out a dwell.
        if current == PostureState::Unknown {
            return (candidate, candidate, now);
        }
        let (p, ps) = if pending != candidate {
            (candidate, now)
        } else {
            (pending, pending_since)
        };
        if now - ps >= hold {
            (candidate, candidate, now)
        } else {
            (current, p, ps)
        }
    } else {
        (current, candidate, now)
    }
}

fn ema(prev: Option<f32>, x: f32, alpha: f32) -> Option<f32> {
    Some(match prev {
        None => x,
        Some(v) => v + alpha * (x - v),
    })
}

impl PostureTracker {
    pub fn new(thresholds: Thresholds) -> Self {
        let now = now_secs();
        PostureTracker {
            baseline: Baseline::default(),
            thresholds,
            recent_centers: VecDeque::with_capacity(30),
            last_face_seen: 0.0,
            last_motion_time: now,
            state: PresenceState::NotPresent,
            state_entered_at: now,
            focused_started_at: None,
            last_process_at: now,
            calib_accum_seconds: 0.0,
            force_calibrate_next: false,
            label: PostureState::Unknown,
            pending_label: PostureState::Unknown,
            pending_since: now,
            ema_score: None,
            ema_shoulder_tilt: None,
            ema_spine_lateral: None,
            ema_head_drop: None,
            ema_torso_compress: None,
        }
    }

    // Used by tests and the M4 config layer.
    #[allow(dead_code)]
    pub fn with_baseline(baseline: Baseline, thresholds: Thresholds) -> Self {
        let mut t = Self::new(thresholds);
        t.baseline = baseline;
        t
    }

    /// Ask the tracker to re-snapshot the baseline on the next frame that has a face.
    pub fn force_calibrate_next(&mut self) {
        self.force_calibrate_next = true;
    }

    #[allow(dead_code)]
    pub fn baseline(&self) -> &Baseline {
        &self.baseline
    }

    #[allow(dead_code)]
    pub fn presence(&self) -> PresenceState {
        self.state
    }

    fn compute_signals(&self, obs: &FaceObs) -> PresenceSignals {
        let th = &self.thresholds;
        let mut sigs = PresenceSignals::default();
        let now = now_secs();

        // 1. Face present
        sigs.face_present = 1.0;

        // 2. Face size within tolerance of baseline
        if self.baseline.is_calibrated() && self.baseline.face_size_ratio > 0.0 {
            let ratio = obs.size_ratio / self.baseline.face_size_ratio;
            sigs.face_size_ok =
                (1.0 - (1.0 - ratio).abs() / th.size_tolerance).max(0.0);
        } else {
            // No baseline yet — accept any reasonable size
            sigs.face_size_ok = if obs.size_ratio > 0.02 && obs.size_ratio < 0.6 {
                1.0
            } else {
                0.0
            };
        }

        // 3. Position stability (low variance over last ~2s)
        if self.recent_centers.len() >= 10 {
            let (mut sx, mut sy, mut sx2, mut sy2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for &(x, y) in &self.recent_centers {
                let x = x as f64;
                let y = y as f64;
                sx += x;
                sy += y;
                sx2 += x * x;
                sy2 += y * y;
            }
            let n = self.recent_centers.len() as f64;
            let var_x = sx2 / n - (sx / n) * (sx / n);
            let var_y = sy2 / n - (sy / n) * (sy / n);
            let var = ((var_x + var_y) / 2.0) as f32;
            // var < 0.0005 -> ~1.0 ; var > 0.01 -> 0.0 (matches reference formula)
            sigs.face_stable = (1.0 - var / 0.01).max(0.0);
        }

        // 4. Head facing (yaw + pitch within tolerance)
        let yaw_ok = obs.yaw_deg.abs() < th.yaw_tolerance_deg;
        let pitch_ok = obs.pitch_deg.abs() < th.pitch_tolerance_deg;
        sigs.head_facing = if yaw_ok && pitch_ok { 1.0 } else { 0.0 };

        // 5. Recent motion (within 60s)
        sigs.recent_motion = if (now - self.last_motion_time) < 60.0 {
            1.0
        } else {
            0.0
        };

        sigs
    }

    fn update_state(&mut self, obs: Option<&FaceObs>) {
        let th = &self.thresholds;
        let now = now_secs();
        let face_seen = obs.is_some();

        let sigs = match obs {
            Some(o) => self.compute_signals(o),
            None => PresenceSignals::default(),
        };
        let score = sigs.total();

        let previous = self.state;
        let mut next_state = previous;
        let dwell = now - self.state_entered_at;

        if !face_seen {
            if (now - self.last_face_seen) > th.absent_timeout_seconds {
                next_state = PresenceState::NotPresent;
            }
        } else if score >= 0.70 {
            next_state = PresenceState::PresentFocused;
        } else if score >= 0.40 {
            next_state = PresenceState::PresentIdle;
        }

        // Hysteresis: state must hold for the required dwell before transition.
        if next_state != previous {
            let required = if next_state == PresenceState::PresentFocused {
                th.focused_dwell_seconds
            } else if next_state == PresenceState::PresentIdle
                && previous == PresenceState::NotPresent
            {
                th.present_dwell_seconds
            } else {
                0.5 // fast exit
            };

            if dwell >= required {
                self.state = next_state;
                self.state_entered_at = now;
                if next_state == PresenceState::PresentFocused
                    && self.focused_started_at.is_none()
                {
                    self.focused_started_at = Some(now);
                }
            }
        }

        // Reset focused streak if we leave focused for > leave_reset_seconds.
        if let Some(_) = self.focused_started_at {
            if self.state != PresenceState::PresentFocused
                && (now - self.state_entered_at) > th.leave_reset_seconds
            {
                self.focused_started_at = None;
            }
        }
    }

    /// Face-geometry-only posture verdict (the original M2–M4 classifier).
    /// Returns `Unknown` until the baseline is calibrated.
    fn compute_posture(&self, obs: &FaceObs) -> PostureState {
        if !self.baseline.is_calibrated() {
            return PostureState::Unknown;
        }
        let th = &self.thresholds;
        let bl = &self.baseline;

        let dx = obs.center_x_norm - bl.face_center_x;
        let dy = obs.center_y_norm - bl.face_center_y;
        let x_drift = dx.abs();
        let y_drift = dy.abs();

        let size_ratio = if bl.face_size_ratio > 0.0 {
            obs.size_ratio / bl.face_size_ratio
        } else {
            1.0
        };

        let pitch = obs.pitch_deg;
        let yaw = obs.yaw_deg;

        // Priority: head_down > slouching > lean_back > leaning > good.
        // NOTE: size-based slouch/lean_back here is a weak proxy (distance
        // change). The skeleton branch below overrides it with a real torso
        // angle when available.
        if pitch > th.posture_pitch_down {
            return PostureState::HeadDown;
        }
        if size_ratio > 1.0 + th.posture_size_close {
            return PostureState::Slouching;
        }
        if size_ratio < 1.0 - th.posture_size_close {
            return PostureState::LeanBack;
        }
        if x_drift > th.posture_x_drift || y_drift > th.posture_y_drift {
            return PostureState::Leaning;
        }
        if yaw.abs() > th.posture_yaw_turned {
            return PostureState::Leaning;
        }
        PostureState::Good
    }

    /// Skeleton-based posture verdict from deviation of the current body angles
    /// from the self-calibrated baseline angles. Returns `Good` (neutral) when
    /// the body pose is absent or the baseline has no body reference, so the
    /// face branch alone decides in that case.
    fn compute_body_posture(&self, angles: Option<&BodyAngles>) -> PostureState {
        let bl = &self.baseline;
        let Some(a) = angles else {
            return PostureState::Good;
        };
        let (Some(b_tilt), Some(b_lat), Some(b_drop), Some(b_comp)) = (
            bl.body_shoulder_tilt_deg,
            bl.body_spine_lateral_deg,
            bl.body_head_drop_ratio,
            bl.body_torso_compress_ratio,
        ) else {
            return PostureState::Good;
        };

        let d_drop = b_drop - a.head_drop_ratio; // head sank relative to baseline
        let d_comp = b_comp - a.torso_compress_ratio; // torso collapsed forward
        let d_lat = (a.spine_lateral_deg - b_lat).abs();
        let d_tilt = (a.shoulder_tilt_deg - b_tilt).abs();

        if d_drop > HEAD_DROP_DEV {
            return PostureState::HeadDown;
        }
        if d_comp > TORSO_DEV || d_lat > SPINE_LAT_DEV_DEG {
            return PostureState::Slouching;
        }
        if d_tilt > SHOULDER_TILT_DEV_DEG {
            return PostureState::Leaning;
        }
        PostureState::Good
    }

    fn posture_score(state: PostureState) -> f32 {
        match state {
            PostureState::Good => 100.0,
            PostureState::HeadDown => 40.0,
            PostureState::Slouching => 50.0,
            PostureState::Leaning => 65.0,
            PostureState::LeanBack => 70.0,
            PostureState::Unknown => 0.0,
        }
    }

    fn snapshot_baseline(&mut self, obs: &FaceObs, angles: Option<&BodyAngles>) {
        let now = now_secs();
        self.baseline = Baseline {
            face_center_x: obs.center_x_norm,
            face_center_y: obs.center_y_norm,
            face_size_ratio: obs.size_ratio,
            yaw_deg: obs.yaw_deg,
            pitch_deg: obs.pitch_deg,
            calibrated_at: now,
            body_shoulder_tilt_deg: angles.map(|a| a.shoulder_tilt_deg),
            body_spine_lateral_deg: angles.map(|a| a.spine_lateral_deg),
            body_head_drop_ratio: angles.map(|a| a.head_drop_ratio),
            body_torso_compress_ratio: angles.map(|a| a.torso_compress_ratio),
        };
    }

    /// Feed one frame. `obs` is `None` when no face was detected; `body` is
    /// `None` when the body-pose model is unavailable or produced no confident
    /// skeleton this frame.
    pub fn process(
        &mut self,
        obs: Option<&FaceObs>,
        body: Option<&BodyPose>,
    ) -> PostureInfo {
        let now = now_secs();
        let dt = (now - self.last_process_at) as f32;
        self.last_process_at = now;

        let body_angles = body.and_then(|b| compute_body_angles(&b.keypoints));

        match obs {
            Some(o) => {
                self.recent_centers.push_back((o.center_x_norm, o.center_y_norm));
                if self.recent_centers.len() > 30 {
                    self.recent_centers.pop_front();
                }
                self.last_face_seen = now;

                // Cheap motion proxy: centre moved a little -> motion
                if self.recent_centers.len() >= 2 {
                    let n = self.recent_centers.len();
                    let cur = self.recent_centers[n - 1];
                    let prev = self.recent_centers[n - 2];
                    let dx = cur.0 - prev.0;
                    let dy = cur.1 - prev.1;
                    if (dx * dx + dy * dy).sqrt() > 0.005 {
                        self.last_motion_time = now;
                    }
                }
            }
            None => {
                self.recent_centers.clear();
            }
        }

        // Auto-calibrate: once focused + stable for >= 2s, snapshot the
        // baseline (face + skeleton). Also snapshots on force_calibrate_next.
        if obs.is_some()
            && self.state == PresenceState::PresentFocused
            && !self.baseline.is_calibrated()
        {
            let stable = self.compute_signals(obs.unwrap()).face_stable;
            if stable >= 0.7 || self.force_calibrate_next {
                self.calib_accum_seconds += dt;
                if self.calib_accum_seconds >= 2.0 || self.force_calibrate_next {
                    self.snapshot_baseline(obs.unwrap(), body_angles.as_ref());
                    self.calib_accum_seconds = 0.0;
                }
            } else {
                self.calib_accum_seconds = 0.0;
            }
        } else {
            self.calib_accum_seconds = 0.0;
        }
        self.force_calibrate_next = false;

        self.update_state(obs);

        // Fuse the two branches. Until the baseline is calibrated we cannot
        // judge at all (the face branch returns Unknown and the body branch has
        // no reference), so the verdict stays Unknown. Once calibrated, the body
        // branch only fires when both the current frame and the baseline have a
        // confident skeleton; otherwise it stays neutral and the face branch
        // alone decides.
        let face_p = match obs {
            Some(o) => self.compute_posture(o),
            None => PostureState::Unknown,
        };
        let body_p = self.compute_body_posture(body_angles.as_ref());
        let candidate = if !self.baseline.is_calibrated() {
            PostureState::Unknown
        } else {
            match obs {
                Some(_) => worse(face_p, body_p),
                None => PostureState::Unknown,
            }
        };

        // Hysteresis on the discrete label (absorbs bbox jitter).
        let (label, pending, pending_since) = decide_label(
            self.label,
            self.pending_label,
            self.pending_since,
            candidate,
            now,
            POSTURE_HOLD_SECS,
        );
        self.label = label;
        self.pending_label = pending;
        self.pending_since = pending_since;

        // EMA on the continuous score (follows the live candidate, smoothed).
        let target = Self::posture_score(candidate);
        self.ema_score = ema(self.ema_score, target, SCORE_ALPHA);

        // EMA on the body metrics (only when a skeleton is present this frame).
        if let Some(a) = &body_angles {
            self.ema_shoulder_tilt = ema(self.ema_shoulder_tilt, a.shoulder_tilt_deg, BODY_ALPHA);
            self.ema_spine_lateral = ema(self.ema_spine_lateral, a.spine_lateral_deg, BODY_ALPHA);
            self.ema_head_drop = ema(self.ema_head_drop, a.head_drop_ratio, BODY_ALPHA);
            self.ema_torso_compress = ema(self.ema_torso_compress, a.torso_compress_ratio, BODY_ALPHA);
        }

        PostureInfo {
            presence: self.state,
            posture: self.label,
            posture_score: self.ema_score.unwrap_or(0.0),
            baseline_calibrated: self.baseline.is_calibrated(),
            focused_since: self.focused_started_at.unwrap_or(0.0),
            body_present: body_angles.is_some(),
            shoulder_tilt_deg: self.ema_shoulder_tilt.unwrap_or(0.0),
            spine_lateral_deg: self.ema_spine_lateral.unwrap_or(0.0),
            head_drop_ratio: self.ema_head_drop.unwrap_or(0.0),
            torso_compress_ratio: self.ema_torso_compress.unwrap_or(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(center_x: f32, center_y: f32, size: f32, yaw: f32, pitch: f32) -> FaceObs {
        FaceObs {
            center_x_norm: center_x,
            center_y_norm: center_y,
            size_ratio: size,
            yaw_deg: yaw,
            pitch_deg: pitch,
            score: 0.95,
        }
    }

    fn calibrated_baseline() -> Baseline {
        Baseline {
            face_center_x: 0.5,
            face_center_y: 0.45,
            face_size_ratio: 0.25,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            calibrated_at: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn absence_transitions_to_not_present() {
        let mut t = PostureTracker::new(Thresholds::default());
        assert_eq!(t.presence(), PresenceState::NotPresent);
        let info = t.process(None, None);
        assert_eq!(info.presence, PresenceState::NotPresent);
        // With no face and no baseline, posture stays Unknown and body absent.
        assert_eq!(info.posture, PostureState::Unknown);
        assert!(!info.body_present);
    }

    #[test]
    fn focused_centered_face_calibrates_and_is_good() {
        let mut t2 = PostureTracker::with_baseline(calibrated_baseline(), Thresholds::default());
        let info = t2.process(Some(&obs(0.5, 0.45, 0.25, 0.0, 0.0)), None);
        assert_eq!(info.posture, PostureState::Good);
        assert_eq!(info.posture_score, 100.0);
    }

    #[test]
    fn looking_down_is_head_down() {
        let mut t = PostureTracker::with_baseline(calibrated_baseline(), Thresholds::default());
        let info = t.process(Some(&obs(0.5, 0.45, 0.25, 0.0, 35.0)), None); // pitch > 20
        assert_eq!(info.posture, PostureState::HeadDown);
        assert_eq!(info.posture_score, 40.0);
    }

    #[test]
    fn leaning_forward_is_slouching() {
        let mut t = PostureTracker::with_baseline(calibrated_baseline(), Thresholds::default());
        // size_ratio 0.25*1.3 = 0.325 > 1 + posture_size_close(0.15)=1.15x
        let info = t.process(Some(&obs(0.5, 0.45, 0.325, 0.0, 0.0)), None);
        assert_eq!(info.posture, PostureState::Slouching);
        assert_eq!(info.posture_score, 50.0);
    }

    #[test]
    fn drifted_is_leaning() {
        let mut t = PostureTracker::with_baseline(calibrated_baseline(), Thresholds::default());
        // x drift 0.3 > posture_x_drift 0.12
        let info = t.process(Some(&obs(0.8, 0.45, 0.25, 0.0, 0.0)), None);
        assert_eq!(info.posture, PostureState::Leaning);
        assert_eq!(info.posture_score, 65.0);
    }

    #[test]
    fn lean_back_is_lean_back() {
        let mut t = PostureTracker::with_baseline(calibrated_baseline(), Thresholds::default());
        // size_ratio 0.25*0.8 = 0.20 < 1 - 0.15 = 0.85x
        let info = t.process(Some(&obs(0.5, 0.45, 0.20, 0.0, 0.0)), None);
        assert_eq!(info.posture, PostureState::LeanBack);
        assert_eq!(info.posture_score, 70.0);
    }

    #[test]
    fn no_baseline_is_unknown() {
        let mut t = PostureTracker::new(Thresholds::default());
        let info = t.process(Some(&obs(0.5, 0.45, 0.25, 0.0, 0.0)), None);
        assert_eq!(info.posture, PostureState::Unknown);
        assert_eq!(info.posture_score, 0.0);
        assert!(!info.baseline_calibrated);
    }

    #[test]
    fn signal_weights_sum_to_one() {
        let s = PresenceSignals {
            face_present: 1.0,
            face_size_ok: 1.0,
            face_stable: 1.0,
            head_facing: 1.0,
            recent_motion: 1.0,
        };
        assert!((s.total() - 1.0).abs() < 1e-6);
    }

    // --- decide_label hysteresis (no wall clock needed) ---
    #[test]
    fn label_hysteresis_holds_through_dwell() {
        // Start neutral; a HeadDown candidate appears at t=10.
        let mut cur = PostureState::Good;
        let mut pend = PostureState::Good;
        let mut pend_since = 0.0;
        // t=10: candidate HeadDown, pending starts now.
        (cur, pend, pend_since) =
            decide_label(cur, pend, pend_since, PostureState::HeadDown, 10.0, 4.0);
        assert_eq!(cur, PostureState::Good); // not switched yet
        assert_eq!(pend, PostureState::HeadDown);
        // t=12: still within 4s dwell.
        (cur, pend, pend_since) =
            decide_label(cur, pend, pend_since, PostureState::HeadDown, 12.0, 4.0);
        assert_eq!(cur, PostureState::Good);
        // t=14: dwell elapsed -> switch.
        (cur, pend, pend_since) =
            decide_label(cur, pend, pend_since, PostureState::HeadDown, 14.0, 4.0);
        assert_eq!(cur, PostureState::HeadDown);
        assert_eq!(pend, PostureState::HeadDown);
    }

    #[test]
    fn label_hysteresis_cancels_when_candidate_reverts() {
        let mut cur = PostureState::Good;
        let mut pend = PostureState::Good;
        let mut pend_since = 0.0;
        (cur, pend, pend_since) =
            decide_label(cur, pend, pend_since, PostureState::HeadDown, 10.0, 4.0);
        assert_eq!(cur, PostureState::Good);
        // Candidate flips back to Good before dwell elapses -> no switch and
        // pending is reset, so a later HeadDown must wait a fresh 4s.
        (cur, pend, pend_since) =
            decide_label(cur, pend, pend_since, PostureState::Good, 11.0, 4.0);
        assert_eq!(cur, PostureState::Good);
        assert_eq!(pend, PostureState::Good);
        (cur, pend, pend_since) =
            decide_label(cur, pend, pend_since, PostureState::HeadDown, 12.0, 4.0);
        // Only 1s since the revert -> still not switched.
        assert_eq!(cur, PostureState::Good);
    }

    // --- body-angle geometry ---
    fn kpt(x: f32, y: f32, s: f32) -> [f32; 3] {
        [x, y, s]
    }

    #[test]
    fn body_angles_upright_are_near_zero() {
        // Shoulders level at y=100, hips below at y=300, nose well above.
        // Image (x right, y down). Shoulder width 200 px.
        let k = [
            kpt(400.0, 40.0, 0.9),  // nose
            kpt(0.0, 0.0, 0.0),     // leye (unused)
            kpt(0.0, 0.0, 0.0),     // reye
            kpt(0.0, 0.0, 0.0),     // lear
            kpt(0.0, 0.0, 0.0),     // rear
            kpt(300.0, 100.0, 0.9), // lshoulder
            kpt(500.0, 100.0, 0.9), // rshoulder
            kpt(0.0, 0.0, 0.0),     // lelbow
            kpt(0.0, 0.0, 0.0),     // relbow
            kpt(0.0, 0.0, 0.0),     // lwrist
            kpt(0.0, 0.0, 0.0),     // rwrist
            kpt(300.0, 300.0, 0.9), // lhip
            kpt(500.0, 300.0, 0.9), // rhip
            kpt(0.0, 0.0, 0.0),     // lknee
            kpt(0.0, 0.0, 0.0),     // rknee
            kpt(0.0, 0.0, 0.0),     // lankle
            kpt(0.0, 0.0, 0.0),     // rankle
        ];
        let a = compute_body_angles(&k).expect("confident upright pose");
        assert!(a.shoulder_tilt_deg.abs() < 1.0, "shoulder level: {}", a.shoulder_tilt_deg);
        assert!(a.spine_lateral_deg.abs() < 1.0, "spine vertical: {}", a.spine_lateral_deg);
        // Nose 60px above shoulder midline (y=40 vs 100), width 200 -> ratio 0.3.
        // That is a short neck; assert it's positive and sane.
        assert!(a.head_drop_ratio > 0.0 && a.head_drop_ratio < 1.0, "head_drop: {}", a.head_drop_ratio);
        // Shoulder->hip drop 200px / width 200 = 1.0.
        assert!((a.torso_compress_ratio - 1.0).abs() < 0.05, "torso: {}", a.torso_compress_ratio);
    }

    #[test]
    fn body_angles_detect_shoulder_tilt() {
        // Right shoulder 40px lower than left -> clear tilt.
        let k = [
            kpt(400.0, 40.0, 0.9),
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
            kpt(300.0, 100.0, 0.9), // lshoulder
            kpt(500.0, 140.0, 0.9), // rshoulder (lower)
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
            kpt(300.0, 300.0, 0.9), kpt(500.0, 300.0, 0.9),
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
        ];
        let a = compute_body_angles(&k).expect("confident tilted pose");
        // atan2(40, 200) ≈ 11.3°.
        assert!(a.shoulder_tilt_deg > 5.0, "tilt detected: {}", a.shoulder_tilt_deg);
    }

    #[test]
    fn body_angles_none_when_spine_missing() {
        let k = [
            kpt(400.0, 40.0, 0.9),
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
            kpt(300.0, 100.0, 0.9), kpt(500.0, 100.0, 0.9),
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
            kpt(300.0, 300.0, 0.05), kpt(500.0, 300.0, 0.05), // hips below threshold
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
        ];
        assert!(compute_body_angles(&k).is_none());
    }

    #[test]
    fn body_branch_flags_slouching_on_torso_collapse() {
        // Baseline: upright (torso_compress 1.0, head_drop 0.3, tilt 0, lat 0).
        let mut bl = calibrated_baseline();
        bl.body_shoulder_tilt_deg = Some(0.0);
        bl.body_spine_lateral_deg = Some(0.0);
        bl.body_head_drop_ratio = Some(0.30);
        bl.body_torso_compress_ratio = Some(1.0);
        let mut t = PostureTracker::with_baseline(bl, Thresholds::default());

        // Current: torso collapsed to half the baseline drop (hunched) while
        // the face stays perfectly centred, so the face branch says Good. The
        // body branch must override with Slouching.
        let k = [
            kpt(400.0, 90.0, 0.9),  // nose only 10px above shoulders now
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
            kpt(300.0, 100.0, 0.9), kpt(500.0, 100.0, 0.9),
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
            kpt(300.0, 200.0, 0.9), kpt(500.0, 200.0, 0.9), // hips only 100px below (was 200)
            kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0), kpt(0.0, 0.0, 0.0),
        ];
        let body = BodyPose {
            keypoints: k.to_vec(),
            present: true,
        };
        let info = t.process(Some(&obs(0.5, 0.45, 0.25, 0.0, 0.0)), Some(&body));
        // torso_compress went 1.0 -> 0.5, d_comp=0.5 > TORSO_DEV(0.2).
        assert_eq!(info.posture, PostureState::Slouching, "body branch overrides face Good");
        assert!(info.body_present);
    }

    #[test]
    fn body_absent_degrades_to_face_only() {
        // Baseline has body angles, but this frame supplies no body -> body
        // branch neutral, verdict comes purely from the face branch.
        let mut bl = calibrated_baseline();
        bl.body_shoulder_tilt_deg = Some(0.0);
        bl.body_spine_lateral_deg = Some(0.0);
        bl.body_head_drop_ratio = Some(0.30);
        bl.body_torso_compress_ratio = Some(1.0);
        let mut t = PostureTracker::with_baseline(bl, Thresholds::default());
        // Face drifted -> Leaning, with no body to contradict.
        let info = t.process(Some(&obs(0.8, 0.45, 0.25, 0.0, 0.0)), None);
        assert_eq!(info.posture, PostureState::Leaning);
        assert!(!info.body_present);
    }
}
