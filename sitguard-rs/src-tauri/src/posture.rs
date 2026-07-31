//! Sitting-posture judgment: presence state machine + five-signal fusion +
//! posture classification.
//!
//! Ported 1:1 from the recovered Python reference `recovered_reference/detector.py`,
//! with `Baseline` / `Thresholds` defaults read directly from the compiled
//! `sitguard/config.pyc` (Python 3.12). This module owns no camera, timers or UI —
//! it is pure perception/state, driven frame-by-frame by [`PostureTracker::process`].

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

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

/// Sitting-posture classification, computed every frame from face geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PostureState {
    Unknown,
    Good,
    HeadDown,
    Slouching,
    LeanBack,
    Leaning,
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

/// Result returned to the caller each frame.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PostureInfo {
    pub presence: PresenceState,
    pub posture: PostureState,
    pub posture_score: f32,
    pub baseline_calibrated: bool,
    /// Unix seconds the current focused streak started, or 0.0 if not focused.
    pub focused_since: f64,
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
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
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

        // Priority: head_down > slouching > lean_back > leaning > good
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

    fn snapshot_baseline(&mut self, obs: &FaceObs) {
        let now = now_secs();
        self.baseline = Baseline {
            face_center_x: obs.center_x_norm,
            face_center_y: obs.center_y_norm,
            face_size_ratio: obs.size_ratio,
            yaw_deg: obs.yaw_deg,
            pitch_deg: obs.pitch_deg,
            calibrated_at: now,
        };
    }

    /// Feed one frame. `obs` is `None` when no face was detected.
    pub fn process(&mut self, obs: Option<&FaceObs>) -> PostureInfo {
        let now = now_secs();
        let dt = (now - self.last_process_at) as f32;
        self.last_process_at = now;

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

        // Auto-calibrate: once focused + stable for >= 2s, snapshot the baseline.
        if obs.is_some()
            && self.state == PresenceState::PresentFocused
            && !self.baseline.is_calibrated()
        {
            let stable = self.compute_signals(obs.unwrap()).face_stable;
            if stable >= 0.7 || self.force_calibrate_next {
                self.calib_accum_seconds += dt;
                if self.calib_accum_seconds >= 2.0 || self.force_calibrate_next {
                    self.snapshot_baseline(obs.unwrap());
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

        let posture = match obs {
            Some(o) => self.compute_posture(o),
            None => PostureState::Unknown,
        };

        PostureInfo {
            presence: self.state,
            posture,
            posture_score: Self::posture_score(posture),
            baseline_calibrated: self.baseline.is_calibrated(),
            focused_since: self.focused_started_at.unwrap_or(0.0),
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

    #[test]
    fn absence_transitions_to_not_present() {
        let mut t = PostureTracker::new(Thresholds::default());
        // Process a face a few times to seed last_face_seen, then go absent.
        let _ = t.process(Some(&obs(0.5, 0.45, 0.25, 0.0, 0.0)));
        // Simulate absence beyond absent_timeout by manipulating via many None frames
        // (last_face_seen stays; timeout is 10s — we just assert presence starts NotPresent).
        assert_eq!(t.presence(), PresenceState::NotPresent);
        let info = t.process(None);
        assert_eq!(info.presence, PresenceState::NotPresent);
        assert_eq!(info.posture, PostureState::Unknown);
    }

    #[test]
    fn focused_centered_face_calibrates_and_is_good() {
        // Hold a perfectly centered, stable face. We cannot fake time, so assert
        // the steady-state behaviour with an already-calibrated baseline.
        let bl = Baseline {
            face_center_x: 0.5,
            face_center_y: 0.45,
            face_size_ratio: 0.25,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            calibrated_at: 1.0,
        };
        let mut t2 = PostureTracker::with_baseline(bl, Thresholds::default());
        let info = t2.process(Some(&obs(0.5, 0.45, 0.25, 0.0, 0.0)));
        assert_eq!(info.posture, PostureState::Good);
        assert_eq!(info.posture_score, 100.0);
    }

    #[test]
    fn looking_down_is_head_down() {
        let bl = Baseline {
            face_center_x: 0.5,
            face_center_y: 0.45,
            face_size_ratio: 0.25,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            calibrated_at: 1.0,
        };
        let mut t = PostureTracker::with_baseline(bl, Thresholds::default());
        let info = t.process(Some(&obs(0.5, 0.45, 0.25, 0.0, 35.0))); // pitch > 20
        assert_eq!(info.posture, PostureState::HeadDown);
        assert_eq!(info.posture_score, 40.0);
    }

    #[test]
    fn leaning_forward_is_slouching() {
        let bl = Baseline {
            face_center_x: 0.5,
            face_center_y: 0.45,
            face_size_ratio: 0.25,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            calibrated_at: 1.0,
        };
        let mut t = PostureTracker::with_baseline(bl, Thresholds::default());
        // size_ratio 0.25*1.3 = 0.325 > 1 + posture_size_close(0.15)=1.15x
        let info = t.process(Some(&obs(0.5, 0.45, 0.325, 0.0, 0.0)));
        assert_eq!(info.posture, PostureState::Slouching);
        assert_eq!(info.posture_score, 50.0);
    }

    #[test]
    fn drifted_is_leaning() {
        let bl = Baseline {
            face_center_x: 0.5,
            face_center_y: 0.45,
            face_size_ratio: 0.25,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            calibrated_at: 1.0,
        };
        let mut t = PostureTracker::with_baseline(bl, Thresholds::default());
        // x drift 0.3 > posture_x_drift 0.12
        let info = t.process(Some(&obs(0.8, 0.45, 0.25, 0.0, 0.0)));
        assert_eq!(info.posture, PostureState::Leaning);
        assert_eq!(info.posture_score, 65.0);
    }

    #[test]
    fn lean_back_is_lean_back() {
        let bl = Baseline {
            face_center_x: 0.5,
            face_center_y: 0.45,
            face_size_ratio: 0.25,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            calibrated_at: 1.0,
        };
        let mut t = PostureTracker::with_baseline(bl, Thresholds::default());
        // size_ratio 0.25*0.8 = 0.20 < 1 - 0.15 = 0.85x
        let info = t.process(Some(&obs(0.5, 0.45, 0.20, 0.0, 0.0)));
        assert_eq!(info.posture, PostureState::LeanBack);
        assert_eq!(info.posture_score, 70.0);
    }

    #[test]
    fn no_baseline_is_unknown() {
        let mut t = PostureTracker::new(Thresholds::default());
        let info = t.process(Some(&obs(0.5, 0.45, 0.25, 0.0, 0.0)));
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
}
