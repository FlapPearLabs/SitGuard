//! Sitting-session timing, statistics and reminder logic.
//!
//! Ports the `Session` semantics from the recovered Python CLI
//! (`recovered_reference/cli.py`): accumulate continuous "sitting" time while
//! the user is present, fire a break reminder after `sitting_threshold_minutes`
//! (cooldown `reminder_cooldown_minutes`), and a posture reminder when a bad
//! posture persists beyond `posture_dwell_seconds` (cooldown
//! `posture_cooldown_minutes`). Also tracks session statistics for the report.

use crate::posture::{PostureState, PresenceState, Thresholds};
use serde::Serialize;
use std::time::{Duration, Instant};

/// Session statistics, sent to the frontend for the report page.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct SessionStats {
    /// Total seconds the user was present at the desk this session.
    pub sitting_seconds: f64,
    /// Seconds the user was "focused" (good presence score) this session.
    pub focus_seconds: f64,
    /// Number of break reminders fired.
    pub break_reminders: u32,
    /// Number of posture-correction reminders fired.
    pub posture_reminders: u32,
    /// Minutes until the next break reminder (for the report/tray tooltip).
    pub next_reminder_minutes: f64,
    /// Seconds spent in each posture bucket (weighted by frame time).
    pub secs_unknown: f64,
    pub secs_good: f64,
    pub secs_head_down: f64,
    pub secs_slouching: f64,
    pub secs_lean_back: f64,
    pub secs_leaning: f64,
}

/// Tracks the sitting session: continuous-presence timer, reminder gating and
/// statistics. Driven frame-by-frame by [`SessionTracker::tick`]; reminder
/// decisions are queried and acknowledged by the caller (which fires the
/// OS notification and calls `mark_*_reminded`).
pub struct SessionTracker {
    th: Thresholds,
    sitting_since: Option<Instant>,
    last_break_reminder: Option<Instant>,
    bad_posture_since: Option<Instant>,
    last_posture_reminder: Option<Instant>,
    paused_until: Option<Instant>,
    // accumulated stats
    sitting_accum: f64,
    focus_accum: f64,
    break_reminders: u32,
    posture_reminders: u32,
    secs: [f64; 6], // indexed by the match in tick()
    last_tick: Option<Instant>,
}

impl SessionTracker {
    pub fn new(th: Thresholds) -> Self {
        SessionTracker {
            th,
            sitting_since: None,
            last_break_reminder: None,
            bad_posture_since: None,
            last_posture_reminder: None,
            paused_until: None,
            sitting_accum: 0.0,
            focus_accum: 0.0,
            break_reminders: 0,
            posture_reminders: 0,
            secs: [0.0; 6],
            last_tick: None,
        }
    }

    pub fn set_thresholds(&mut self, th: Thresholds) {
        self.th = th;
    }

    /// Current thresholds (used by the caller to format reminder copy).
    pub fn thresholds(&self) -> Thresholds {
        self.th
    }

    pub fn is_paused(&self, now: Instant) -> bool {
        match self.paused_until {
            Some(t) => now < t,
            None => false,
        }
    }

    /// Pause reminders/accumulation for `minutes` (e.g. user clicked "snooze").
    pub fn pause(&mut self, minutes: f64, now: Instant) {
        let until = now + Duration::from_secs_f64((minutes * 60.0).max(0.0));
        self.paused_until = Some(until);
    }

    /// Advance the session by one frame.
    pub fn tick(&mut self, presence: PresenceState, posture: PostureState, now: Instant) {
        let dt = match self.last_tick {
            Some(prev) if now >= prev => (now - prev).as_secs_f64(),
            _ => 0.0,
        };
        self.last_tick = Some(now);

        if self.is_paused(now) {
            // Paused: freeze sitting streak (mirrors cli.py tick(NOT_PRESENT)).
            self.sitting_since = None;
            self.bad_posture_since = None;
            return;
        }

        if presence == PresenceState::NotPresent {
            self.sitting_since = None;
            self.bad_posture_since = None;
            return;
        }

        // Present → accumulate.
        if self.sitting_since.is_none() {
            self.sitting_since = Some(now);
        }
        self.sitting_accum += dt;
        if presence == PresenceState::PresentFocused {
            self.focus_accum += dt;
        }

        // Posture buckets weighted by frame time.
        let idx = match posture {
            PostureState::Unknown => 0,
            PostureState::Good => 1,
            PostureState::HeadDown => 2,
            PostureState::Slouching => 3,
            PostureState::LeanBack => 4,
            PostureState::Leaning => 5,
        };
        self.secs[idx] += dt;

        // Bad-posture streak (ignore Unknown — not enough info to nag).
        if posture != PostureState::Good && posture != PostureState::Unknown {
            if self.bad_posture_since.is_none() {
                self.bad_posture_since = Some(now);
            }
        } else {
            self.bad_posture_since = None;
        }
    }

    /// Minutes until the next break reminder (for tray tooltip / UI).
    pub fn minutes_to_reminder(&self, now: Instant) -> f64 {
        if self.is_paused(now) {
            if let Some(u) = self.paused_until {
                return (u.saturating_duration_since(now).as_secs_f64() / 60.0).max(0.0);
            }
        }
        let threshold_min = self.th.sitting_threshold_minutes;
        let cooldown_min = self.th.reminder_cooldown_minutes;
        let seated_min = match self.sitting_since {
            Some(s) => now.saturating_duration_since(s).as_secs_f64() / 60.0,
            None => 0.0,
        };
        let to_threshold = (threshold_min - seated_min).max(0.0);
        let since_last = match self.last_break_reminder {
            Some(l) => now.saturating_duration_since(l).as_secs_f64() / 60.0,
            None => f64::INFINITY,
        };
        let to_cooldown = (cooldown_min - since_last).max(0.0);
        to_threshold.max(to_cooldown)
    }

    /// Should a break reminder fire now?
    pub fn should_remind_break(&self, now: Instant) -> bool {
        if self.is_paused(now) {
            return false;
        }
        let seated = match self.sitting_since {
            Some(s) => now.saturating_duration_since(s).as_secs_f64(),
            None => return false,
        };
        if seated < self.th.sitting_threshold_minutes * 60.0 {
            return false;
        }
        match self.last_break_reminder {
            Some(l) => {
                now.saturating_duration_since(l).as_secs_f64() >= self.th.reminder_cooldown_minutes * 60.0
            }
            None => true,
        }
    }

    pub fn mark_break_reminded(&mut self, now: Instant) {
        self.last_break_reminder = Some(now);
        self.break_reminders += 1;
        // Restart the sitting streak so the next reminder is another full
        // threshold of continuous sitting (still gated by cooldown).
        self.sitting_since = Some(now);
    }

    /// Should a posture-correction reminder fire now?
    pub fn should_remind_posture(&self, now: Instant) -> bool {
        if self.is_paused(now) {
            return false;
        }
        let bad = match self.bad_posture_since {
            Some(s) => now.saturating_duration_since(s).as_secs_f64(),
            None => return false,
        };
        if bad < self.th.posture_dwell_seconds {
            return false;
        }
        match self.last_posture_reminder {
            Some(l) => {
                now.saturating_duration_since(l).as_secs_f64() >= self.th.posture_cooldown_minutes * 60.0
            }
            None => true,
        }
    }

    pub fn mark_posture_reminded(&mut self, now: Instant) {
        self.last_posture_reminder = Some(now);
        self.posture_reminders += 1;
        self.bad_posture_since = Some(now);
    }

    pub fn stats(&self) -> SessionStats {
        SessionStats {
            sitting_seconds: self.sitting_accum,
            focus_seconds: self.focus_accum,
            break_reminders: self.break_reminders,
            posture_reminders: self.posture_reminders,
            next_reminder_minutes: self.minutes_to_reminder(Instant::now()),
            secs_unknown: self.secs[0],
            secs_good: self.secs[1],
            secs_head_down: self.secs[2],
            secs_slouching: self.secs[3],
            secs_lean_back: self.secs[4],
            secs_leaning: self.secs[5],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posture::Thresholds;

    fn t0() -> Instant {
        Instant::now()
    }

    #[test]
    fn no_reminder_before_threshold() {
        let th = Thresholds::default();
        let mut s = SessionTracker::new(th);
        let t = t0();
        s.tick(PresenceState::PresentFocused, PostureState::Good, t);
        let t2 = t + Duration::from_secs(10 * 60); // 10 min < 45 min
        s.tick(PresenceState::PresentFocused, PostureState::Good, t2);
        assert!(!s.should_remind_break(t2));
    }

    #[test]
    fn break_reminder_fires_after_threshold_then_cooldown() {
        let th = Thresholds::default();
        let mut s = SessionTracker::new(th);
        let t = t0();
        s.tick(PresenceState::PresentFocused, PostureState::Good, t);
        let t2 = t + Duration::from_secs(46 * 60); // > 45 min
        s.tick(PresenceState::PresentFocused, PostureState::Good, t2);
        assert!(s.should_remind_break(t2));
        s.mark_break_reminded(t2);
        assert!(!s.should_remind_break(t2)); // cooldown blocks immediate re-fire
        let t3 = t2 + Duration::from_secs(4 * 60); // under 5-min cooldown
        assert!(!s.should_remind_break(t3));
        let t4 = t2 + Duration::from_secs(6 * 60); // streak restarted, only 6 min
        assert!(!s.should_remind_break(t4));
    }

    #[test]
    fn bad_posture_reminder_after_dwell() {
        let th = Thresholds::default();
        let mut s = SessionTracker::new(th);
        let t = t0();
        s.tick(PresenceState::PresentFocused, PostureState::HeadDown, t);
        let t2 = t + Duration::from_secs(10); // < 15s dwell
        s.tick(PresenceState::PresentFocused, PostureState::HeadDown, t2);
        assert!(!s.should_remind_posture(t2));
        let t3 = t + Duration::from_secs(20); // > 15s dwell
        s.tick(PresenceState::PresentFocused, PostureState::HeadDown, t3);
        assert!(s.should_remind_posture(t3));
        s.mark_posture_reminded(t3);
        assert!(!s.should_remind_posture(t3));
    }

    #[test]
    fn absence_resets_streak() {
        let th = Thresholds::default();
        let mut s = SessionTracker::new(th);
        let t = t0();
        s.tick(PresenceState::PresentFocused, PostureState::Good, t);
        let t2 = t + Duration::from_secs(46 * 60);
        s.tick(PresenceState::PresentFocused, PostureState::Good, t2);
        assert!(s.should_remind_break(t2));
        let t3 = t2 + Duration::from_secs(5);
        s.tick(PresenceState::NotPresent, PostureState::Unknown, t3);
        assert!(!s.should_remind_break(t3 + Duration::from_secs(46 * 60)));
    }

    #[test]
    fn pause_freezes_and_blocks() {
        let th = Thresholds::default();
        let mut s = SessionTracker::new(th);
        let t = t0();
        s.tick(PresenceState::PresentFocused, PostureState::Good, t);
        s.pause(10.0, t); // pause for 600s from t
        let tp = t + Duration::from_secs(100); // within pause window
        assert!(s.is_paused(tp));
        assert!(!s.should_remind_break(tp));
        // ~8.33 min left in the pause
        assert!((s.minutes_to_reminder(tp) - 500.0 / 60.0).abs() < 0.5);
        let te = t + Duration::from_secs(700); // past pause
        assert!(!s.is_paused(te));
    }

    #[test]
    fn stats_accumulate_sitting_and_posture() {
        let th = Thresholds::default();
        let mut s = SessionTracker::new(th);
        let t = t0();
        s.tick(PresenceState::PresentFocused, PostureState::Good, t);
        let t2 = t + Duration::from_secs(60);
        s.tick(PresenceState::PresentFocused, PostureState::Good, t2);
        let st = s.stats();
        assert!((st.sitting_seconds - 60.0).abs() < 1.0, "sitting={}", st.sitting_seconds);
        assert!((st.focus_seconds - 60.0).abs() < 1.0);
        assert!(st.secs_good > 50.0, "good={}", st.secs_good);
    }
}
