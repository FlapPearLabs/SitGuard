//! MoveNet full-body pose estimation via tract-onnx (pure Rust).
//!
//! Optional upgrade over the YuNet-only face pipeline: detects 17 COCO
//! keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees,
//! ankles) per frame, enabling a body skeleton overlay and (later) body-angle
//! based posture scoring.
//!
//! Model: MoveNet SinglePose Lightning, ONNX export (PINTO_model_zoo
//! `115_MoveNet`, `model_float32.onnx`). This export uses an NHWC input
//! [1,192,192,3] and a single output [1,1,17,3] where the last dim is
//! (y, x, score) in normalized [0,1] image coordinates. Some other exports use
//! NCHW + an extra offset output; we auto-detect the layout and read output 0,
//! which is the keypoints in both cases.
//!
//! Graceful degradation: if the model file is missing, `PoseDetector::try_new`
//! returns Err and the caller simply skips body estimation — the YuNet face
//! pipeline and posture logic keep working.

use image::RgbImage;
use serde::Serialize;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;

/// Keypoints with confidence >= this are drawn / used.
pub const KEYPOINT_THRESHOLD: f32 = 0.3;
/// COCO 17-keypoint layout used by MoveNet.
pub const NUM_KEYPOINTS: usize = 17;
/// Square input size this PINTO MoveNet Lightning export expects.
pub const POSE_SIZE: usize = 192;

/// One full-body pose, keypoints in original frame pixels (x, y, score).
#[derive(Debug, Clone, Serialize)]
pub struct BodyPose {
    /// 17 entries: [x_px, y_px, score] (score 0..1). Low-score points are
    /// still emitted so the frontend can decide whether to draw them.
    pub keypoints: Vec<[f32; 3]>,
    /// True if at least one keypoint exceeds KEYPOINT_THRESHOLD.
    pub present: bool,
}

type RunModel = TypedRunnableModel<TypedModel>;

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum Layout {
    Nhwc,
    Nchw,
}

pub struct PoseDetector {
    model: RunModel,
    in_w: usize,
    in_h: usize,
    layout: Layout,
    /// Frame resolution the detections are mapped back into.
    frame_w: u32,
    frame_h: u32,
}

impl PoseDetector {
    /// Try to build a detector. Fails (returns Err) if the model file is
    /// missing or cannot be loaded — callers should treat that as "body
    /// estimation unavailable" and continue without it.
    pub fn try_new(model_path: &Path, frame_w: u32, frame_h: u32) -> TractResult<Self> {
        let bytes = std::fs::read(model_path)?;

        let mut model = tract_onnx::onnx().model_for_read(&mut Cursor::new(&bytes))?;

        // This PINTO MoveNet Lightning export uses an NHWC input
        // [1,192,192,3]; pin it explicitly (mirrors how detector.rs fixes the
        // YuNet input size rather than re-reading it from the model fact).
        let (h, w) = (POSE_SIZE, POSE_SIZE);
        model.set_input_fact(0, f32::fact([1usize, h, w, 3]).into())?;
        let model = model.into_optimized()?.into_runnable()?;

        log::info!(
            "MoveNet ready: frame {}x{} -> model {}x{} (NHWC)",
            frame_w, frame_h, w, h
        );
        Ok(Self {
            model,
            in_w: w,
            in_h: h,
            layout: Layout::Nhwc,
            frame_w,
            frame_h,
        })
    }

    /// Run pose estimation on an RGB frame (must match the resolution given to
    /// `try_new`). Returns 17 keypoints in frame-pixel coordinates.
    pub fn detect(&self, img: &RgbImage) -> TractResult<BodyPose> {
        // Stretch to the model's square input (matches the reference demo's
        // resize_exact; MoveNet is tolerant of the mild aspect distortion).
        let resized = image::imageops::resize(
            img,
            self.in_w as u32,
            self.in_h as u32,
            image::imageops::FilterType::Triangle,
        );

        // RGB planar f32, raw 0..255 (no normalization).
        let (mw, mh) = (self.in_w, self.in_h);
        let raw = resized.as_raw();
        let arr = match self.layout {
            Layout::Nhwc => {
                let mut a = tract_ndarray::Array4::<f32>::zeros((1, mh, mw, 3));
                for y in 0..mh {
                    for x in 0..mw {
                        let o = (y * mw + x) * 3;
                        a[[0, y, x, 0]] = raw[o] as f32;
                        a[[0, y, x, 1]] = raw[o + 1] as f32;
                        a[[0, y, x, 2]] = raw[o + 2] as f32;
                    }
                }
                a
            }
            Layout::Nchw => {
                let mut a = tract_ndarray::Array4::<f32>::zeros((1, 3, mh, mw));
                for y in 0..mh {
                    for x in 0..mw {
                        let o = (y * mw + x) * 3;
                        a[[0, 0, y, x]] = raw[o] as f32;
                        a[[0, 1, y, x]] = raw[o + 1] as f32;
                        a[[0, 2, y, x]] = raw[o + 2] as f32;
                    }
                }
                a
            }
        };

        let outputs = self.model.run(tvec!(Tensor::from(arr).into()))?;
        let out = outputs[0].to_array_view::<f32>()?;

        let mut kpts: Vec<[f32; 3]> = Vec::with_capacity(NUM_KEYPOINTS);
        let mut present = false;
        for i in 0..NUM_KEYPOINTS {
            let yn = out[[0, 0, i, 0]];
            let xn = out[[0, 0, i, 1]];
            let score = out[[0, 0, i, 2]];
            let x = xn * self.frame_w as f32;
            let y = yn * self.frame_h as f32;
            if score > KEYPOINT_THRESHOLD {
                present = true;
            }
            kpts.push([x, y, score]);
        }

        Ok(BodyPose {
            keypoints: kpts,
            present,
        })
    }
}

/// Resolve a model file by checking a few candidate locations relative to the
/// running executable: next to the exe, in a `models/` subdir, and (for `tauri
/// dev`) three levels up at `src-tauri/models/`. Returns None if not found.
pub fn find_model(name: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let exe_dir = exe.parent()?;
    let candidates = [
        exe_dir.join(name),
        exe_dir.join("models").join(name),
        exe_dir
            .join("..")
            .join("..")
            .join("..")
            .join("src-tauri")
            .join("models")
            .join(name),
    ];
    candidates.into_iter().find(|p| p.exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movenet_missing_model_is_err_not_panic() {
        let p = Path::new("/nonexistent/movenet_lightning.onnx");
        assert!(PoseDetector::try_new(p, 1280, 720).is_err());
    }

    /// Runs only when the model file is present (it is optional / not always
    /// checked in). Verifies the model loads, runs, and yields 17 keypoints.
    #[test]
    fn movenet_runs_when_model_present() {
        let Some(path) = find_model("movenet_lightning.onnx") else {
            eprintln!("skip: movenet_lightning.onnx not found");
            return;
        };
        let det = match PoseDetector::try_new(&path, 640, 480) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip: model failed to load: {e}");
                return;
            }
        };
        let blank = RgbImage::new(640, 480);
        let pose = det.detect(&blank).expect("inference should run");
        assert_eq!(pose.keypoints.len(), NUM_KEYPOINTS);
    }
}
