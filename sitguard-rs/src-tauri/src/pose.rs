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

/// Resolve a model file by checking candidate locations, in priority order.
/// Covers every layout the binary can be launched from:
///   1. `<exe_dir>/<name>`                       — portable, model next to exe
///   2. `<exe_dir>/models/<name>`                — portable, model in a models/ dir
///   3. `<exe_dir>/resources/models/<name>`      — Tauri *installed* bundle on
///                                                  Windows / Linux (Tauri copies
///                                                  `bundle.resources` into a
///                                                  `resources/` dir next to the exe)
///   4. `<exe_dir>/../Resources/models/<name>`   — Tauri *installed* bundle on macOS
///                                                  (Resources live one level up from
///                                                  the MacOS/ exe dir)
///   5. `<exe_dir>/../../../src-tauri/models/<name>` — `tauri dev` / local debug build
///   6. `<CARGO_MANIFEST_DIR>/models/<name>`     — `cargo test` / build-path fallback
///
/// Candidate 6 exists because candidates 1-5 all miss under `cargo test`: the
/// test binary lives one level deeper (target/debug/deps/), which makes the
/// relative walk in candidate 5 resolve to `src-tauri/src-tauri/models/`.
/// Without it the MoveNet test silently skipped and reported a false pass.
/// CARGO_MANIFEST_DIR is baked in at compile time, so for a shipped exe it
/// points at a build path that no longer exists — harmless, since every
/// candidate is existence-checked.
pub fn find_model(name: &str) -> Option<PathBuf> {
    let manifest_models = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name);

    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let candidates = [
                // 1. portable: model next to the exe
                exe_dir.join(name),
                // 2. portable: model under a models/ dir next to the exe
                exe_dir.join("models").join(name),
                // 3. Tauri installed bundle (Windows / Linux)
                exe_dir.join("resources").join("models").join(name),
                // 4. Tauri installed bundle (macOS)
                exe_dir
                    .join("..")
                    .join("Resources")
                    .join("models")
                    .join(name),
                // 5. tauri dev / local debug build
                exe_dir
                    .join("..")
                    .join("..")
                    .join("..")
                    .join("src-tauri")
                    .join("models")
                    .join(name),
                // 6. cargo test / build-path fallback
                manifest_models.clone(),
            ];
            return candidates.into_iter().find(|p| p.exists());
        }
    }

    manifest_models.exists().then_some(manifest_models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movenet_missing_model_is_err_not_panic() {
        let p = Path::new("/nonexistent/movenet_lightning.onnx");
        assert!(PoseDetector::try_new(p, 1280, 720).is_err());
    }

    /// End-to-end MoveNet check: the model is committed to the repo, so this
    /// must hard-fail rather than skip. An earlier version returned early when
    /// the model was not found, which made the test pass without running any
    /// inference at all (a false green in CI).
    #[test]
    fn movenet_loads_and_infers_17_keypoints() {
        let path = find_model("movenet_lightning.onnx").expect(
            "movenet_lightning.onnx must be resolvable — it is committed under \
             src-tauri/models/. If this fails, find_model's candidate paths are \
             wrong for the current build layout.",
        );
        let det = PoseDetector::try_new(&path, 640, 480)
            .expect("MoveNet model must load via tract-onnx");
        let blank = RgbImage::new(640, 480);
        let pose = det.detect(&blank).expect("inference should run");
        assert_eq!(pose.keypoints.len(), NUM_KEYPOINTS);

        // Coordinates must be mapped back into frame pixel space, not left as
        // the model's normalised 0..1 output.
        for kp in &pose.keypoints {
            assert!(
                kp[0] >= -1.0 && kp[0] <= 641.0,
                "x out of frame range: {}",
                kp[0]
            );
            assert!(
                kp[1] >= -1.0 && kp[1] <= 481.0,
                "y out of frame range: {}",
                kp[1]
            );
            assert!(
                kp[2] >= 0.0 && kp[2] <= 1.0,
                "score not a probability: {}",
                kp[2]
            );
        }
    }

    /// Numerical regression test against ONNX Runtime.
    ///
    /// tract 0.21.17 needed three local patches before MoveNet produced the
    /// right numbers (see `vendor/` and the `[patch.crates-io]` block in
    /// Cargo.toml). Two of them were silent: the model still loaded and still
    /// returned a correctly-shaped `[1,1,17,3]` tensor, but the values were
    /// wrong, so nothing short of comparing against a reference runtime could
    /// catch it. These expectations were produced by onnxruntime 1.x on the
    /// exact same input and pin that behaviour down.
    ///
    /// The frame is deliberately 192x192 so the pre-resize is the identity and
    /// tract sees byte-for-byte the tensor ORT was given.
    #[test]
    fn movenet_matches_onnxruntime_reference() {
        // ORT reference for input pixel (x, y, c) = (x*3 + y*5 + c*7) % 256,
        // as (y_norm, x_norm, score).
        const REF: [[f32; 3]; NUM_KEYPOINTS] = [
            [0.026041, 0.438785, 0.036977],
            [0.029832, 0.479036, 0.017419],
            [0.035259, 0.372011, 0.018954],
            [0.020303, 0.552235, 0.015315],
            [0.035263, 0.352457, 0.014147],
            [0.140714, 0.637687, 0.028879],
            [0.143770, 0.336519, 0.015608],
            [0.474040, 0.554813, 0.066268],
            [0.476009, 0.275067, 0.027204],
            [0.554401, 0.417411, 0.079596],
            [0.644042, 0.268581, 0.067682],
            [0.671733, 0.674539, 0.104292],
            [0.675350, 0.441578, 0.114022],
            [0.967952, 0.624858, 0.054924],
            [0.894670, 0.291335, 0.054344],
            [0.989118, 0.593188, 0.095599],
            [0.917624, 0.282131, 0.042064],
        ];
        // Loose enough for f32 accumulation-order differences between runtimes
        // (observed: <1e-5), tight enough to fail on the half_pixel
        // extrapolation bug, which moved keypoints by up to 0.02 and scores by
        // up to 0.028.
        const TOL: f32 = 0.01;

        let side = POSE_SIZE as u32;
        let path = find_model("movenet_lightning.onnx").expect("model must be resolvable");
        let det = PoseDetector::try_new(&path, side, side).expect("MoveNet must load");
        let frame = RgbImage::from_fn(side, side, |x, y| {
            image::Rgb([
                ((x * 3 + y * 5) % 256) as u8,
                ((x * 3 + y * 5 + 7) % 256) as u8,
                ((x * 3 + y * 5 + 14) % 256) as u8,
            ])
        });

        let pose = det.detect(&frame).expect("inference should run");
        for (i, (kp, want)) in pose.keypoints.iter().zip(REF.iter()).enumerate() {
            // detect() returns [x_px, y_px, score]; REF is [y_norm, x_norm, score].
            let got = [kp[1] / side as f32, kp[0] / side as f32, kp[2]];
            for (c, label) in ["y", "x", "score"].iter().enumerate() {
                assert!(
                    (got[c] - want[c]).abs() < TOL,
                    "kp{i} {label}: tract {} vs onnxruntime {} (delta {:.6})",
                    got[c],
                    want[c],
                    (got[c] - want[c]).abs()
                );
            }
        }
    }
}
