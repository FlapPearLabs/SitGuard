//! YuNet face detection via tract-onnx (pure Rust, no external runtime).
//!
//! Faithful port of the Python `detector.py` pipeline:
//!   frame -> YuNet -> bbox + 5 landmarks + score -> yaw/pitch estimate
//!
//! Model: face_detection_yunet_2023mar.onnx (~228 KB, embedded in the exe).
//! Outputs per stride s in {8,16,32}: cls_s [N,1], obj_s [N,1],
//! bbox_s [N,4], kps_s [N,10], where N = (H/s)*(W/s), row-major.
//! Decode (mirrors OpenCV FaceDetectorYN):
//!   score = sqrt(clamp01(cls) * clamp01(obj))
//!   cx = (col + dx) * s;  cy = (row + dy) * s
//!   w  = exp(dw) * s;     h  = exp(dh) * s
//!   lm_x = (col + kx) * s; lm_y = (row + ky) * s
//!
//! Original thresholds preserved: score 0.6, NMS IoU 0.3, top_k 5.

use image::RgbImage;
use serde::Serialize;
use std::io::Cursor;
use tract_onnx::prelude::*;

// "_320" variant: original opencv_zoo export bakes 640x640 value_info shapes
// which tract refuses to re-specialize; this copy has value_info stripped and
// input fixed to 1x3x320x320 (see scripts/exe_patch notes / project memory).
static MODEL_BYTES: &[u8] =
    include_bytes!("../models/face_detection_yunet_2023mar_320.onnx");

const SCORE_THRESHOLD: f32 = 0.6;
const NMS_THRESHOLD: f32 = 0.3;
const TOP_K: usize = 5;
/// This ONNX export bakes 320x320 into its intermediate value_info shapes,
/// so tract cannot re-specialize to other sizes (unlike OpenCV's DNN which
/// re-infers). We always resize frames to 320x320 and map coords back with
/// independent x/y scales — mild aspect distortion, YuNet handles it fine.
const DET_W: usize = 320;
const DET_H: usize = 320;
const STRIDES: [usize; 3] = [8, 16, 32];

/// Output tensor order we pin explicitly (verified against the model file).
const OUTPUT_NAMES: [&str; 12] = [
    "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
];

/// One detected face, coordinates in original frame pixels.
#[derive(Debug, Clone, Serialize)]
pub struct FaceDet {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    /// 5 points: right_eye, left_eye, nose, right_mouth, left_mouth
    pub landmarks: Vec<[f32; 2]>,
    /// Geometric head-pose estimate (deg). Positive pitch = head down.
    /// NOTE: coarse approximation from 5 landmarks; the Python version used
    /// solvePnP — M3 will calibrate thresholds against real behaviour.
    pub yaw_deg: f32,
    pub pitch_deg: f32,
}

type RunModel = TypedRunnableModel<TypedModel>;

pub struct FaceDetector {
    model: RunModel,
    det_w: usize,
    det_h: usize,
    /// Map detection-space coords back to frame space.
    sx: f32,
    sy: f32,
}

impl FaceDetector {
    /// Build a detector for a fixed frame resolution (input is always 320x320).
    pub fn new(frame_w: u32, frame_h: u32) -> TractResult<Self> {
        let mut model = tract_onnx::onnx().model_for_read(&mut Cursor::new(MODEL_BYTES))?;
        model.set_input_fact(0, f32::fact([1, 3, DET_H, DET_W]).into())?;
        // Pin output order so decode indices are deterministic.
        model.set_output_names(OUTPUT_NAMES)?;
        let model = model.into_optimized()?.into_runnable()?;

        log::info!("YuNet ready: frame {frame_w}x{frame_h} -> det {DET_W}x{DET_H}");
        Ok(Self {
            model,
            det_w: DET_W,
            det_h: DET_H,
            sx: frame_w as f32 / DET_W as f32,
            sy: frame_h as f32 / DET_H as f32,
        })
    }

    /// Run detection on an RGB frame (must match the resolution given to new()).
    pub fn detect(&self, img: &RgbImage) -> TractResult<Vec<FaceDet>> {
        // 1. Downscale to detection size (independent x/y scale; YuNet
        //    tolerates the mild aspect distortion and we map back exactly).
        let resized = image::imageops::resize(
            img,
            self.det_w as u32,
            self.det_h as u32,
            image::imageops::FilterType::Triangle,
        );

        // 2. RGB -> BGR planar f32 tensor [1,3,H,W] (raw 0..255, no normalization)
        let (dw, dh) = (self.det_w, self.det_h);
        let mut arr = tract_ndarray::Array4::<f32>::zeros((1, 3, dh, dw));
        let raw = resized.as_raw();
        for y in 0..dh {
            for x in 0..dw {
                let o = (y * dw + x) * 3;
                arr[[0, 0, y, x]] = raw[o + 2] as f32; // B
                arr[[0, 1, y, x]] = raw[o + 1] as f32; // G
                arr[[0, 2, y, x]] = raw[o] as f32; // R
            }
        }

        let outputs = self.model.run(tvec!(Tensor::from(arr).into()))?;

        // 3. Decode all strides
        let mut cands: Vec<FaceDet> = Vec::new();
        for (si, &stride) in STRIDES.iter().enumerate() {
            let cols = dw / stride;
            let rows = dh / stride;
            let n = cols * rows;

            let cls = outputs[si].to_array_view::<f32>()?;
            let obj = outputs[3 + si].to_array_view::<f32>()?;
            let bbox = outputs[6 + si].to_array_view::<f32>()?;
            let kps = outputs[9 + si].to_array_view::<f32>()?;
            let cls = cls.as_slice().unwrap_or(&[]);
            let obj = obj.as_slice().unwrap_or(&[]);
            let bbox = bbox.as_slice().unwrap_or(&[]);
            let kps = kps.as_slice().unwrap_or(&[]);
            if cls.len() < n || obj.len() < n || bbox.len() < n * 4 || kps.len() < n * 10 {
                log::warn!(
                    "stride {stride}: unexpected output sizes cls={} obj={} bbox={} kps={}",
                    cls.len(), obj.len(), bbox.len(), kps.len()
                );
                continue;
            }

            for i in 0..n {
                let c = cls[i].clamp(0.0, 1.0);
                let o = obj[i].clamp(0.0, 1.0);
                let score = (c * o).sqrt();
                if score < SCORE_THRESHOLD {
                    continue;
                }
                let row = (i / cols) as f32;
                let col = (i % cols) as f32;
                let s = stride as f32;

                let cx = (col + bbox[i * 4]) * s;
                let cy = (row + bbox[i * 4 + 1]) * s;
                let bw = bbox[i * 4 + 2].exp() * s;
                let bh = bbox[i * 4 + 3].exp() * s;

                let mut landmarks = Vec::with_capacity(5);
                for k in 0..5 {
                    let lx = (col + kps[i * 10 + k * 2]) * s * self.sx;
                    let ly = (row + kps[i * 10 + k * 2 + 1]) * s * self.sy;
                    landmarks.push([lx, ly]);
                }

                let (yaw, pitch) = estimate_head_pose(&landmarks);
                cands.push(FaceDet {
                    x: (cx - bw / 2.0) * self.sx,
                    y: (cy - bh / 2.0) * self.sy,
                    w: bw * self.sx,
                    h: bh * self.sy,
                    score,
                    landmarks,
                    yaw_deg: yaw,
                    pitch_deg: pitch,
                });
            }
        }

        // 4. NMS (IoU 0.3), keep top 5 — same as the original detector.
        Ok(nms(cands, NMS_THRESHOLD, TOP_K))
    }
}

/// Coarse yaw/pitch from the 5 YuNet landmarks.
/// yaw: nose x-offset from eye midpoint, normalized by eye distance.
/// pitch: nose vertical position within the eye->mouth span
/// (looking down projects the nose closer to the mouth -> positive pitch).
fn estimate_head_pose(lm: &[[f32; 2]]) -> (f32, f32) {
    if lm.len() < 5 {
        return (0.0, 0.0);
    }
    let (re, le, nose, rm, lmo) = (lm[0], lm[1], lm[2], lm[3], lm[4]);
    let eye_mid = [(re[0] + le[0]) / 2.0, (re[1] + le[1]) / 2.0];
    let mouth_mid = [(rm[0] + lmo[0]) / 2.0, (rm[1] + lmo[1]) / 2.0];
    let eye_dist = ((le[0] - re[0]).powi(2) + (le[1] - re[1]).powi(2))
        .sqrt()
        .max(1.0);

    let yaw = ((nose[0] - eye_mid[0]) / eye_dist * 70.0).clamp(-90.0, 90.0);

    let span = (mouth_mid[1] - eye_mid[1]).max(1.0);
    let nose_ratio = (nose[1] - eye_mid[1]) / span; // neutral ≈ 0.55
    let pitch = ((nose_ratio - 0.55) * 120.0).clamp(-90.0, 90.0);
    (yaw, pitch)
}

fn iou(a: &FaceDet, b: &FaceDet) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.w * a.h + b.w * b.h - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

fn nms(mut dets: Vec<FaceDet>, iou_thresh: f32, top_k: usize) -> Vec<FaceDet> {
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<FaceDet> = Vec::new();
    for d in dets {
        if keep.len() >= top_k {
            break;
        }
        if keep.iter().all(|k| iou(k, &d) <= iou_thresh) {
            keep.push(d);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Model loads, plan optimizes for 1280x720, blank frame -> 0 faces, no panic.
    /// (Real-face accuracy is verified on a machine with a working camera.)
    #[test]
    fn yunet_loads_and_runs_on_blank_frame() {
        let det = FaceDetector::new(1280, 720).expect("model should load");
        assert_eq!((det.det_w, det.det_h), (DET_W, DET_H));
        let blank = RgbImage::new(1280, 720);
        let faces = det.detect(&blank).expect("inference should run");
        assert!(faces.is_empty(), "blank frame must yield no faces");
    }

    /// Gradient frame exercises non-trivial activations through all strides.
    #[test]
    fn yunet_runs_on_gradient_frame() {
        let det = FaceDetector::new(640, 480).expect("model should load");
        let img = RgbImage::from_fn(640, 480, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let faces = det.detect(&img).expect("inference should run");
        assert!(faces.len() <= 5, "NMS must cap at top_k");
    }
}
