//! Diagnostic: find where tract's shape inference blows up on the MoveNet ONNX.
//!
//! Run: cargo run --example pose_diag
//!
//! Prints every typed node's output fact and flags any tensor whose element
//! count is absurd (which is what causes the 2 GiB allocation abort at run
//! time).

use tract_onnx::prelude::*;

fn main() -> TractResult<()> {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "movenet_lightning.onnx".to_string());
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(&name);
    println!("model: {}", path.display());

    println!("\n--- stage 1: parse ---");
    let mut model = tract_onnx::onnx().model_for_path(&path)?;
    println!("parsed ok, {} nodes (inference graph)", model.nodes().len());

    println!("\n--- stage 2: set input fact [1,192,192,3] f32 ---");
    model.set_input_fact(0, f32::fact([1, 192, 192, 3]).into())?;
    println!("input fact set");

    println!("\n--- stage 3: into_typed (shape inference) ---");
    let typed = match model.into_typed() {
        Ok(t) => {
            println!("typed ok, {} nodes", t.nodes().len());
            t
        }
        Err(e) => {
            println!("into_typed FAILED: {e}");
            return Ok(());
        }
    };

    println!("\n--- stage 4: scan node output shapes ---");
    let mut worst: Vec<(u64, usize, String, String)> = Vec::new();
    for node in typed.nodes() {
        for out in &node.outputs {
            let shape_dbg = format!("{:?}", out.fact.shape);
            // Try to get a concrete element count.
            let vol: u64 = out
                .fact
                .shape
                .as_concrete()
                .map(|d| d.iter().map(|&x| x as u64).product())
                .unwrap_or(0);
            if vol > 10_000_000 {
                worst.push((vol, node.id, node.op().name().to_string(), shape_dbg));
            }
        }
    }
    worst.sort_by(|a, b| b.0.cmp(&a.0));
    if worst.is_empty() {
        println!("no node output above 10M elements — shapes look sane");
    } else {
        println!("!! oversized node outputs (elements > 10M):");
        for (vol, id, op, shape) in worst.iter().take(20) {
            println!("   node #{id:<4} {op:<20} elems={vol:<15} shape={shape}");
        }
    }

    println!("\n--- stage 5: into_optimized ---");
    let optimized = match typed.into_optimized() {
        Ok(o) => {
            println!("optimized ok, {} nodes", o.nodes().len());
            o
        }
        Err(e) => {
            println!("into_optimized FAILED: {e}");
            return Ok(());
        }
    };

    println!("\n--- stage 6: into_runnable ---");
    let plan = optimized.into_runnable()?;
    println!("runnable ok");

    println!("\n--- stage 7: run one deterministic frame ---");
    // Deterministic synthetic pattern (mirrored exactly in the Python/ORT
    // cross-check) — a blank frame is a degenerate input that can hide a
    // silently-wrong op, e.g. a Resize that returns its input unchanged.
    let mut arr = tract_ndarray::Array4::<f32>::zeros((1, 192, 192, 3));
    for y in 0..192 {
        for x in 0..192 {
            for c in 0..3 {
                arr[[0, y, x, c]] = ((x * 3 + y * 5 + c * 7) % 256) as f32;
            }
        }
    }
    let out = plan.run(tvec!(Tensor::from(arr).into()))?;
    println!("run ok, {} output(s)", out.len());

    // Per-output statistics, in the same format the Python/ORT cross-check
    // prints, so both can be diffed line by line.
    println!(
        "\n{:<22} {:<18} {:>14} {:>14} {:>14} {:>14}",
        "tensor", "shape", "sum", "mean", "min", "max"
    );
    for (i, o) in out.iter().enumerate() {
        let shape = format!("{:?}", o.shape());
        let vals: Option<Vec<f64>> = match o.datum_type() {
            DatumType::F32 => Some(o.as_slice::<f32>()?.iter().map(|&v| v as f64).collect()),
            DatumType::I64 => Some(o.as_slice::<i64>()?.iter().map(|&v| v as f64).collect()),
            DatumType::I32 => Some(o.as_slice::<i32>()?.iter().map(|&v| v as f64).collect()),
            _ => None,
        };
        match vals {
            Some(v) if !v.is_empty() => {
                let sum: f64 = v.iter().sum();
                let mean = sum / v.len() as f64;
                let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                println!(
                    "out[{:<2}]{:<15} {:<18} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
                    i, "", shape, sum, mean, min, max
                );
            }
            _ => println!(
                "out[{:<2}]{:<15} {:<18} (dt {:?}, stats skipped)",
                i,
                "",
                shape,
                o.datum_type()
            ),
        }
    }

    // Dump all 17 keypoints so the values (not just the shape) can be compared
    // against ONNX Runtime.
    if out[0].len() == 51 && out[0].datum_type() == DatumType::F32 {
        let view = out[0].to_array_view::<f32>()?;
        println!("\n--- keypoints (y, x, score) normalised ---");
        for i in 0..17 {
            println!(
                "  kp{:<2} y={:.6} x={:.6} score={:.6}",
                i,
                view[[0, 0, i, 0]],
                view[[0, 0, i, 1]],
                view[[0, 0, i, 2]]
            );
        }
    }
    Ok(())
}
