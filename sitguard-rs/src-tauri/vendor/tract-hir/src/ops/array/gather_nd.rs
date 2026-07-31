use crate::internal::*;
pub use tract_core::ops::array::GatherNd;

// ---------------------------------------------------------------------------
// SitGuard local patch (upstream bug, still present in tract-hir 0.23.4)
//
// The stock shape-inference rule for GatherND reads the *indices* rank where
// the ONNX spec requires the *data* rank, and then indexes the *indices* shape
// where it should index the *data* shape:
//
//     s.given_2(
//         &inputs[1].shape[indices_rank - 1],
//         &inputs[1].rank,                     // <-- should be inputs[0].rank
//         move |s, n, input_rank| {
//             for i in 0..(input_rank - n) as usize {   // <-- underflows
//                 s.equals(&outputs[0].shape[indices_rank - 1 + i],
//                          &inputs[1].shape[i])?;       // <-- should be inputs[0]
//
// When `n > indices_rank` the subtraction goes negative and `as usize` wraps to
// ~1.8e19, so the loop pushes solver rules until the process aborts with a
// multi-gigabyte allocation failure (not a catchable Err).
//
// MoveNet Lightning hits this on 3 of its 4 GatherND nodes, e.g. data
// [1,48,48,17] with indices [17,4]: 2 - 4 = -2 -> abort.
//
// Per the ONNX spec (batch_dims = 0):
//     output.shape = indices.shape[:-1] ++ data.shape[indices.shape[-1]:]
// so output dim (indices_rank - 1 + i) == data.shape[n + i] for
// i in 0..(data_rank - n).
// ---------------------------------------------------------------------------

impl InferenceRulesOp for GatherNd {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(inputs, 2)?;
        check_output_arity(outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.given(&inputs[1].rank, move |s, indices_rank| {
            let indices_rank = indices_rank as usize;
            for i in 0..(indices_rank - 1) {
                s.equals(&outputs[0].shape[i], &inputs[1].shape[i])?;
            }
            s.given_2(
                &inputs[1].shape[indices_rank - 1],
                &inputs[0].rank,
                move |s, n, data_rank| {
                    if let Ok(n) = n.to_i64() {
                        // Guard against a malformed graph as well: never let the
                        // loop bound underflow.
                        let n = n.max(0);
                        let data_rank = data_rank as i64;
                        if n <= data_rank {
                            for i in 0..(data_rank - n) as usize {
                                s.equals(
                                    &outputs[0].shape[indices_rank - 1 + i],
                                    &inputs[0].shape[n as usize + i],
                                )?;
                            }
                        }
                    }
                    Ok(())
                },
            )
        })
    }

    as_op!();
    to_typed!();
}
