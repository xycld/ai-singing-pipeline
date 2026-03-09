//! WSOLA (Waveform Similarity Overlap-Add) time-stretcher.
//!
//! Stretches or compresses audio segments without changing pitch,
//! replacing the external RubberBand CLI dependency.
//!
//! Algorithm: for each output frame, find the best-matching position in the
//! input (within a tolerance window) using cross-correlation, then overlap-add
//! with a Hann window.

/// Time-stretch mono audio according to a timemap of anchor points.
///
/// Each anchor is `(source_sample, target_sample)` — the sample at position
/// `source_sample` in the input should land at `target_sample` in the output.
/// Anchors must be sorted and strictly increasing in both dimensions.
pub fn timemap_stretch(input: &[f64], timemap: &[(usize, usize)]) -> Vec<f64> {
    if timemap.len() < 2 || input.is_empty() {
        return input.to_vec();
    }

    let last_target = timemap.last().unwrap().1;
    let mut output = vec![0.0f64; last_target];

    // Process each segment between consecutive anchors
    for pair in timemap.windows(2) {
        let (src_start, tgt_start) = pair[0];
        let (src_end, tgt_end) = pair[1];

        let src_len = src_end.saturating_sub(src_start);
        let tgt_len = tgt_end.saturating_sub(tgt_start);

        if src_len == 0 || tgt_len == 0 {
            continue;
        }

        let src_segment = if src_end <= input.len() {
            &input[src_start..src_end]
        } else {
            &input[src_start..input.len().min(src_end)]
        };

        let ratio = tgt_len as f64 / src_len as f64;
        let stretched = wsola_stretch(src_segment, ratio);

        // Copy into output (truncate if needed)
        let copy_len = stretched.len().min(tgt_len).min(output.len() - tgt_start);
        output[tgt_start..tgt_start + copy_len].copy_from_slice(&stretched[..copy_len]);
    }

    output
}

/// Core WSOLA: stretch a mono segment by the given ratio.
fn wsola_stretch(input: &[f64], ratio: f64) -> Vec<f64> {
    if input.len() < 64 {
        // Too short for WSOLA — just resample linearly
        return linear_resample(input, ratio);
    }

    // For ratio very close to 1.0, skip processing
    if (ratio - 1.0).abs() < 0.005 {
        return input.to_vec();
    }

    let window_size = 1024usize.min(input.len());
    let hop_out = window_size / 2; // output hop
    let tolerance = window_size / 4; // search window for best overlap

    let output_len = (input.len() as f64 * ratio).round() as usize;
    let mut output = vec![0.0f64; output_len];
    let mut window_sum = vec![0.0f64; output_len];

    let window = hann(window_size);

    // Input position (fractional, advances by hop_in per frame)
    let hop_in = hop_out as f64 / ratio;
    let mut in_pos = 0.0f64;
    let mut out_pos = 0usize;

    // First frame: no cross-correlation needed
    let mut best_offset: isize = 0;

    while out_pos + window_size <= output_len {
        // Nominal input position
        let nominal = (in_pos + best_offset as f64).round() as isize;

        // Find best match within tolerance window
        let search_center = nominal.max(0) as usize;
        let search_start = search_center.saturating_sub(tolerance);
        let search_end = (search_center + tolerance).min(input.len().saturating_sub(window_size));

        if search_start >= search_end {
            // Near the end — just use nominal position
            let src = (nominal.max(0) as usize).min(input.len().saturating_sub(window_size));
            overlap_add(&mut output, &mut window_sum, &input[src..src + window_size], &window, out_pos);
            in_pos += hop_in;
            out_pos += hop_out;
            continue;
        }

        // Cross-correlate to find best alignment
        // Use the previous output region as the reference for continuity
        let best_pos = if out_pos >= hop_out && out_pos + window_size <= output_len {
            // Compare candidate input frames against what's already in the output
            // (overlap region from previous frame)
            let mut best_corr = f64::NEG_INFINITY;
            let mut best_p = search_start;

            for p in search_start..=search_end {
                if p + window_size > input.len() {
                    break;
                }
                // Correlate the overlap region (first hop_out samples of window)
                let corr = cross_correlate_short(
                    &output[out_pos..out_pos + hop_out.min(window_size)],
                    &input[p..p + hop_out.min(window_size)],
                );
                if corr > best_corr {
                    best_corr = corr;
                    best_p = p;
                }
            }
            best_p
        } else {
            search_start
        };

        if best_pos + window_size <= input.len() {
            overlap_add(&mut output, &mut window_sum, &input[best_pos..best_pos + window_size], &window, out_pos);
        }

        // Track offset drift for next frame
        best_offset = best_pos as isize - nominal;

        in_pos += hop_in;
        out_pos += hop_out;
    }

    // Normalize by window overlap
    for i in 0..output_len {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }

    output
}

fn overlap_add(output: &mut [f64], window_sum: &mut [f64], frame: &[f64], window: &[f64], pos: usize) {
    let len = frame.len().min(output.len() - pos);
    for i in 0..len {
        output[pos + i] += frame[i] * window[i];
        window_sum[pos + i] += window[i];
    }
}

fn cross_correlate_short(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..n {
        sum += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom > 1e-12 { sum / denom } else { 0.0 }
}

fn hann(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos()))
        .collect()
}

/// Fallback for very short segments: simple linear interpolation resampling.
fn linear_resample(input: &[f64], ratio: f64) -> Vec<f64> {
    let out_len = (input.len() as f64 * ratio).round() as usize;
    if out_len == 0 {
        return vec![];
    }
    (0..out_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let lo = (src as usize).min(input.len() - 1);
            let hi = (lo + 1).min(input.len() - 1);
            let frac = src - lo as f64;
            input[lo] * (1.0 - frac) + input[hi] * frac
        })
        .collect()
}
