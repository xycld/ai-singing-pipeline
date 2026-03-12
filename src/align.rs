use crate::{audio, Error, Result};
use std::path::Path;

const FS: i32 = 22050;

#[derive(Debug, Clone)]
pub struct AlignConfig {
    pub safe_mode: bool,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self { safe_mode: true }
    }
}

/// Load user and reference audio, time-align at phrase level, and write the result.
///
/// Strategy: compute a per-segment time offset via onset cross-correlation,
/// smooth into a slowly varying offset curve, then apply as a variable delay.
/// No time-stretching is done — the offset changes so gradually (~0.1%/s)
/// that the implicit resampling is inaudible.
pub fn process_align(
    user_path: &Path,
    ref_path: &Path,
    output_path: &Path,
    output_sr: i32,
    config: &AlignConfig,
) -> Result<()> {
    eprintln!("1. Loading audio ({} Hz)...", FS);
    let (user_orig, user_sr) =
        audio::read_wav(user_path).map_err(|e| Error::Audio(e.to_string()))?;
    let (ref_orig, ref_sr) =
        audio::read_wav(ref_path).map_err(|e| Error::Audio(e.to_string()))?;

    let audio_user = if user_sr != FS {
        audio::resample(&user_orig, user_sr, FS)
    } else {
        user_orig
    };
    let audio_ref = if ref_sr != FS {
        audio::resample(&ref_orig, ref_sr, FS)
    } else {
        ref_orig
    };

    eprintln!(
        "   User: {:.1}s, Ref: {:.1}s",
        audio_user.len() as f64 / FS as f64,
        audio_ref.len() as f64 / FS as f64,
    );

    let target_len = audio_ref.len();

    // --- Per-segment cross-correlation offsets ---
    eprintln!("2. Computing per-segment offsets...");
    let eval_hop = 256;
    let o_user = audio::onset_strength(&audio_user, FS, eval_hop);
    let o_ref = audio::onset_strength(&audio_ref, FS, eval_hop);
    let n_eval = o_user.len().min(o_ref.len());

    let win_eval = 20.max((4.0 * FS as f64 / eval_hop as f64) as usize);
    let hop_eval = (2.0 * FS as f64 / eval_hop as f64) as usize;
    let max_lag_eval = (1.5 * FS as f64 / eval_hop as f64) as usize;

    // Each entry: (center_sample, offset_seconds)
    let mut raw_offsets: Vec<(f64, f64)> = Vec::new();
    let mut s = 0;
    while s + win_eval <= n_eval {
        let a = &o_ref[s..s + win_eval];
        let b = &o_user[s..s + win_eval];
        if let Some(best_lag) = cross_correlate_peak(a, b, max_lag_eval) {
            let center_sample = (s + win_eval / 2) as f64 * eval_hop as f64;
            let offset_sec = -best_lag as f64 * eval_hop as f64 / FS as f64;
            raw_offsets.push((center_sample, offset_sec));
        }
        s += hop_eval;
    }

    if raw_offsets.is_empty() {
        eprintln!("   No offsets computed, copying input unchanged.");
        let y_out = if output_sr != FS {
            audio::resample(&audio_user, FS, output_sr)
        } else {
            audio_user
        };
        audio::write_wav(output_path, &y_out, output_sr)
            .map_err(|e| Error::Audio(e.to_string()))?;
        return Ok(());
    }

    // Global stats for logging.
    let all_offsets: Vec<f64> = raw_offsets.iter().map(|&(_, o)| o).collect();
    let mut sorted = all_offsets.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let global_median = sorted[sorted.len() / 2];
    let good = all_offsets
        .iter()
        .filter(|&&o| (o - global_median).abs() < 0.20)
        .count();
    let pct_good = good as f64 / all_offsets.len() as f64 * 100.0;

    eprintln!("   Segments: {}", raw_offsets.len());
    eprintln!("   Global median: {:+.3}s", global_median);
    eprintln!("   Consistent segments: {:.0}%", pct_good);

    // --- Smooth offset curve ---
    // 1. Median-filter to remove outliers (window = 7 segments).
    let filtered = median_filter(&all_offsets, 7);
    // 2. Build smoothed (center_sample, offset) pairs.
    let smooth_offsets: Vec<(f64, f64)> = raw_offsets
        .iter()
        .zip(filtered.iter())
        .map(|(&(center, _), &filt_off)| (center, filt_off))
        .collect();

    eprintln!(
        "   Offset range: {:+.3}s .. {:+.3}s",
        filtered.iter().cloned().fold(f64::INFINITY, f64::min),
        filtered.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    // --- Apply variable delay ---
    eprintln!("3. Applying phrase-level alignment...");
    let mut y_aligned = vec![0.0f64; target_len];

    for i in 0..target_len {
        let pos = i as f64;
        // Interpolate the offset at this sample position.
        let offset_samples = interpolate_offset(&smooth_offsets, pos) * FS as f64;

        // Source position: output position minus the offset.
        let src = pos - offset_samples;
        if src < 0.0 || src >= (audio_user.len() - 1) as f64 {
            continue; // silence
        }
        // Linear interpolation in source.
        let lo = src as usize;
        let hi = (lo + 1).min(audio_user.len() - 1);
        let frac = src - lo as f64;
        y_aligned[i] = audio_user[lo] * (1.0 - frac) + audio_user[hi] * frac;
    }

    if config.safe_mode {
        let mut base = audio_user.clone();
        base.resize(target_len, 0.0);
        base.truncate(target_len);

        let base_score = alignment_score(&base, &audio_ref, eval_hop);
        let aligned_score = alignment_score(&y_aligned, &audio_ref, eval_hop);
        eprintln!(
            "   [Eval] Baseline={:.3}s, Aligned={:.3}s",
            base_score, aligned_score
        );
    }

    let y_out = if output_sr != FS {
        eprintln!("4. Resampling {} -> {} Hz...", FS, output_sr);
        audio::resample(&y_aligned, FS, output_sr)
    } else {
        y_aligned.clone()
    };

    audio::write_wav(output_path, &y_out, output_sr)
        .map_err(|e| Error::Audio(e.to_string()))?;
    eprintln!("Alignment done! -> {}", output_path.display());

    // Verification mix: user vocal + quiet reference for A/B checking.
    let audio_ref_out = if output_sr != FS {
        audio::resample(&audio_ref, FS, output_sr)
    } else {
        audio_ref
    };
    let mix_len = y_out.len().min(audio_ref_out.len());
    let mut y_mix: Vec<f64> = (0..mix_len)
        .map(|i| y_out[i] + audio_ref_out[i] * 0.15)
        .collect();
    let mix_peak = audio::peak(&y_mix);
    if mix_peak > 1.0 {
        for s in y_mix.iter_mut() {
            *s /= mix_peak;
        }
    }
    let mix_path = {
        let stem = output_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let ext = output_path
            .extension()
            .unwrap_or_default()
            .to_string_lossy();
        output_path
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{}_mix.{}", stem, ext))
    };
    audio::write_wav(&mix_path, &y_mix, output_sr)
        .map_err(|e| Error::Audio(e.to_string()))?;
    eprintln!("Verification mix -> {}", mix_path.display());

    Ok(())
}

/// Linearly interpolate the offset (in seconds) at a given sample position.
fn interpolate_offset(offsets: &[(f64, f64)], pos: f64) -> f64 {
    if offsets.is_empty() {
        return 0.0;
    }
    if offsets.len() == 1 {
        return offsets[0].1;
    }
    // Before first anchor: use first value.
    if pos <= offsets[0].0 {
        return offsets[0].1;
    }
    // After last anchor: use last value.
    if pos >= offsets.last().unwrap().0 {
        return offsets.last().unwrap().1;
    }
    // Find bracketing pair and lerp.
    for w in offsets.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if pos >= x0 && pos <= x1 {
            let t = if (x1 - x0).abs() > 1e-6 {
                (pos - x0) / (x1 - x0)
            } else {
                0.5
            };
            return y0 + t * (y1 - y0);
        }
    }
    offsets.last().unwrap().1
}

/// 1-D median filter with the given window radius.
fn median_filter(data: &[f64], window: usize) -> Vec<f64> {
    let half = window / 2;
    data.iter()
        .enumerate()
        .map(|(i, _)| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(data.len());
            let mut win: Vec<f64> = data[lo..hi].to_vec();
            win.sort_by(|a, b| a.partial_cmp(b).unwrap());
            win[win.len() / 2]
        })
        .collect()
}

fn cross_correlate_peak(a: &[f64], b: &[f64], max_lag: usize) -> Option<isize> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let n = a.len();
    let m = b.len();

    let mut best_lag: isize = 0;
    let mut best_corr = f64::NEG_INFINITY;

    let max_lag = max_lag as isize;
    for lag in -max_lag..=max_lag {
        let mut corr = 0.0;
        for i in 0..n {
            let j = i as isize + lag;
            if j >= 0 && (j as usize) < m {
                corr += a[i] * b[j as usize];
            }
        }
        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    Some(best_lag)
}

fn alignment_score(y_test: &[f64], y_ref: &[f64], hop: usize) -> f64 {
    let o_test = audio::onset_strength(y_test, FS, hop);
    let o_ref = audio::onset_strength(y_ref, FS, hop);
    let n = o_test.len().min(o_ref.len());
    if n < 10 {
        return f64::INFINITY;
    }

    let win = 20.max((8.0 * FS as f64 / hop as f64) as usize).min(n);
    let max_lag = 5.max((1.25 * FS as f64 / hop as f64) as usize);

    let starts: Vec<usize> = if n <= win {
        vec![0]
    } else {
        (0..5).map(|i| i * (n - win) / 4).collect()
    };

    let mut lags_s = Vec::new();
    for &s in &starts {
        if s + win > n {
            continue;
        }
        if let Some(lag) = cross_correlate_peak(&o_ref[s..s + win], &o_test[s..s + win], max_lag) {
            lags_s.push((lag as f64 * hop as f64 / FS as f64).abs());
        }
    }

    if lags_s.is_empty() {
        f64::INFINITY
    } else {
        lags_s.iter().sum::<f64>() / lags_s.len() as f64
    }
}
