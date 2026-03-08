//! Time-alignment via chroma features + DTW + RubberBand CLI.
//!
//! Port of align.py: extracts chroma features, runs band-constrained DTW,
//! evaluates onset correlation, then either applies a global shift (Strategy A)
//! or time-stretches via RubberBand CLI subprocess (Strategy B).

use crate::{audio, Error, Result};
use std::io::Write;
use std::path::Path;

/// Internal sample rate for alignment processing (matches librosa default).
const FS: i32 = 22050;
/// Chroma feature rate in Hz.
const FEATURE_RATE: usize = 50;

/// Configuration for the alignment stage.
#[derive(Debug, Clone)]
pub struct AlignConfig {
    pub safe_mode: bool,
    pub rubberband_bin: std::path::PathBuf,
}

impl Default for AlignConfig {
    fn default() -> Self {
        Self {
            safe_mode: true,
            rubberband_bin: std::path::PathBuf::from("rubberband"),
        }
    }
}

/// Run time-alignment: load user + ref, align, write output.
pub fn process_align(
    user_path: &Path,
    ref_path: &Path,
    output_path: &Path,
    output_sr: i32,
    config: &AlignConfig,
) -> Result<()> {
    // 1. Load and resample to 22050 Hz for analysis
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

    // 2. Extract chroma features
    eprintln!("2. Extracting chroma features...");
    let chroma_user = extract_chroma(&audio_user, FS, FEATURE_RATE);
    let chroma_ref = extract_chroma(&audio_ref, FS, FEATURE_RATE);
    eprintln!(
        "   User: {} frames, Ref: {} frames",
        chroma_user.len(),
        chroma_ref.len()
    );

    // 3. DTW alignment
    eprintln!("3. DTW alignment...");
    let step_weights = [1.5, 1.5, 2.0];
    let band_width = (10.0 * FEATURE_RATE as f64) as usize; // 10s band
    let wp = dtw_band(&chroma_user, &chroma_ref, &step_weights, band_width);
    eprintln!("   Path length: {}", wp.len());

    // Convert warping path to sample positions
    let time_map: Vec<(f64, f64)> = wp
        .iter()
        .map(|&(i, j)| {
            let src = i as f64 / FEATURE_RATE as f64 * FS as f64;
            let tgt = j as f64 / FEATURE_RATE as f64 * FS as f64;
            (src, tgt)
        })
        .collect();

    let target_len = audio_ref.len();

    // 4. Adaptive alignment: onset cross-correlation to choose strategy
    eprintln!("4. Adaptive alignment...");
    let eval_hop = 256;
    let o_user = audio::onset_strength(&audio_user, FS, eval_hop);
    let o_ref = audio::onset_strength(&audio_ref, FS, eval_hop);
    let n_eval = o_user.len().min(o_ref.len());

    let win_eval = 20.max((4.0 * FS as f64 / eval_hop as f64) as usize);
    let hop_eval = (2.0 * FS as f64 / eval_hop as f64) as usize;
    let max_lag_eval = (1.5 * FS as f64 / eval_hop as f64) as usize;

    let mut segment_lags: Vec<f64> = Vec::new();
    let mut s = 0;
    while s + win_eval <= n_eval {
        let a = &o_ref[s..s + win_eval];
        let b = &o_user[s..s + win_eval];
        if let Some(best_lag) = cross_correlate_peak(a, b, max_lag_eval) {
            // Negate: our correlate convention is opposite to scipy's.
            // Positive best_lag means user is ahead of ref, needs negative shift.
            segment_lags.push(-best_lag as f64 * eval_hop as f64 / FS as f64);
        }
        s += hop_eval;
    }

    let (median_offset, pct_good) = if segment_lags.is_empty() {
        (0.0, 0.0)
    } else {
        let mut sorted = segment_lags.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let residuals: Vec<f64> = segment_lags.iter().map(|&l| (l - median).abs()).collect();
        let good = residuals.iter().filter(|&&r| r < 0.20).count();
        (median, good as f64 / segment_lags.len() as f64 * 100.0)
    };

    eprintln!("   Global offset: {:+.3}s", median_offset);
    eprintln!("   Aligned segments: {:.0}% (residual <200ms)", pct_good);

    let y_aligned = if pct_good >= 75.0 {
        // Strategy A: global shift only — zero time-stretch
        eprintln!("   -> Strategy A: global shift {:+.3}s", median_offset);
        let shift_samples = (median_offset * FS as f64).round() as isize;
        apply_global_shift(&audio_user, shift_samples)
    } else {
        // Strategy B: SmartAlign + RubberBand R3
        eprintln!("   -> Strategy B: SmartAlign + RubberBand");
        let step_b = FEATURE_RATE / 2; // ~2 anchors/second
        let lo = (0.12 * FS as f64) as f64;
        let hi = (0.40 * FS as f64) as f64;

        let mut timemap: Vec<(usize, usize)> = vec![(0, 0)];
        let mut prev_src = 0usize;
        let mut prev_tgt = 0usize;

        let anchor_indices: Vec<usize> = (0..time_map.len())
            .step_by(step_b.max(1))
            .skip(1) // skip first
            .collect();

        for &idx in &anchor_indices {
            if idx >= time_map.len() - 1 {
                break;
            }
            let (src, dtw_tgt) = time_map[idx];
            let offset = (src - dtw_tgt).abs();

            let alpha = if offset <= lo {
                0.0
            } else if offset >= hi {
                1.0
            } else {
                (offset - lo) / (hi - lo)
            };

            let blended = src + alpha * (dtw_tgt - src);
            let src_int = (src.round() as usize).clamp(1, audio_user.len() - 1);
            let tgt_int = (blended.round() as usize).clamp(1, target_len - 1);

            if src_int > prev_src && tgt_int > prev_tgt {
                timemap.push((src_int, tgt_int));
                prev_src = src_int;
                prev_tgt = tgt_int;
            }
        }
        timemap.push((audio_user.len(), target_len));
        eprintln!("   Anchors: {}", timemap.len());

        rubberband_timemap_stretch(
            &audio_user,
            FS,
            &timemap,
            &config.rubberband_bin,
        )?
    };

    // 5. Length matching
    let mut y_aligned = y_aligned;
    if y_aligned.len() > target_len {
        y_aligned.truncate(target_len);
    } else {
        y_aligned.resize(target_len, 0.0);
    }

    // 6. Evaluation
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

    // 7. Resample to output SR + save
    let y_out = if output_sr != FS {
        eprintln!("5. Resampling {} -> {} Hz...", FS, output_sr);
        audio::resample(&y_aligned, FS, output_sr)
    } else {
        y_aligned.clone()
    };

    audio::write_wav(output_path, &y_out, output_sr)
        .map_err(|e| Error::Audio(e.to_string()))?;
    eprintln!("Alignment done! -> {}", output_path.display());

    // Verification mix: user_aligned + ref * 0.15
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

// ---------------------------------------------------------------------------
// Chroma feature extraction
// ---------------------------------------------------------------------------

/// Extract chroma features (12 pitch classes) at the given feature rate.
fn extract_chroma(samples: &[f64], sr: i32, feature_rate: usize) -> Vec<[f64; 12]> {
    let hop_size = sr as usize / feature_rate;
    let window_size = 4096;
    let frames = audio::stft(samples, window_size, hop_size);
    let bin_hz = sr as f64 / window_size as f64;

    frames
        .iter()
        .map(|frame| {
            let mut chroma = [0.0f64; 12];
            for (k, c) in frame.iter().enumerate() {
                let freq = k as f64 * bin_hz;
                if freq < 50.0 || freq > 5000.0 {
                    continue;
                }
                // Map frequency to pitch class (MIDI note mod 12)
                let midi = 69.0 + 12.0 * (freq / 440.0).log2();
                let pitch_class = ((midi.round() as i32 % 12 + 12) % 12) as usize;
                let mag = c.norm();
                chroma[pitch_class] += mag * mag;
            }
            // L2 normalize
            let norm = chroma.iter().sum::<f64>().sqrt();
            if norm > 1e-10 {
                for v in chroma.iter_mut() {
                    *v /= norm;
                }
            }
            chroma
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Band-constrained DTW
// ---------------------------------------------------------------------------

/// Cosine distance between two chroma vectors.
fn chroma_distance(a: &[f64; 12], b: &[f64; 12]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

/// Band-constrained DTW with step weights.
///
/// Returns the warping path as (user_frame, ref_frame) pairs, sorted ascending.
fn dtw_band(
    features_a: &[[f64; 12]],
    features_b: &[[f64; 12]],
    step_weights: &[f64; 3],
    band_width: usize,
) -> Vec<(usize, usize)> {
    let n = features_a.len();
    let m = features_b.len();
    if n == 0 || m == 0 {
        return vec![];
    }

    // Use a banded cost matrix to reduce memory: store only band_width*2+1 columns per row.
    // For simplicity with moderate sizes, use full matrix if feasible, else banded.
    let use_full = (n as u64) * (m as u64) < 100_000_000; // ~800MB limit

    if use_full {
        dtw_full(features_a, features_b, step_weights)
    } else {
        dtw_banded(features_a, features_b, step_weights, band_width)
    }
}

fn dtw_full(
    a: &[[f64; 12]],
    b: &[[f64; 12]],
    w: &[f64; 3],
) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    let inf = f64::INFINITY;

    // Cost matrix
    let mut d = vec![vec![inf; m]; n];
    d[0][0] = chroma_distance(&a[0], &b[0]);

    // Fill first row and column
    for j in 1..m {
        d[0][j] = d[0][j - 1] + chroma_distance(&a[0], &b[j]) * w[1];
    }
    for i in 1..n {
        d[i][0] = d[i - 1][0] + chroma_distance(&a[i], &b[0]) * w[0];
    }

    // Fill rest
    for i in 1..n {
        for j in 1..m {
            let cost = chroma_distance(&a[i], &b[j]);
            let c1 = d[i - 1][j] + cost * w[0]; // vertical step
            let c2 = d[i][j - 1] + cost * w[1]; // horizontal step
            let c3 = d[i - 1][j - 1] + cost * w[2]; // diagonal step
            d[i][j] = c1.min(c2).min(c3);
        }
    }

    // Backtrack
    backtrack(&d, n, m)
}

fn dtw_banded(
    a: &[[f64; 12]],
    b: &[[f64; 12]],
    w: &[f64; 3],
    band: usize,
) -> Vec<(usize, usize)> {
    let n = a.len();
    let m = b.len();
    let inf = f64::INFINITY;
    let bw = band.max(1);

    // For banded DTW, we store a strip of width 2*bw+1 per row
    let strip_w = 2 * bw + 1;
    let mut d = vec![vec![inf; strip_w]; n];

    // Map (i, j) -> strip index: j_local = j - (j_center - bw)
    // where j_center = i * m / n (diagonal)
    let j_center = |i: usize| -> usize { (i as u64 * m as u64 / n as u64) as usize };
    let j_range = |i: usize| -> (usize, usize) {
        let center = j_center(i);
        let lo = center.saturating_sub(bw);
        let hi = (center + bw).min(m - 1);
        (lo, hi)
    };
    let to_local = |i: usize, j: usize| -> Option<usize> {
        let (lo, hi) = j_range(i);
        if j >= lo && j <= hi {
            Some(j - lo)
        } else {
            None
        }
    };

    // Initialize (0, 0)
    if let Some(jl) = to_local(0, 0) {
        d[0][jl] = chroma_distance(&a[0], &b[0]);
    }

    // Fill
    for i in 0..n {
        let (jlo, jhi) = j_range(i);
        for j in jlo..=jhi {
            if i == 0 && j == 0 {
                continue;
            }
            let jl = j - jlo;
            let cost = chroma_distance(&a[i], &b[j]);

            let mut best = inf;
            // Vertical: (i-1, j)
            if i > 0 {
                if let Some(pjl) = to_local(i - 1, j) {
                    best = best.min(d[i - 1][pjl] + cost * w[0]);
                }
            }
            // Horizontal: (i, j-1)
            if j > 0 {
                if let Some(pjl) = to_local(i, j - 1) {
                    best = best.min(d[i][pjl] + cost * w[1]);
                }
            }
            // Diagonal: (i-1, j-1)
            if i > 0 && j > 0 {
                if let Some(pjl) = to_local(i - 1, j - 1) {
                    best = best.min(d[i - 1][pjl] + cost * w[2]);
                }
            }

            d[i][jl] = best;
        }
    }

    // Backtrack through banded matrix
    let mut path = Vec::new();
    let mut i = n - 1;
    let mut j = m - 1;
    path.push((i, j));

    while i > 0 || j > 0 {
        let mut best = (f64::INFINITY, i, j);

        if i > 0 {
            if let Some(pjl) = to_local(i - 1, j) {
                if d[i - 1][pjl] < best.0 {
                    best = (d[i - 1][pjl], i - 1, j);
                }
            }
        }
        if j > 0 {
            if let Some(pjl) = to_local(i, j - 1) {
                if d[i][pjl] < best.0 {
                    best = (d[i][pjl], i, j - 1);
                }
            }
        }
        if i > 0 && j > 0 {
            if let Some(pjl) = to_local(i - 1, j - 1) {
                if d[i - 1][pjl] < best.0 {
                    best = (d[i - 1][pjl], i - 1, j - 1);
                }
            }
        }

        i = best.1;
        j = best.2;
        path.push((i, j));

        if i == 0 && j == 0 {
            break;
        }
    }

    path.reverse();
    path
}

fn backtrack(d: &[Vec<f64>], n: usize, m: usize) -> Vec<(usize, usize)> {
    let mut path = Vec::new();
    let mut i = n - 1;
    let mut j = m - 1;
    path.push((i, j));

    while i > 0 || j > 0 {
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let candidates = [
                (d[i - 1][j], i - 1, j),
                (d[i][j - 1], i, j - 1),
                (d[i - 1][j - 1], i - 1, j - 1),
            ];
            let best = candidates
                .iter()
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap();
            i = best.1;
            j = best.2;
        }
        path.push((i, j));
    }

    path.reverse();
    path
}

// ---------------------------------------------------------------------------
// Onset cross-correlation
// ---------------------------------------------------------------------------

/// Find the lag (in frames) that maximizes cross-correlation between two signals.
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

/// Alignment score: mean absolute onset lag across sliding windows.
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

// ---------------------------------------------------------------------------
// Time-stretching helpers
// ---------------------------------------------------------------------------

/// Apply a simple global sample shift (Strategy A).
fn apply_global_shift(audio: &[f64], shift_samples: isize) -> Vec<f64> {
    if shift_samples > 0 {
        let mut out = vec![0.0; shift_samples as usize];
        out.extend_from_slice(audio);
        out
    } else if shift_samples < 0 {
        let skip = (-shift_samples) as usize;
        if skip < audio.len() {
            audio[skip..].to_vec()
        } else {
            vec![]
        }
    } else {
        audio.to_vec()
    }
}

/// Time-stretch audio using RubberBand CLI with a timemap file.
fn rubberband_timemap_stretch(
    audio: &[f64],
    sr: i32,
    timemap: &[(usize, usize)],
    rubberband_bin: &Path,
) -> Result<Vec<f64>> {
    use std::process::Command;

    // Write input audio to temp file
    let tmp_dir = std::env::temp_dir();
    let input_path = tmp_dir.join("_align_rb_input.wav");
    let output_path = tmp_dir.join("_align_rb_output.wav");
    let map_path = tmp_dir.join("_align_rb_timemap.txt");

    audio::write_wav(&input_path, audio, sr).map_err(|e| Error::Audio(e.to_string()))?;

    // Write timemap file (format: "source_frame target_frame\n")
    {
        let mut f = std::fs::File::create(&map_path)?;
        for &(src, tgt) in timemap {
            writeln!(f, "{} {}", src, tgt)?;
        }
    }

    // Run RubberBand
    let result = Command::new(rubberband_bin)
        .arg("--timemap")
        .arg(&map_path)
        .arg("-3") // R3 engine (faster)
        .arg("--formant") // preserve formants
        .arg(&input_path)
        .arg(&output_path)
        .output();

    // Clean up input + map regardless of result
    let _ = std::fs::remove_file(&input_path);
    let _ = std::fs::remove_file(&map_path);

    match result {
        Ok(output) => {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                let _ = std::fs::remove_file(&output_path);
                return Err(Error::Alignment(format!(
                    "rubberband failed: {}",
                    stderr.trim()
                )));
            }
        }
        Err(e) => {
            return Err(Error::Alignment(format!(
                "rubberband not found at '{}': {}",
                rubberband_bin.display(),
                e
            )));
        }
    }

    // Read output
    let (aligned, _) =
        audio::read_wav(&output_path).map_err(|e| Error::Audio(e.to_string()))?;
    let _ = std::fs::remove_file(&output_path);

    Ok(aligned)
}
