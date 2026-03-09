use crate::dsp::{Biquad, Compressor, Gate, Limiter, Reverb};
use crate::{audio, Error, Result};
use std::path::Path;

/// Mixing stage configuration.
#[derive(Debug, Clone)]
pub struct MixConfig {
    pub enable_emotion: bool,
    pub emotion_strength: f64,
}

impl Default for MixConfig {
    fn default() -> Self {
        Self {
            enable_emotion: true,
            emotion_strength: 1.0,
        }
    }
}

struct ReferenceAnalysis {
    target_balance_db: f64,
    target_brightness: f64,
}

/// Run the full mixing pipeline, producing vocal, main mix, and comparison outputs.
pub fn process_mix(
    user_path: &Path,
    inst_path: &Path,
    ref_path: &Path,
    output_path: &Path,
    config: &MixConfig,
) -> Result<()> {
    eprintln!("1. Analyzing reference...");
    let analysis = analyze_reference(ref_path, inst_path)?;

    eprintln!("2. Loading and processing user vocal...");
    let (user_channels, user_sr) =
        audio::read_wav_stereo(user_path).map_err(|e| Error::Audio(e.to_string()))?;
    let mut user_channels = audio::ensure_stereo(user_channels);

    if config.enable_emotion {
        let (ref_channels, ref_sr) =
            audio::read_wav_stereo(ref_path).map_err(|e| Error::Audio(e.to_string()))?;
        let mut ref_channels = audio::ensure_stereo(ref_channels);
        clamp_channels(&mut ref_channels);

        if ref_sr != user_sr {
            for ch in ref_channels.iter_mut() {
                *ch = audio::resample(ch, ref_sr, user_sr);
            }
        }

        user_channels = apply_emotion_alignment(
            &user_channels,
            &ref_channels,
            user_sr,
            config.emotion_strength,
        );
    }

    let user_mono = audio::to_mono(&user_channels);
    let user_db = audio::rms_db(&user_mono);

    let user_brightness = audio::spectral_centroid(&user_mono, user_sr);
    eprintln!("   User brightness: {:.0} Hz", user_brightness);

    let mut processed_channels = user_channels.clone();
    apply_smart_chain(
        &mut processed_channels,
        user_sr,
        analysis.target_brightness,
        user_brightness,
        user_db,
    );

    let (inst_channels, inst_sr) =
        audio::read_wav_stereo(inst_path).map_err(|e| Error::Audio(e.to_string()))?;
    let mut inst_channels = audio::ensure_stereo(inst_channels);
    clamp_channels(&mut inst_channels);

    if inst_sr != user_sr {
        for ch in inst_channels.iter_mut() {
            *ch = audio::resample(ch, inst_sr, user_sr);
        }
    }

    let processed_mono = audio::to_mono(&processed_channels);
    let processed_db = audio::rms_db(&processed_mono);
    let inst_mono = audio::to_mono(&inst_channels);
    let current_inst_db = audio::rms_db(&inst_mono);

    let target_user_db = current_inst_db + analysis.target_balance_db;
    let gain_needed_db = target_user_db - processed_db;
    let gain_linear = 10.0_f64.powf(gain_needed_db / 20.0);

    eprintln!(
        "   Auto-level: {:.1}dB gain to match reference balance",
        gain_needed_db
    );

    for ch in processed_channels.iter_mut() {
        for s in ch.iter_mut() {
            *s *= gain_linear;
        }
    }

    eprintln!("3. Mixing...");
    let min_len = processed_channels[0]
        .len()
        .min(inst_channels[0].len());

    for ch in processed_channels.iter_mut() {
        ch.truncate(min_len);
    }
    for ch in inst_channels.iter_mut() {
        ch.truncate(min_len);
    }

    let mut limiter;

    let mut vocal_only = processed_channels.clone();
    for ch in vocal_only.iter_mut() {
        limiter = Limiter::new(user_sr as f64, -1.0);
        limiter.process_mono(ch);
    }
    let vocal_path = with_suffix(output_path, "_vocal");
    audio::write_wav_stereo(&vocal_path, &vocal_only, user_sr)
        .map_err(|e| Error::Audio(e.to_string()))?;
    eprintln!("   Vocal only -> {}", vocal_path.display());

    let mix_channels = add_channels(&processed_channels, &inst_channels);
    let mut final_mix = mix_channels;
    for ch in final_mix.iter_mut() {
        limiter = Limiter::new(user_sr as f64, -1.0);
        limiter.process_mono(ch);
    }
    audio::write_wav_stereo(output_path, &final_mix, user_sr)
        .map_err(|e| Error::Audio(e.to_string()))?;
    eprintln!("   Mix -> {}", output_path.display());

    if let Ok((ref_ch, ref_sr)) = audio::read_wav_stereo(ref_path) {
        let mut ref_ch = audio::ensure_stereo(ref_ch);
        clamp_channels(&mut ref_ch);
        if ref_sr != user_sr {
            for ch in ref_ch.iter_mut() {
                *ch = audio::resample(ch, ref_sr, user_sr);
            }
        }
        for ch in ref_ch.iter_mut() {
            ch.truncate(min_len);
            ch.resize(min_len, 0.0);
        }

        let user_ref = add_channels(&processed_channels, &ref_ch);
        let mut user_ref_limited = user_ref;
        for ch in user_ref_limited.iter_mut() {
            limiter = Limiter::new(user_sr as f64, -1.0);
            limiter.process_mono(ch);
        }
        let with_ref_path = with_suffix(output_path, "_with_ref");
        audio::write_wav_stereo(&with_ref_path, &user_ref_limited, user_sr)
            .map_err(|e| Error::Audio(e.to_string()))?;
        eprintln!("   With ref -> {}", with_ref_path.display());

        // -6dB (~0.5 linear) so reference sits behind the user vocal
        let ref_quiet: Vec<Vec<f64>> = ref_ch
            .iter()
            .map(|ch| ch.iter().map(|&s| s * 0.5).collect())
            .collect();
        let guide = add_channels(&add_channels(&processed_channels, &inst_channels), &ref_quiet);
        let mut guide_limited = guide;
        for ch in guide_limited.iter_mut() {
            limiter = Limiter::new(user_sr as f64, -1.0);
            limiter.process_mono(ch);
        }
        let guide_path = with_suffix(output_path, "_plus_ref");
        audio::write_wav_stereo(&guide_path, &guide_limited, user_sr)
            .map_err(|e| Error::Audio(e.to_string()))?;
        eprintln!("   Guide mix -> {}", guide_path.display());
    }

    eprintln!("Done!");
    Ok(())
}

// Demucs float outputs can spike beyond +/-1.0
fn clamp_channels(channels: &mut [Vec<f64>]) {
    for ch in channels.iter_mut() {
        for s in ch.iter_mut() {
            *s = s.clamp(-1.0, 1.0);
        }
    }
}

fn clamp_mono(samples: &mut Vec<f64>) {
    for s in samples.iter_mut() {
        *s = s.clamp(-1.0, 1.0);
    }
}

fn analyze_reference(ref_path: &Path, inst_path: &Path) -> Result<ReferenceAnalysis> {
    let (mut ref_mono, ref_sr) =
        audio::read_wav(ref_path).map_err(|e| Error::Audio(e.to_string()))?;
    let (mut inst_mono, _inst_sr) =
        audio::read_wav(inst_path).map_err(|e| Error::Audio(e.to_string()))?;
    clamp_mono(&mut ref_mono);
    clamp_mono(&mut inst_mono);

    let ref_db = audio::rms_db(&ref_mono);
    let inst_db = audio::rms_db(&inst_mono);
    let target_balance = ref_db - inst_db;

    let ref_brightness = audio::spectral_centroid(&ref_mono, ref_sr);

    eprintln!(
        "   Ref loudness: {:.1}dB, Inst: {:.1}dB, Balance: {:+.1}dB",
        ref_db, inst_db, target_balance
    );
    eprintln!("   Ref brightness: {:.0} Hz", ref_brightness);

    Ok(ReferenceAnalysis {
        target_balance_db: target_balance,
        target_brightness: ref_brightness,
    })
}

fn apply_emotion_alignment(
    user_channels: &[Vec<f64>],
    ref_channels: &[Vec<f64>],
    _sr: i32,
    strength: f64,
) -> Vec<Vec<f64>> {
    let strength = strength.clamp(0.0, 1.0);
    let hop_length = 1024usize;

    let user_mono = audio::to_mono(user_channels);
    let ref_mono = audio::to_mono(ref_channels);

    let min_len = user_mono.len().min(ref_mono.len());
    if min_len < hop_length * 2 {
        return user_channels.to_vec();
    }

    let ref_rms = audio::rms_frames(&ref_mono[..min_len], 2048, hop_length);
    let user_rms = audio::rms_frames(&user_mono[..min_len], 2048, hop_length);

    let min_frames = ref_rms.len().min(user_rms.len());
    if min_frames < 2 {
        return user_channels.to_vec();
    }

    let ref_rms = audio::smooth(&ref_rms[..min_frames], 5);
    let user_rms = audio::smooth(&user_rms[..min_frames], 5);

    // Clamp to +/-9dB to avoid pumping artifacts on quiet passages
    let gain_db: Vec<f64> = ref_rms
        .iter()
        .zip(user_rms.iter())
        .map(|(&r, &u)| {
            let r_db = 20.0 * (r + 1e-6).log10();
            let u_db = 20.0 * (u + 1e-6).log10();
            ((r_db - u_db) * strength).clamp(-9.0, 9.0)
        })
        .collect();

    let gain_linear: Vec<f64> = gain_db.iter().map(|&db| 10.0_f64.powf(db / 20.0)).collect();

    let n_samples = user_channels[0].len();
    let frame_positions: Vec<f64> = (0..min_frames).map(|i| (i * hop_length) as f64).collect();

    let sample_gain = interpolate_linear(&frame_positions, &gain_linear, n_samples);

    eprintln!(
        "   Emotion: gain range {:.1} ~ {:.1} dB",
        gain_db.iter().cloned().fold(f64::INFINITY, f64::min),
        gain_db.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    user_channels
        .iter()
        .map(|ch| {
            ch.iter()
                .enumerate()
                .map(|(i, &s)| s * sample_gain[i.min(sample_gain.len() - 1)])
                .collect()
        })
        .collect()
}

fn apply_smart_chain(
    channels: &mut [Vec<f64>],
    sr: i32,
    target_brightness: f64,
    user_brightness: f64,
    vocal_rms_db: f64,
) {
    let sr_f = sr as f64;

    let brightness_diff = target_brightness - user_brightness;
    let high_shelf_gain = (brightness_diff / 500.0).clamp(-4.0, 6.0);
    let comp_threshold = (-40.0_f64).max(vocal_rms_db - 8.0);
    let gate_threshold = (-60.0_f64).max(vocal_rms_db - 35.0);

    eprintln!("   Auto EQ: HighShelf {:+.1}dB", high_shelf_gain);

    for ch in channels.iter_mut() {
        let mut gate = Gate::new(sr_f, gate_threshold, 4.0, 200.0);
        gate.process_mono(ch);

        let mut hpf = Biquad::highpass(sr_f, 80.0);
        hpf.process_mono(ch);

        if high_shelf_gain.abs() > 0.1 {
            let mut shelf = Biquad::high_shelf(sr_f, 8000.0, high_shelf_gain);
            shelf.process_mono(ch);
        }

        // Cut 400Hz mud that builds up from close-mic recording
        let mut peak = Biquad::peak_eq(sr_f, 400.0, -2.5, 1.0);
        peak.process_mono(ch);

        let mut comp = Compressor::new(sr_f, comp_threshold, 2.5, 10.0, 100.0);
        comp.process_mono(ch);

        let mut reverb = Reverb::new(sr_f, 0.5, 0.2, 0.8);
        reverb.process_mono(ch);
    }
}

fn add_channels(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_ch = a.len().min(b.len());
    (0..n_ch)
        .map(|ch| {
            let n = a[ch].len().min(b[ch].len());
            (0..n).map(|i| a[ch][i] + b[ch][i]).collect()
        })
        .collect()
}

fn interpolate_linear(positions: &[f64], values: &[f64], n_samples: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![1.0; n_samples];
    }
    if values.len() == 1 {
        return vec![values[0]; n_samples];
    }

    let mut out = Vec::with_capacity(n_samples);
    let last_pos = positions.last().copied().unwrap_or(0.0);
    let last_val = values.last().copied().unwrap_or(1.0);
    let first_val = values[0];

    for i in 0..n_samples {
        let x = i as f64;
        if x <= positions[0] {
            out.push(first_val);
        } else if x >= last_pos {
            out.push(last_val);
        } else {
            let mut lo = 0;
            let mut hi = positions.len() - 1;
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if positions[mid] <= x {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let t = (x - positions[lo]) / (positions[hi] - positions[lo]);
            out.push(values[lo] + t * (values[hi] - values[lo]));
        }
    }
    out
}

fn with_suffix(path: &Path, suffix: &str) -> std::path::PathBuf {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy();
    let ext = path.extension().unwrap_or_default().to_string_lossy();
    let parent = path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}{}.{}", stem, suffix, ext))
}
