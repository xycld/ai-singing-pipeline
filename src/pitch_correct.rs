//! Reference-guided pitch correction.
//!
//! Compares a user vocal against a reference melody and corrects pitch using
//! one of several backends: Praat PSOLA, WORLD vocoder, Signalsmith Stretch, or TD-PSOLA.

use crate::{audio, world, Error, Result};
use std::path::{Path, PathBuf};

/// Pitch correction backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// Praat PSOLA via CLI subprocess — best quality, no GPL infection.
    Praat,
    /// WORLD vocoder — full spectral decomposition/resynthesis.
    World,
    /// Signalsmith Stretch — spectral phase-vocoder pitch shifting.
    Signalsmith,
    /// TD-PSOLA with variable wavelength tracking.
    Tdpsola,
}

/// Configuration for the pitch correction stage.
#[derive(Debug, Clone)]
pub struct PitchCorrectConfig {
    pub method: Method,
    pub strength: f64,
    pub tolerance_cents: f64,
    pub f0_min: f64,
    pub f0_max: f64,
    pub praat_bin: PathBuf,
    pub praat_script: PathBuf,
}

impl Default for PitchCorrectConfig {
    fn default() -> Self {
        Self {
            method: Method::Praat,
            strength: 0.6,
            tolerance_cents: 50.0,
            f0_min: 80.0,
            f0_max: 600.0,
            praat_bin: PathBuf::from("praat"),
            praat_script: PathBuf::from("scripts/pitch_correct.praat"),
        }
    }
}

/// Builder for pitch correction.
#[derive(Debug, Clone)]
pub struct PitchCorrector {
    config: PitchCorrectConfig,
}

impl Default for PitchCorrector {
    fn default() -> Self {
        Self {
            config: PitchCorrectConfig::default(),
        }
    }
}

impl PitchCorrector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_config(config: PitchCorrectConfig) -> Self {
        Self { config }
    }

    pub fn method(mut self, method: Method) -> Self {
        self.config.method = method;
        self
    }

    pub fn strength(mut self, strength: f64) -> Self {
        self.config.strength = strength.clamp(0.0, 1.0);
        self
    }

    pub fn tolerance_cents(mut self, cents: f64) -> Self {
        self.config.tolerance_cents = cents;
        self
    }

    pub fn f0_min(mut self, hz: f64) -> Self {
        self.config.f0_min = hz;
        self
    }

    pub fn f0_max(mut self, hz: f64) -> Self {
        self.config.f0_max = hz;
        self
    }

    pub fn praat_bin(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.praat_bin = path.into();
        self
    }

    pub fn praat_script(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.praat_script = path.into();
        self
    }

    /// Run pitch correction: read WAVs, correct, write output.
    pub fn process(
        &self,
        user_wav: impl AsRef<Path>,
        ref_wav: impl AsRef<Path>,
        out_wav: impl AsRef<Path>,
    ) -> Result<CorrectionResult> {
        let user_wav = user_wav.as_ref();
        let ref_wav = ref_wav.as_ref();
        let out_wav = out_wav.as_ref();

        if self.config.method == Method::Praat {
            return self.process_praat(user_wav, ref_wav, out_wav);
        }

        let (user_audio, sr_user) =
            audio::read_wav(user_wav).map_err(|e| Error::Audio(e.to_string()))?;
        let (ref_audio, sr_ref) =
            audio::read_wav(ref_wav).map_err(|e| Error::Audio(e.to_string()))?;

        if sr_user != sr_ref {
            return Err(Error::SampleRateMismatch {
                user: sr_user,
                reference: sr_ref,
            });
        }

        let (output, stats) = self.process_audio(&user_audio, &ref_audio, sr_user)?;
        audio::write_wav(out_wav, &output, sr_user).map_err(|e| Error::Audio(e.to_string()))?;
        Ok(stats)
    }

    /// In-memory pitch correction (non-Praat backends only).
    pub fn process_audio(
        &self,
        user_audio: &[f64],
        ref_audio: &[f64],
        sample_rate: i32,
    ) -> Result<(Vec<f64>, CorrectionResult)> {
        let corrections = analyze_corrections(
            user_audio,
            ref_audio,
            sample_rate,
            self.config.strength,
            self.config.tolerance_cents,
        );

        let output = match self.config.method {
            Method::World => synthesize_world(user_audio, &corrections, sample_rate),
            Method::Signalsmith => synthesize_signalsmith(user_audio, &corrections, sample_rate),
            Method::Tdpsola => synthesize_tdpsola(user_audio, &corrections, sample_rate),
            Method::Praat => {
                return Err(Error::PitchCorrect(
                    "Praat backend requires file paths — use process() instead".into(),
                ));
            }
        };

        let stats = &corrections.stats;
        Ok((
            output,
            CorrectionResult {
                n_frames: stats.n_frames,
                n_voiced: stats.n_voiced,
                n_corrected: stats.n_corrected,
                n_skipped_close: stats.n_skipped_close,
                n_skipped_unvoiced: stats.n_skipped_unvoiced,
                median_correction_cents: stats.median_correction_cents,
            },
        ))
    }

    fn resolve_path(path: &Path, base: Option<&Path>) -> Option<PathBuf> {
        if path.exists() {
            return Some(path.to_path_buf());
        }
        if let Some(base) = base {
            let candidate = base.join(path);
            if candidate.exists() {
                return Some(candidate);
            }
        }
        None
    }

    fn process_praat(
        &self,
        user_wav: &Path,
        ref_wav: &Path,
        out_wav: &Path,
    ) -> Result<CorrectionResult> {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()));
        let base = exe_dir.as_deref();

        let praat_bin = Self::resolve_path(&self.config.praat_bin, base)
            .ok_or_else(|| Error::PraatNotFound(self.config.praat_bin.clone()))?;

        let praat_script = Self::resolve_path(&self.config.praat_script, base)
            .ok_or_else(|| Error::PraatScriptNotFound(self.config.praat_script.clone()))?;

        let user_wav = std::fs::canonicalize(user_wav)?;
        let ref_wav = std::fs::canonicalize(ref_wav)?;
        let out_wav = if out_wav.exists() {
            std::fs::canonicalize(out_wav)?
        } else {
            let parent = out_wav.parent().unwrap_or(Path::new("."));
            std::fs::canonicalize(parent)?.join(out_wav.file_name().unwrap())
        };

        let output = std::process::Command::new(&praat_bin)
            .arg("--run")
            .arg(&praat_script)
            .arg(&user_wav)
            .arg(&ref_wav)
            .arg(&out_wav)
            .arg(self.config.strength.to_string())
            .arg(self.config.tolerance_cents.to_string())
            .arg(self.config.f0_min.to_string())
            .arg(self.config.f0_max.to_string())
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(Error::PitchCorrect(format!(
                "praat exit {}: {}{}",
                output.status,
                stderr.trim(),
                stdout.trim()
            )));
        }

        let info = String::from_utf8_lossy(&output.stdout);
        Ok(Self::parse_praat_stats(&info))
    }

    fn parse_praat_stats(info: &str) -> CorrectionResult {
        let mut result = CorrectionResult::default();
        for line in info.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("Frames:") {
                result.n_frames = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("Voiced:") {
                result.n_voiced = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("Corrected:") {
                if let Some(n) = rest.trim().split_whitespace().next() {
                    result.n_corrected = n.parse().unwrap_or(0);
                }
            } else if let Some(rest) = line.strip_prefix("Skipped (close):") {
                result.n_skipped_close = rest.trim().parse().unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("Skipped (unvoiced):") {
                result.n_skipped_unvoiced = rest.trim().parse().unwrap_or(0);
            }
        }
        result
    }
}

/// Statistics returned after pitch correction.
#[derive(Debug, Clone, Default)]
pub struct CorrectionResult {
    pub n_frames: usize,
    pub n_voiced: usize,
    pub n_corrected: usize,
    pub n_skipped_close: usize,
    pub n_skipped_unvoiced: usize,
    pub median_correction_cents: f64,
}

// ---------------------------------------------------------------------------
// F0 analysis + per-method synthesis (from correct.rs)
// ---------------------------------------------------------------------------

pub struct FrameCorrections {
    pub f0_user: Vec<f64>,
    pub f0_corrected: Vec<f64>,
    pub shift_st: Vec<f64>,
    pub times: Vec<f64>,
    pub frame_period: f64,
    pub stats: CorrectionStats,
}

pub struct CorrectionStats {
    pub n_frames: usize,
    pub n_voiced: usize,
    pub n_corrected: usize,
    pub n_skipped_close: usize,
    pub n_skipped_unvoiced: usize,
    pub median_correction_cents: f64,
}

pub fn analyze_corrections(
    user_audio: &[f64],
    ref_audio: &[f64],
    sr: i32,
    strength: f64,
    tolerance_cents: f64,
) -> FrameCorrections {
    let frame_period = 5.0;

    eprintln!("  Harvest (user)...");
    let (times_u, f0_u) = world::harvest(user_audio, sr, frame_period);
    eprintln!("  Harvest (ref)...");
    let (_times_r, f0_r) = world::harvest(ref_audio, sr, frame_period);

    let n_frames = f0_u.len().min(f0_r.len());
    let tolerance_st = tolerance_cents / 100.0;

    let mut f0_corrected = f0_u.clone();
    let mut shift_st = vec![0.0f64; f0_u.len()];
    let mut n_voiced = 0usize;
    let mut n_corrected = 0usize;
    let mut n_skipped_close = 0usize;
    let mut n_skipped_unvoiced = 0usize;
    let mut corrections_cents = Vec::new();

    for i in 0..n_frames {
        let fu = f0_u[i];
        let fr = f0_r[i];

        if fu <= 0.0 || fr <= 0.0 {
            n_skipped_unvoiced += 1;
            continue;
        }

        n_voiced += 1;

        let st_u = hz_to_semitone(fu);
        let mut st_r = hz_to_semitone(fr);

        let diff = st_r - st_u;
        if diff.abs() > 6.0 {
            st_r -= 12.0 * (diff / 12.0).round();
        }

        let diff = st_r - st_u;
        let diff_cents = diff.abs() * 100.0;

        if diff.abs() < tolerance_st {
            n_skipped_close += 1;
            continue;
        }

        let correction = strength * diff;
        let st_corrected = st_u + correction;
        let f0_new = semitone_to_hz(st_corrected).clamp(80.0, 600.0);
        f0_corrected[i] = f0_new;
        shift_st[i] = correction;

        n_corrected += 1;
        corrections_cents.push(diff_cents);
    }

    smooth_f0_voiced(&mut f0_corrected, 11);

    for i in 0..f0_u.len() {
        if f0_u[i] > 0.0 && f0_corrected[i] > 0.0 {
            shift_st[i] = hz_to_semitone(f0_corrected[i]) - hz_to_semitone(f0_u[i]);
        } else {
            shift_st[i] = 0.0;
        }
    }

    let median_correction_cents = if corrections_cents.is_empty() {
        0.0
    } else {
        corrections_cents.sort_by(|a, b| a.partial_cmp(b).unwrap());
        corrections_cents[corrections_cents.len() / 2]
    };

    FrameCorrections {
        f0_user: f0_u,
        f0_corrected,
        shift_st,
        times: times_u,
        frame_period,
        stats: CorrectionStats {
            n_frames,
            n_voiced,
            n_corrected,
            n_skipped_close,
            n_skipped_unvoiced,
            median_correction_cents,
        },
    }
}

pub fn synthesize_world(
    user_audio: &[f64],
    corrections: &FrameCorrections,
    sr: i32,
) -> Vec<f64> {
    eprintln!("  CheapTrick...");
    let (spectrogram, fft_size) =
        world::cheaptrick(user_audio, sr, &corrections.times, &corrections.f0_user);

    eprintln!("  D4C...");
    let aperiodicity =
        world::d4c(user_audio, sr, &corrections.times, &corrections.f0_user, fft_size);

    eprintln!("  Synthesis (WORLD)...");
    let mut output = world::synthesis(
        &corrections.f0_corrected,
        &spectrogram,
        &aperiodicity,
        fft_size,
        corrections.frame_period,
        sr,
    );

    normalize_to_input_peak(user_audio, &mut output);
    output
}

pub fn synthesize_signalsmith(
    user_audio: &[f64],
    corrections: &FrameCorrections,
    sr: i32,
) -> Vec<f64> {
    use signalsmith_stretch::Stretch;

    eprintln!("  Synthesis (Signalsmith Stretch)...");

    let mut stretch = Stretch::preset_default(1, sr as u32);

    let input_f32: Vec<f32> = user_audio.iter().map(|&s| s as f32).collect();
    let total_samples = input_f32.len();

    let block_size = 4096usize;
    let frame_hop = (corrections.frame_period * sr as f64 / 1000.0) as usize;

    let mut raw_output = Vec::with_capacity(total_samples + block_size);
    let n_blocks = (total_samples + block_size - 1) / block_size;

    for i in 0..n_blocks {
        let in_start = i * block_size;
        let in_end = (in_start + block_size).min(total_samples);
        let block_len = in_end - in_start;

        let frame_start =
            (in_start / frame_hop).min(corrections.shift_st.len().saturating_sub(1));
        let frame_end =
            ((in_end + frame_hop - 1) / frame_hop).min(corrections.shift_st.len());
        let avg_shift = if frame_start < frame_end {
            corrections.shift_st[frame_start..frame_end]
                .iter()
                .sum::<f64>()
                / (frame_end - frame_start) as f64
        } else {
            0.0
        };

        stretch.set_transpose_factor_semitones(avg_shift as f32, Some(0.5));

        let mut out_block = vec![0.0f32; block_len];
        stretch.process(&input_f32[in_start..in_end], &mut out_block);
        raw_output.extend_from_slice(&out_block);
    }

    raw_output.truncate(total_samples);
    let mut output: Vec<f64> = raw_output.iter().map(|&s| s as f64).collect();
    normalize_to_input_peak(user_audio, &mut output);
    output
}

struct VariableHann {
    s: f32,
    c: f32,
    s_delta: f32,
    c_delta: f32,
    s_square: f32,
    c_square: f32,
}

impl VariableHann {
    fn new(wavelength: f32) -> Self {
        let delta = 2.0 * std::f32::consts::PI / (wavelength * 4.0);
        Self {
            s: 0.0,
            c: 1.0,
            s_square: 0.0,
            c_square: 1.0,
            s_delta: delta.sin(),
            c_delta: delta.cos(),
        }
    }

    fn set_wavelength(&mut self, wavelength: f32) {
        let delta = 2.0 * std::f32::consts::PI / (wavelength * 4.0);
        self.s_delta = delta.sin();
        self.c_delta = delta.cos();
    }
}

impl tdpsola::AlternatingWindow for VariableHann {
    fn step(&mut self) {
        let s = self.s;
        self.s = self.c * self.s_delta + self.s * self.c_delta;
        self.c = self.c * self.c_delta - s * self.s_delta;
        self.s_square = self.s * self.s;
        self.c_square = self.c * self.c;
        let factor = (3.0 - self.s_square - self.c_square) / 2.0;
        self.s *= factor;
        self.c *= factor;
    }

    fn window_a(&self) -> f32 {
        self.c_square
    }

    fn window_b(&self) -> f32 {
        self.s_square
    }

    fn a_is_ending(&self) -> bool {
        self.s.is_sign_positive() == self.c.is_sign_positive()
    }
}

pub fn synthesize_tdpsola(
    user_audio: &[f64],
    corrections: &FrameCorrections,
    sr: i32,
) -> Vec<f64> {
    use tdpsola::{Speed, TdpsolaAnalysis, TdpsolaSynthesis};

    eprintln!("  Synthesis (TD-PSOLA)...");

    let hop_samples = (corrections.frame_period * sr as f64 / 1000.0) as usize;

    let initial_f0 = corrections
        .f0_user
        .iter()
        .find(|&&f| f > 0.0)
        .copied()
        .unwrap_or(200.0);
    let initial_wl = sr as f32 / initial_f0 as f32;

    let mut window = VariableHann::new(initial_wl);
    let mut analysis = TdpsolaAnalysis::new(&window);

    let padding_len = initial_wl as usize + 1;
    for _ in 0..padding_len {
        analysis.push_sample(0.0, &mut window);
    }

    for (i, &sample) in user_audio.iter().enumerate() {
        let frame_idx =
            (i / hop_samples).min(corrections.f0_user.len().saturating_sub(1));
        let source_f0 = if corrections.f0_user[frame_idx] > 0.0 {
            corrections.f0_user[frame_idx]
        } else {
            initial_f0 as f64
        };
        window.set_wavelength(sr as f32 / source_f0 as f32);
        analysis.push_sample(sample as f32, &mut window);
    }

    let target_f0_init = if corrections.f0_corrected[0] > 0.0 {
        corrections.f0_corrected[0]
    } else {
        initial_f0 as f64
    };
    let mut synthesis = TdpsolaSynthesis::new(
        Speed::from_f32(1.0),
        sr as f32 / target_f0_init as f32,
    );

    for _ in 0..padding_len {
        let _ = synthesis.try_get_sample(&analysis);
        synthesis.step(&analysis);
    }

    let mut output = Vec::with_capacity(user_audio.len());
    let mut current_frame = usize::MAX;

    for sample_idx in 0..user_audio.len() {
        let frame_idx = (sample_idx / hop_samples)
            .min(corrections.f0_corrected.len().saturating_sub(1));
        if frame_idx != current_frame {
            current_frame = frame_idx;
            let target_f0 = if corrections.f0_corrected[frame_idx] > 0.0 {
                corrections.f0_corrected[frame_idx]
            } else if corrections.f0_user[frame_idx] > 0.0 {
                corrections.f0_user[frame_idx]
            } else {
                initial_f0 as f64
            };
            synthesis.set_wavelength(sr as f32 / target_f0 as f32);
        }

        match synthesis.try_get_sample(&analysis) {
            Ok(sample) => {
                output.push(sample as f64);
                synthesis.step(&analysis);
            }
            Err(_) => break,
        }
    }

    output.resize(user_audio.len(), 0.0);
    normalize_to_input_peak(user_audio, &mut output);
    output
}

fn normalize_to_input_peak(input: &[f64], output: &mut [f64]) {
    let input_peak = input.iter().fold(0.0f64, |m, &s| m.max(s.abs()));
    let output_peak = output.iter().fold(0.0f64, |m, &s| m.max(s.abs()));
    if output_peak > 1e-6 {
        let scale = input_peak / output_peak;
        for s in output.iter_mut() {
            *s *= scale;
        }
        eprintln!(
            "  Normalized: output peak {:.4} -> {:.4} (scale {:.4})",
            output_peak, input_peak, scale
        );
    }
}

pub fn hz_to_semitone(f: f64) -> f64 {
    12.0 * (f / 440.0).log2()
}

pub fn semitone_to_hz(s: f64) -> f64 {
    440.0 * 2.0_f64.powf(s / 12.0)
}

fn smooth_f0_voiced(f0: &mut [f64], radius: usize) {
    let n = f0.len();
    if n == 0 {
        return;
    }
    let orig = f0.to_vec();
    let mut window = Vec::with_capacity(radius * 2 + 1);

    for i in 0..n {
        if orig[i] <= 0.0 {
            continue;
        }
        window.clear();
        let lo = i.saturating_sub(radius);
        let hi = (i + radius).min(n - 1);
        for j in lo..=hi {
            if orig[j] > 0.0 {
                window.push(orig[j]);
            }
        }
        if !window.is_empty() {
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            f0[i] = window[window.len() / 2];
        }
    }
}
