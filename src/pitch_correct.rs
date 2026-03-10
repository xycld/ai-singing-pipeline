use crate::{Error, Result};
use std::path::{Path, PathBuf};

/// Embedded Praat script — compiled into the binary so it works regardless of CWD.
const EMBEDDED_PRAAT_SCRIPT: &str = include_str!("../scripts/pitch_correct.praat");

/// Configuration for the pitch correction stage.
#[derive(Debug, Clone)]
pub struct PitchCorrectConfig {
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
        self.process_praat(user_wav.as_ref(), ref_wav.as_ref(), out_wav.as_ref())
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

    /// Resolve the Praat script path, falling back to writing the embedded script
    /// to a temp file if the configured path can't be found.
    fn resolve_praat_script(&self, base: Option<&Path>) -> Result<PathBuf> {
        if let Some(found) = Self::resolve_path(&self.config.praat_script, base) {
            return Ok(found);
        }

        // Script not found on disk — write embedded copy to temp.
        let tmp = std::env::temp_dir().join("_embedded_pitch_correct.praat");
        std::fs::write(&tmp, EMBEDDED_PRAAT_SCRIPT)?;
        Ok(tmp)
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

        let praat_script = self.resolve_praat_script(base)?;

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
