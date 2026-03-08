//! AI singing vocal post-processing pipeline.
//!
//! Four-stage pipeline: **denoise** -> **align** -> **pitch correct** -> **mix**.
//!
//! Each stage can be used independently or chained via [`Pipeline`].
//!
//! # Quick start
//!
//! ```no_run
//! use ai_singing_pipeline::{Pipeline, PipelineConfig};
//!
//! let pipeline = Pipeline::new(PipelineConfig::default());
//! pipeline.run("raw.wav", "ref_vocals.wav", "instrumental.wav", "output.wav").unwrap();
//! ```

pub mod align;
pub mod audio;
pub mod denoise;
pub mod dsp;
pub mod mix;
pub mod pitch_correct;
pub mod world;
pub(crate) mod world_sys;

use std::path::{Path, PathBuf};

// Re-exports for convenience
pub use align::AlignConfig;
pub use denoise::DenoiseConfig;
pub use mix::MixConfig;
pub use pitch_correct::{CorrectionResult, Method, PitchCorrectConfig, PitchCorrector};

/// Pipeline stages for progress reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Denoise,
    Align,
    PitchCorrect,
    Mix,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stage::Denoise => write!(f, "denoise"),
            Stage::Align => write!(f, "align"),
            Stage::PitchCorrect => write!(f, "pitch-correct"),
            Stage::Mix => write!(f, "mix"),
        }
    }
}

/// Unified error type for the pipeline.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("audio: {0}")]
    Audio(String),

    #[error("sample rate mismatch: user={user} ref={reference}")]
    SampleRateMismatch { user: i32, reference: i32 },

    #[error("pitch correction: {0}")]
    PitchCorrect(String),

    #[error("praat binary not found: {0}")]
    PraatNotFound(PathBuf),

    #[error("praat script not found: {0}")]
    PraatScriptNotFound(PathBuf),

    #[error("alignment: {0}")]
    Alignment(String),

    #[error("denoise: {0}")]
    Denoise(String),

    #[error("mixing: {0}")]
    Mixing(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Full pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub sample_rate: i32,
    pub denoise: Option<DenoiseConfig>,
    pub align: AlignConfig,
    pub pitch_correct: PitchCorrectConfig,
    pub mix: MixConfig,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            denoise: None,
            align: AlignConfig::default(),
            pitch_correct: PitchCorrectConfig::default(),
            mix: MixConfig::default(),
        }
    }
}

/// Pipeline orchestrator.
///
/// Chains denoise -> align -> pitch correct -> mix, with optional
/// progress callbacks and individual stage access.
pub struct Pipeline {
    config: PipelineConfig,
    progress: Option<Box<dyn Fn(Stage, f32) + Send + Sync>>,
}

impl Pipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            progress: None,
        }
    }

    /// Set a progress callback: `Fn(stage, progress_0_to_1)`.
    pub fn on_progress(mut self, cb: impl Fn(Stage, f32) + Send + Sync + 'static) -> Self {
        self.progress = Some(Box::new(cb));
        self
    }

    fn report(&self, stage: Stage, progress: f32) {
        if let Some(cb) = &self.progress {
            cb(stage, progress);
        }
    }

    /// Run the full pipeline: denoise -> align -> pitch correct -> mix.
    ///
    /// - `user_wav`: raw user vocal recording
    /// - `ref_wav`: reference vocal (clean, time-aligned original)
    /// - `inst_wav`: instrumental / backing track
    /// - `output_wav`: final mixed output path
    pub fn run(
        &self,
        user_wav: impl AsRef<Path>,
        ref_wav: impl AsRef<Path>,
        inst_wav: impl AsRef<Path>,
        output_wav: impl AsRef<Path>,
    ) -> Result<()> {
        let user_wav = user_wav.as_ref();
        let ref_wav = ref_wav.as_ref();
        let inst_wav = inst_wav.as_ref();
        let output_wav = output_wav.as_ref();

        let tmp_dir = std::env::temp_dir();

        // Stage 1: Denoise (optional)
        let denoised_path;
        if let Some(denoise_config) = &self.config.denoise {
            self.report(Stage::Denoise, 0.0);
            denoised_path = tmp_dir.join("_pipeline_denoised.wav");
            denoise::process_denoise(user_wav, &denoised_path, denoise_config)?;
            self.report(Stage::Denoise, 1.0);
        } else {
            denoised_path = user_wav.to_path_buf();
        }

        // Stage 2: Align
        self.report(Stage::Align, 0.0);
        let aligned_path = tmp_dir.join("_pipeline_aligned.wav");
        align::process_align(
            &denoised_path,
            ref_wav,
            &aligned_path,
            self.config.sample_rate,
            &self.config.align,
        )?;
        self.report(Stage::Align, 1.0);

        // Stage 3: Pitch correct
        self.report(Stage::PitchCorrect, 0.0);
        let corrected_path = tmp_dir.join("_pipeline_corrected.wav");
        let corrector = PitchCorrector::from_config(self.config.pitch_correct.clone());
        let _stats = corrector.process(&aligned_path, ref_wav, &corrected_path)?;
        self.report(Stage::PitchCorrect, 1.0);

        // Stage 4: Mix
        self.report(Stage::Mix, 0.0);
        mix::process_mix(
            &corrected_path,
            inst_wav,
            ref_wav,
            output_wav,
            &self.config.mix,
        )?;
        self.report(Stage::Mix, 1.0);

        // Cleanup temp files
        let _ = std::fs::remove_file(&tmp_dir.join("_pipeline_denoised.wav"));
        let _ = std::fs::remove_file(&aligned_path);
        let _ = std::fs::remove_file(&corrected_path);

        Ok(())
    }

    // --- Individual stage methods ---

    /// Run only the denoise stage.
    pub fn denoise(&self, input: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<()> {
        let config = self
            .config
            .denoise
            .as_ref()
            .cloned()
            .unwrap_or_default();
        denoise::process_denoise(input.as_ref(), output.as_ref(), &config)
    }

    /// Run only the alignment stage.
    pub fn align(
        &self,
        user_wav: impl AsRef<Path>,
        ref_wav: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<()> {
        align::process_align(
            user_wav.as_ref(),
            ref_wav.as_ref(),
            output.as_ref(),
            self.config.sample_rate,
            &self.config.align,
        )
    }

    /// Run only the pitch correction stage.
    pub fn pitch_correct(
        &self,
        user_wav: impl AsRef<Path>,
        ref_wav: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<CorrectionResult> {
        let corrector = PitchCorrector::from_config(self.config.pitch_correct.clone());
        corrector.process(user_wav, ref_wav, output)
    }

    /// Run only the mixing stage.
    pub fn mix(
        &self,
        user_wav: impl AsRef<Path>,
        inst_wav: impl AsRef<Path>,
        ref_wav: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<()> {
        mix::process_mix(
            user_wav.as_ref(),
            inst_wav.as_ref(),
            ref_wav.as_ref(),
            output.as_ref(),
            &self.config.mix,
        )
    }
}
