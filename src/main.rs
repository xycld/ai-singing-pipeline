#[cfg(feature = "cli")]
mod cli {
    use clap::{Parser, Subcommand};
    use std::path::PathBuf;

    use ai_singing_pipeline::{
        AlignConfig, DenoiseConfig, Method, MixConfig, Pipeline, PipelineConfig,
        PitchCorrectConfig, PitchCorrector,
    };

    #[derive(Clone, clap::ValueEnum)]
    enum CliMethod {
        Praat,
        World,
        Signalsmith,
        Tdpsola,
    }

    #[derive(Parser)]
    #[command(name = "ai-singing-pipeline")]
    #[command(about = "AI singing vocal post-processing pipeline")]
    struct Cli {
        #[command(subcommand)]
        command: Commands,
    }

    #[derive(Subcommand)]
    enum Commands {
        /// Run the full pipeline: denoise -> align -> pitch correct -> mix
        Run {
            /// Raw user vocal recording
            user_wav: PathBuf,
            /// Reference vocal (clean original)
            ref_wav: PathBuf,
            /// Instrumental / backing track
            inst_wav: PathBuf,
            /// Output file path
            out_wav: PathBuf,

            /// Sample rate for processing
            #[arg(long, default_value_t = 44100)]
            sr: i32,

            /// Path to ONNX model for denoising (omit to skip denoise)
            #[arg(long)]
            denoise_model: Option<PathBuf>,

            /// Pitch correction method
            #[arg(long, value_enum, default_value_t = CliMethod::Praat)]
            method: CliMethod,

            /// Pitch correction strength 0.0-1.0
            #[arg(long, default_value_t = 0.6)]
            strength: f64,

            /// Pitch tolerance in cents
            #[arg(long, default_value_t = 50.0)]
            tolerance: f64,

            /// Path to Praat binary
            #[arg(long, default_value = "praat")]
            praat_bin: PathBuf,

            /// Path to Praat pitch correction script
            #[arg(long, default_value = "scripts/pitch_correct.praat")]
            praat_script: PathBuf,

            /// Disable emotion alignment in mixing
            #[arg(long)]
            disable_emotion: bool,

            /// Emotion alignment strength 0.0-1.0
            #[arg(long, default_value_t = 1.0)]
            emotion_strength: f64,
        },

        /// Denoise a vocal recording (requires --features denoise)
        Denoise {
            input: PathBuf,
            output: PathBuf,
            /// Path to ONNX model
            #[arg(long, default_value = "models/bs_roformer.onnx")]
            model: PathBuf,
        },

        /// Time-align user vocal to reference
        Align {
            user_wav: PathBuf,
            ref_wav: PathBuf,
            out_wav: PathBuf,
            /// Output sample rate
            #[arg(long, default_value_t = 44100)]
            sr: i32,
        },

        /// Pitch-correct user vocal using reference
        PitchCorrect {
            user_wav: PathBuf,
            ref_wav: PathBuf,
            out_wav: PathBuf,
            /// Correction strength 0.0-1.0
            #[arg(long, default_value_t = 0.6)]
            strength: f64,
            /// Tolerance in cents
            #[arg(long, default_value_t = 50.0)]
            tolerance: f64,
            /// Synthesis method
            #[arg(long, value_enum, default_value_t = CliMethod::Praat)]
            method: CliMethod,
            /// Path to Praat binary
            #[arg(long, default_value = "praat")]
            praat_bin: PathBuf,
            /// Path to Praat script
            #[arg(long, default_value = "scripts/pitch_correct.praat")]
            praat_script: PathBuf,
        },

        /// Mix processed vocal with instrumental using reference-guided parameters
        Mix {
            user_wav: PathBuf,
            inst_wav: PathBuf,
            ref_wav: PathBuf,
            out_wav: PathBuf,
            /// Emotion alignment strength (0=off, 1=full)
            #[arg(long, default_value_t = 1.0)]
            emotion_strength: f64,
            /// Disable emotion alignment
            #[arg(long)]
            disable_emotion: bool,
        },
    }

    fn to_method(m: &CliMethod) -> Method {
        match m {
            CliMethod::Praat => Method::Praat,
            CliMethod::World => Method::World,
            CliMethod::Signalsmith => Method::Signalsmith,
            CliMethod::Tdpsola => Method::Tdpsola,
        }
    }

    pub fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let cli = Cli::parse();

        match cli.command {
            Commands::Run {
                user_wav,
                ref_wav,
                inst_wav,
                out_wav,
                sr,
                denoise_model,
                method,
                strength,
                tolerance,
                praat_bin,
                praat_script,
                disable_emotion,
                emotion_strength,
            } => {
                let config = PipelineConfig {
                    sample_rate: sr,
                    denoise: denoise_model.map(|p| DenoiseConfig { model_path: p }),
                    align: AlignConfig::default(),
                    pitch_correct: PitchCorrectConfig {
                        method: to_method(&method),
                        strength,
                        tolerance_cents: tolerance,
                        praat_bin,
                        praat_script,
                        ..Default::default()
                    },
                    mix: MixConfig {
                        enable_emotion: !disable_emotion,
                        emotion_strength,
                    },
                };

                let pipeline = Pipeline::new(config).on_progress(|stage, pct| {
                    eprintln!("[{stage}] {:.0}%", pct * 100.0);
                });

                pipeline.run(&user_wav, &ref_wav, &inst_wav, &out_wav)?;
            }

            Commands::Denoise {
                input,
                output,
                model,
            } => {
                let config = PipelineConfig {
                    denoise: Some(DenoiseConfig { model_path: model }),
                    ..Default::default()
                };
                Pipeline::new(config).denoise(&input, &output)?;
            }

            Commands::Align {
                user_wav,
                ref_wav,
                out_wav,
                sr,
            } => {
                let config = PipelineConfig {
                    sample_rate: sr,
                    ..Default::default()
                };
                Pipeline::new(config).align(&user_wav, &ref_wav, &out_wav)?;
            }

            Commands::PitchCorrect {
                user_wav,
                ref_wav,
                out_wav,
                strength,
                tolerance,
                method,
                praat_bin,
                praat_script,
            } => {
                let method = to_method(&method);
                let method_name = match method {
                    Method::Praat => "Praat PSOLA",
                    Method::World => "WORLD vocoder",
                    Method::Signalsmith => "Signalsmith Stretch",
                    Method::Tdpsola => "TD-PSOLA",
                };
                eprintln!(
                    "Correcting pitch [{}] (strength={:.0}%, tolerance={:.0} cents)...",
                    method_name,
                    strength * 100.0,
                    tolerance,
                );

                let corrector = PitchCorrector::new()
                    .method(method)
                    .strength(strength)
                    .tolerance_cents(tolerance)
                    .praat_bin(praat_bin)
                    .praat_script(praat_script);

                let result = corrector.process(&user_wav, &ref_wav, &out_wav)?;

                eprintln!("\nStats:");
                eprintln!("  Frames:    {}", result.n_frames);
                eprintln!("  Voiced:    {}", result.n_voiced);
                eprintln!(
                    "  Corrected: {} ({:.0}%)",
                    result.n_corrected,
                    result.n_corrected as f64 / result.n_voiced.max(1) as f64 * 100.0,
                );
                eprintln!("  Skipped (within tolerance): {}", result.n_skipped_close);
                eprintln!("  Skipped (unvoiced): {}", result.n_skipped_unvoiced);
                if result.median_correction_cents > 0.0 {
                    eprintln!(
                        "  Median correction: {:.0} cents",
                        result.median_correction_cents,
                    );
                }
                eprintln!("\nDone! -> {}", out_wav.display());
            }

            Commands::Mix {
                user_wav,
                inst_wav,
                ref_wav,
                out_wav,
                emotion_strength,
                disable_emotion,
            } => {
                let config = PipelineConfig {
                    mix: MixConfig {
                        enable_emotion: !disable_emotion,
                        emotion_strength,
                    },
                    ..Default::default()
                };
                Pipeline::new(config).mix(&user_wav, &inst_wav, &ref_wav, &out_wav)?;
            }
        }

        Ok(())
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "cli")]
    {
        cli::run()
    }
    #[cfg(not(feature = "cli"))]
    {
        eprintln!("CLI not available — build with `cargo build --features cli`");
        std::process::exit(1);
    }
}
