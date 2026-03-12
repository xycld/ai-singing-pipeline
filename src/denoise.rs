use crate::{Error, Result};
use std::path::Path;

/// Configuration for the denoising stage.
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    pub model_path: std::path::PathBuf,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("models/bs_roformer.onnx"),
        }
    }
}

/// Run vocal separation / denoising via ONNX inference.
pub fn process_denoise(
    input_path: &Path,
    output_path: &Path,
    config: &DenoiseConfig,
) -> Result<()> {
    #[cfg(feature = "denoise")]
    {
        denoise_onnx(input_path, output_path, config)
    }

    #[cfg(not(feature = "denoise"))]
    {
        let _ = (input_path, output_path, config);
        Err(Error::Denoise(
            "denoise feature not enabled — rebuild with `--features denoise`".into(),
        ))
    }
}

#[cfg(feature = "denoise")]
fn denoise_onnx(
    input_path: &Path,
    output_path: &Path,
    config: &DenoiseConfig,
) -> Result<()> {
    use ort::session::Session;

    if !config.model_path.exists() {
        return Err(Error::Denoise(format!(
            "ONNX model not found: {} — run scripts/export_bs_roformer.py first",
            config.model_path.display()
        )));
    }

    eprintln!("Loading ONNX model: {}...", config.model_path.display());
    let mut session = Session::builder()
        .map_err(|e| Error::Denoise(format!("ort session builder: {}", e)))?
        .commit_from_file(&config.model_path)
        .map_err(|e| Error::Denoise(format!("ort load model: {}", e)))?;

    let (samples, sr) = crate::audio::read_wav(input_path).map_err(|e| Error::Audio(e.to_string()))?;

    let window_size = 2048;
    let hop_size = 512;

    eprintln!("Computing STFT...");
    let stft_frames = crate::audio::stft(&samples, window_size, hop_size);
    let n_frames = stft_frames.len();
    let n_bins = window_size / 2 + 1;

    // Model expects [batch=1, channels=2, freq_bins, time_frames]
    // Channel 0 = real, Channel 1 = imaginary
    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    for (t, frame) in stft_frames.iter().enumerate() {
        for (f, c) in frame.iter().enumerate() {
            real[f * n_frames + t] = c.re as f32;
            imag[f * n_frames + t] = c.im as f32;
        }
    }

    let mut input_data = Vec::with_capacity(2 * n_bins * n_frames);
    input_data.extend_from_slice(&real);
    input_data.extend_from_slice(&imag);

    eprintln!("Running inference ({} frames)...", n_frames);

    let input_tensor = ort::value::Value::from_array(
        (vec![2i64, n_bins as i64, n_frames as i64], input_data),
    )
    .map_err(|e| Error::Denoise(format!("input tensor: {}", e)))?;

    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| Error::Denoise(format!("inference: {}", e)))?;

    let output_tensor = outputs
        .values()
        .next()
        .ok_or_else(|| Error::Denoise("no output tensor".into()))?;

    let mask_view = output_tensor
        .try_extract_tensor::<f32>()
        .map_err(|e| Error::Denoise(format!("extract output: {}", e)))?;
    let mask_data = mask_view.1;

    eprintln!("Applying mask and reconstructing...");
    let mut masked_frames = stft_frames.clone();

    let mask_has_two_channels = mask_data.len() >= 2 * n_bins * n_frames;

    for (t, frame) in masked_frames.iter_mut().enumerate() {
        for (f, c) in frame.iter_mut().enumerate() {
            if mask_has_two_channels {
                let mask_re = mask_data[f * n_frames + t] as f64;
                let mask_im = mask_data[n_bins * n_frames + f * n_frames + t] as f64;
                c.re *= mask_re;
                c.im *= mask_im;
            } else {
                let mask_val = if f * n_frames + t < mask_data.len() {
                    mask_data[f * n_frames + t] as f64
                } else {
                    1.0
                };
                c.re *= mask_val;
                c.im *= mask_val;
            }
        }
    }

    let vocals = crate::audio::istft(&masked_frames, window_size, hop_size);

    let vocals: Vec<f64> = vocals.into_iter().take(samples.len()).collect();

    crate::audio::write_wav(output_path, &vocals, sr).map_err(|e| Error::Audio(e.to_string()))?;
    eprintln!("Denoising done! -> {}", output_path.display());

    Ok(())
}
