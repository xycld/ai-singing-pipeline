use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rustfft::{num_complex::Complex64, FftPlanner};
use std::path::Path;

/// Read a WAV file into mono f64 samples and sample rate.
pub fn read_wav(path: &Path) -> Result<(Vec<f64>, i32), Box<dyn std::error::Error>> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate as i32;
    let channels = spec.channels as usize;

    let samples_f64: Vec<f64> = match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u64 << (bits - 1)) as f64;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f64 / max_val)
                .collect()
        }
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap() as f64)
            .collect(),
    };

    let mono = if channels > 1 {
        samples_f64
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f64>() / channels as f64)
            .collect()
    } else {
        samples_f64
    };

    Ok((mono, sr))
}

/// Read a WAV file preserving channel layout: returns (channels x samples, sample_rate).
pub fn read_wav_stereo(
    path: &Path,
) -> Result<(Vec<Vec<f64>>, i32), Box<dyn std::error::Error>> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate as i32;
    let channels = spec.channels as usize;

    let interleaved: Vec<f64> = match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u64 << (bits - 1)) as f64;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f64 / max_val)
                .collect()
        }
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap() as f64)
            .collect(),
    };

    let n_samples = interleaved.len() / channels;
    let mut out = vec![vec![0.0f64; n_samples]; channels];
    for (i, chunk) in interleaved.chunks(channels).enumerate() {
        for (ch, &val) in chunk.iter().enumerate() {
            out[ch][i] = val;
        }
    }

    Ok((out, sr))
}

/// Write mono f64 samples as 16-bit WAV.
pub fn write_wav(
    path: &Path,
    samples: &[f64],
    sr: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: sr as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val = (clamped * 32767.0) as i16;
        writer.write_sample(val)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Write multi-channel f64 samples as 16-bit WAV.
pub fn write_wav_stereo(
    path: &Path,
    channels: &[Vec<f64>],
    sr: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_ch = channels.len() as u16;
    let spec = WavSpec {
        channels: n_ch,
        sample_rate: sr as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let n_samples = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    let mut writer = WavWriter::create(path, spec)?;
    for i in 0..n_samples {
        for ch in channels {
            let clamped = ch[i].clamp(-1.0, 1.0);
            writer.write_sample((clamped * 32767.0) as i16)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

/// Average multiple channels down to mono.
pub fn to_mono(channels: &[Vec<f64>]) -> Vec<f64> {
    if channels.is_empty() {
        return vec![];
    }
    if channels.len() == 1 {
        return channels[0].clone();
    }
    let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    let n_ch = channels.len() as f64;
    (0..n)
        .map(|i| channels.iter().map(|c| c[i]).sum::<f64>() / n_ch)
        .collect()
}

/// Duplicate mono to stereo.
pub fn mono_to_stereo(mono: &[f64]) -> Vec<Vec<f64>> {
    vec![mono.to_vec(), mono.to_vec()]
}

/// Ensure at least 2 channels, duplicating mono if needed.
pub fn ensure_stereo(channels: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    if channels.len() >= 2 {
        channels
    } else if channels.len() == 1 {
        vec![channels[0].clone(), channels[0].clone()]
    } else {
        vec![vec![], vec![]]
    }
}

/// Resample mono audio between sample rates using sinc interpolation.
pub fn resample(samples: &[f64], from_sr: i32, to_sr: i32) -> Vec<f64> {
    if from_sr == to_sr || samples.is_empty() {
        return samples.to_vec();
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    let ratio = to_sr as f64 / from_sr as f64;
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk_size = 1024;
    let mut resampler =
        SincFixedIn::<f64>::new(ratio, 2.0, params, chunk_size, 1).expect("resampler init");

    let mut output = Vec::with_capacity((samples.len() as f64 * ratio * 1.1) as usize);
    let mut pos = 0;

    while pos < samples.len() {
        let end = (pos + chunk_size).min(samples.len());
        let mut chunk = samples[pos..end].to_vec();
        // SincFixedIn requires exactly chunk_size input samples per call
        if chunk.len() < chunk_size {
            chunk.resize(chunk_size, 0.0);
        }
        let input = vec![chunk];
        if let Ok(result) = resampler.process(&input, None) {
            output.extend_from_slice(&result[0]);
        }
        pos += chunk_size;
    }

    let expected = (samples.len() as f64 * ratio) as usize;
    output.truncate(expected);
    output
}

fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / size as f64).cos())
        })
        .collect()
}

/// Short-time Fourier Transform, returning `window_size / 2 + 1` complex bins per frame.
pub fn stft(samples: &[f64], window_size: usize, hop_size: usize) -> Vec<Vec<Complex64>> {
    let window = hann_window(window_size);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(window_size);
    let n_bins = window_size / 2 + 1;

    let mut frames = Vec::new();
    // Start at -window_size/2 so each frame centers on frame_index * hop_size (matches librosa center=True)
    let mut pos: isize = -(window_size as isize / 2);

    while pos < samples.len() as isize {
        let mut buf: Vec<Complex64> = (0..window_size)
            .map(|j| {
                let idx = pos + j as isize;
                let val = if idx >= 0 && (idx as usize) < samples.len() {
                    samples[idx as usize] * window[j]
                } else {
                    0.0
                };
                Complex64::new(val, 0.0)
            })
            .collect();

        fft.process(&mut buf);
        frames.push(buf[..n_bins].to_vec());
        pos += hop_size as isize;
    }

    frames
}

/// Inverse STFT with overlap-add synthesis.
pub fn istft(frames: &[Vec<Complex64>], window_size: usize, hop_size: usize) -> Vec<f64> {
    if frames.is_empty() {
        return vec![];
    }

    let window = hann_window(window_size);
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(window_size);

    let output_len = (frames.len() - 1) * hop_size + window_size;
    let mut output = vec![0.0f64; output_len];
    let mut window_sum = vec![0.0f64; output_len];

    for (i, frame) in frames.iter().enumerate() {
        // Reconstruct full spectrum by mirroring conjugate bins
        let mut buf: Vec<Complex64> = Vec::with_capacity(window_size);
        buf.extend_from_slice(frame);
        for j in (1..window_size / 2).rev() {
            buf.push(frame[j].conj());
        }
        while buf.len() < window_size {
            buf.push(Complex64::new(0.0, 0.0));
        }

        ifft.process(&mut buf);

        let offset = i * hop_size;
        let scale = 1.0 / window_size as f64;
        for j in 0..window_size {
            if offset + j < output_len {
                output[offset + j] += buf[j].re * scale * window[j];
                window_sum[offset + j] += window[j] * window[j];
            }
        }
    }

    for i in 0..output_len {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }

    output
}

/// Root mean square of a signal.
pub fn rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&s| s * s).sum::<f64>() / samples.len() as f64).sqrt()
}

/// RMS in decibels.
pub fn rms_db(samples: &[f64]) -> f64 {
    20.0 * (rms(samples) + 1e-9).log10()
}

/// Peak amplitude.
pub fn peak(samples: &[f64]) -> f64 {
    samples.iter().fold(0.0f64, |m, &s| m.max(s.abs()))
}

/// Frame-wise RMS values.
pub fn rms_frames(samples: &[f64], frame_len: usize, hop_len: usize) -> Vec<f64> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < samples.len() {
        let end = (pos + frame_len).min(samples.len());
        out.push(rms(&samples[pos..end]));
        pos += hop_len;
    }
    out
}

/// Spectral centroid in Hz (brightness measure).
pub fn spectral_centroid(samples: &[f64], sr: i32) -> f64 {
    let window_size = 2048;
    let hop_size = 1024;
    let frames = stft(samples, window_size, hop_size);
    if frames.is_empty() {
        return 0.0;
    }

    let bin_hz = sr as f64 / window_size as f64;

    let mut total_centroid = 0.0;
    let mut n = 0;

    for frame in &frames {
        let magnitudes: Vec<f64> = frame.iter().map(|c| c.norm()).collect();
        let mag_sum: f64 = magnitudes.iter().sum();
        if mag_sum > 1e-10 {
            let weighted: f64 = magnitudes
                .iter()
                .enumerate()
                .map(|(k, &m)| k as f64 * bin_hz * m)
                .sum();
            total_centroid += weighted / mag_sum;
            n += 1;
        }
    }

    if n > 0 {
        total_centroid / n as f64
    } else {
        0.0
    }
}

/// Simple moving-average smoothing.
pub fn smooth(values: &[f64], window: usize) -> Vec<f64> {
    if window <= 1 || values.is_empty() {
        return values.to_vec();
    }
    let w = window.min(values.len());
    let mut out = vec![0.0; values.len()];

    for i in 0..values.len() {
        let half = w / 2;
        let lo = i.saturating_sub(half);
        let hi = (i + half).min(values.len() - 1);
        let sum: f64 = values[lo..=hi].iter().sum();
        let count = hi - lo + 1;
        out[i] = sum / count as f64;
    }
    out
}

/// Onset strength via spectral flux (half-wave rectified magnitude differences).
pub fn onset_strength(samples: &[f64], _sr: i32, hop_size: usize) -> Vec<f64> {
    let window_size = 2048;
    let frames = stft(samples, window_size, hop_size);
    if frames.len() < 2 {
        return vec![0.0; frames.len()];
    }

    let mut strengths = vec![0.0];
    for i in 1..frames.len() {
        let flux: f64 = frames[i]
            .iter()
            .zip(frames[i - 1].iter())
            .map(|(cur, prev)| (cur.norm() - prev.norm()).max(0.0))
            .sum();
        strengths.push(flux);
    }
    strengths
}
