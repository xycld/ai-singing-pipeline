# ai-singing-pipeline

[中文](README.md)

End-to-end Rust library for AI singing vocal post-processing — denoise, align, pitch correct, and mix in one shot.

## Features

- **Denoise** — ONNX-based vocal noise reduction (optional `denoise` feature)
- **Align** — DTW + envelope stretching to align a recording to backing-track timing
- **Pitch Correction** — WORLD vocoder analysis + TD-PSOLA per-frame F0 correction
- **Mix** — blend corrected vocals with the backing track

## Installation

```toml
[dependencies]
ai-singing-pipeline = "0.1"
```

Enable denoising (requires ONNX Runtime):

```toml
[dependencies]
ai-singing-pipeline = { version = "0.1", features = ["denoise"] }
```

## CLI

```bash
cargo install ai-singing-pipeline

ai-singing-pipeline \
  --vocal vocal.wav \
  --backing backing.wav \
  --reference reference.wav \
  --output output.wav
```

## License

[Apache-2.0](LICENSE)
