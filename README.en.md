# ai-singing-pipeline

[中文](README.md)

End-to-end Rust library for AI singing vocal post-processing — align, pitch correct, and mix in one shot.

## Features

- **Align** — DTW + chroma features to align a recording to a reference vocal timeline
- **Pitch Correction** — Praat PSOLA, corrects pitch toward a reference vocal (script embedded in binary)
- **Mix** — auto loudness matching, emotion envelope alignment, high-shelf EQ, blended with instrumental
- **Denoise** — ONNX-based vocal noise reduction (optional `denoise` feature)

## Prerequisites

- [Praat](https://www.fon.hum.uva.nl/praat/) — required for the pitch correction stage

## Build

```bash
cargo build --release
```

## CLI Usage

### Full Pipeline

```bash
ai-singing-pipeline run \
  recording.wav \      # user recording
  reference.wav \      # reference vocal
  instrumental.wav \   # backing track
  output.wav \         # output file
  --praat-bin /path/to/praat
```

### Align Only

```bash
ai-singing-pipeline align recording.wav reference.wav aligned.wav
```

### Pitch Correct Only

```bash
ai-singing-pipeline pitch-correct \
  aligned.wav reference.wav corrected.wav \
  --praat-bin /path/to/praat \
  --strength 0.6 \     # correction strength 0.0-1.0
  --tolerance 50       # tolerance in cents
```

### Mix Only

```bash
ai-singing-pipeline mix vocal.wav instrumental.wav reference.wav output.wav
```

## Library Usage

```toml
[dependencies]
ai-singing-pipeline = "0.1"
```

```rust
use ai_singing_pipeline::{Pipeline, PipelineConfig, PitchCorrectConfig};

let config = PipelineConfig {
    pitch_correct: PitchCorrectConfig {
        praat_bin: "/path/to/praat".into(),
        strength: 0.6,
        tolerance_cents: 50.0,
        ..Default::default()
    },
    ..Default::default()
};

Pipeline::new(config)
    .run("recording.wav", "reference.wav", "instrumental.wav", "output.wav")?;
```

## License

[Unlicense](LICENSE)
