# ai-singing-pipeline

[English](README.en.md)

AI 歌声后处理全流程 Rust 库 —— 对齐、音高修正、混音，一站式完成。

## 功能

- **对齐** — DTW + 色度特征，将录音对齐到参考人声时间线
- **音高修正** — Praat PSOLA，按参考人声修正音高（脚本已嵌入二进制）
- **混音** — 自动响度匹配、情感包络对齐、高架 EQ，与伴奏混合输出
- **降噪** — 基于 ONNX 推理的人声降噪（可选特性 `denoise`）

## 前置依赖

- [Praat](https://www.fon.hum.uva.nl/praat/) — 音高修正阶段需要 Praat 二进制

## 编译

```bash
cargo build --release
```

## CLI 使用

### 全流程

```bash
ai-singing-pipeline run \
  录音.wav \           # 用户录音
  参考人声.wav \       # 原唱人声
  伴奏.wav \           # 伴奏轨
  输出.wav \           # 输出文件
  --praat-bin /path/to/praat
```

### 单独对齐

```bash
ai-singing-pipeline align 录音.wav 参考人声.wav 对齐输出.wav
```

### 单独修音

```bash
ai-singing-pipeline pitch-correct \
  对齐后.wav 参考人声.wav 修音输出.wav \
  --praat-bin /path/to/praat \
  --strength 0.6 \     # 修正强度 0.0-1.0
  --tolerance 50       # 容差（音分）
```

### 单独混音

```bash
ai-singing-pipeline mix 人声.wav 伴奏.wav 参考人声.wav 输出.wav
```

## 作为库使用

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
    .run("录音.wav", "参考人声.wav", "伴奏.wav", "输出.wav")?;
```

## 许可证

[Unlicense](LICENSE)
