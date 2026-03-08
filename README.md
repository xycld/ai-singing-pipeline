# ai-singing-pipeline

[English](README.en.md)

AI 歌声后处理全流程 Rust 库 —— 降噪、对齐、音高修正、混音，一站式完成。

## 功能

- **降噪** — 基于 ONNX 推理的人声降噪（可选特性 `denoise`）
- **对齐** — 使用 DTW + 信封拉伸将录音对齐到伴奏时间线
- **音高修正** — WORLD 声码器分析 + TD-PSOLA 逐帧修正基频
- **混音** — 将修正后的人声与伴奏混合输出

## 安装

```toml
[dependencies]
ai-singing-pipeline = "0.1"
```

启用降噪（需要 ONNX Runtime）：

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

## 许可证

[Apache-2.0](LICENSE)
