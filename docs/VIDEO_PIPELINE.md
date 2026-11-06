# Video Processing Pipeline

## 1. Principles of Real-World Video Ingestion

Video footage from modern devices varies widely in format, resolution, encoding, and framerate. The video ingestion subsystem in `iris.video` enforces the following principles:

1. **Format Validation:** Ingests MP4 (H.264/H.265/AV1), MOV, AVI, and WebM containers via OpenCV and FFmpeg probes.
2. **Streaming Frame Extraction:** Frames are yielded lazily on-demand or extracted into temporary memory-mapped arrays in chunks. Video streams are **never** fully decompressed into system RAM at once.
3. **Adaptive Sampling:** Ingesting 60 or 120 FPS high-speed video for skeletal tracking introduces redundant computation. The pipeline resamples frames to a configurable canonical processing rate (e.g. 15 or 30 FPS) while recording exact fractional timestamps.
4. **Resolution Scaling:** Frames exceeding a bounding dimension (e.g. 1280px or 720px on CPU profile) are downscaled with bilinear interpolation to preserve aspect ratio and avoid CPU memory exhaustion.
5. **Corrupt Frame Recovery:** Individual dropped or corrupted video frames do not cause pipeline crashes. The reader flags dropped frames and marks them for temporal motion imputation.

## 2. CLI Tooling

To inspect any video file from the command line:

```bash
python main.py inspect-video path/to/input.mp4
```

Output includes:
- Container format & codec
- Dimensions $(W \times H)$
- Frame rate & total frame count
- Estimated duration
- Hardware compatibility status & recommended sample step
