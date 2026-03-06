# Multipurpose Scripts

A collection of standalone utility scripts for various tasks.

---

## Scripts

### `video_merger.py` — Conference Video Merger

Merges two conference recordings — slides and speaker — into a single synchronized side-by-side video.

**What it does:**

- Places slides (75%) and speaker (25%) side by side in a 1920x1080 output
- Crops the slides video to remove any participant column on the right
- Auto-syncs slides to the speaker using audio cross-correlation (trims sponsor intros automatically)
- Uses the speaker video as the master timeline — output starts and ends with the speaker
- Freezes the last slide frame if slides end before the speaker finishes

**Requirements:**

```bash
brew install ffmpeg
pip3 install numpy scipy
```

**Usage:**

```bash
# Auto-detect sync offset
python3 video_merger.py slides.mp4 speaker.mp4 output.mp4

# Manual offset (seconds to skip from slides start)
python3 video_merger.py slides.mp4 speaker.mp4 output.mp4 --offset 312

# Custom slides width percentage (default: 0.72)
python3 video_merger.py slides.mp4 speaker.mp4 output.mp4 --slides-pct 0.74

# Custom output resolution
python3 video_merger.py slides.mp4 speaker.mp4 output.mp4 --width 1920 --height 1080
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--offset` | auto | Seconds to trim from the slides start |
| `--slides-pct` | `0.72` | Fraction of slides width to keep (crops right side) |
| `--width` | `1920` | Output video width in pixels |
| `--height` | `1080` | Output video height in pixels |
