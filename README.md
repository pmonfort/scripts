# Multipurpose Scripts

A collection of standalone utility scripts for various tasks.

---

## Scripts

### `video_merger.py` — Conference Video Merger

Merges two conference recordings — slides and speaker — into a single synchronized side-by-side video, with optional background image and intro sequence.

**What it does:**

- Places slides (75%) and speaker (25%) side by side in a 1920x1080 output
- Trims the slides width to exclude participant thumbnails on the right (`--slides-pct`)
- Auto-syncs slides to the speaker using audio cross-correlation (trims sponsor intros automatically)
- Uses the speaker video as the master timeline — output starts and ends with the speaker
- Handles both sync directions: trims slides start when slides has a preamble, or pads slides start with a frozen first frame when the speaker started recording before the slides
- Freezes the last slide frame if slides end before the speaker finishes
- Optionally overlays both panels on a background image with a visible margin border
- Optionally auto-detects and removes black borders from the slides (`--slides-autocrop`)
- Optionally prepends a branded intro sequence (community/sponsor logos + music)

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

# Background image + auto-crop black borders
python3 video_merger.py slides.mp4 speaker.mp4 output.mp4 \
    --background background.jpg \
    --slides-autocrop

# Full example with intro sequence
python3 video_merger.py slides.mp4 speaker.mp4 output.mp4 \
    --background background.jpg \
    --slides-autocrop \
    --intro-community logo_ruby.png \
    --intro-sponsor   logo_sponsor.png \
    --intro-music     music.mp3
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--offset` | auto | Signed seconds to align slides to speaker: positive trims slides start (slides has a preamble), negative pads slides start with a frozen first frame (speaker started before slides recording began) |
| `--slides-pct` | `0.72` | Fraction of slides width that contains slides (excludes thumbnails) |
| `--width` | `1920` | Output video width in pixels |
| `--height` | `1080` | Output video height in pixels |
| `--background` | — | Background image shown behind slides and speaker panels |
| `--slides-autocrop` | off | Auto-detect and remove black borders from slides |
| `--slides-crop-top` | `0` | Pixels to crop from the top of slides |
| `--slides-crop-bottom` | `0` | Pixels to crop from the bottom of slides |
| `--slides-crop-left` | `0` | Pixels to crop from the left of slides |
| `--slides-crop-right` | `0` | Pixels to crop from the right of slides |
| `--intro-community` | — | Community logo PNG shown first in intro |
| `--intro-sponsor` | — | Sponsor logo PNG shown second in intro |
| `--intro-music` | — | Background music MP3 for the intro |
| `--intro-community-duration` | `2` | Seconds to show the community screen |
| `--intro-sponsor-duration` | `2` | Seconds to show the sponsor screen |

---

### `add_intro.py` — Video Intro Prepender

Prepends a branded intro sequence (community logo, sponsor logo, optional music) to any video, with smooth fade transitions.

**What it does:**

- Generates logo screens (white background, centered logo) for community and/or sponsor
- Concatenates them into an intro sequence with optional background music
- Joins the intro to the main video with a configurable fade transition
- Preserves the input video resolution by default

**Requirements:**

```bash
brew install ffmpeg
```

**Usage:**

```bash
# Community + sponsor logos with music
python3 add_intro.py input.mp4 output.mp4 \
    --intro-community logo_ruby.png \
    --intro-sponsor   logo_sponsor.png \
    --intro-music     music.mp3

# Community logo only, custom durations
python3 add_intro.py input.mp4 output.mp4 \
    --intro-community logo_ruby.png \
    --intro-community-duration 3 \
    --transition 0.5
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--intro-community` | — | Path to community logo PNG |
| `--intro-sponsor` | — | Path to sponsor logo PNG |
| `--intro-music` | — | Path to intro music MP3 |
| `--intro-community-duration` | `2` | Seconds to show the community screen |
| `--intro-sponsor-duration` | `2` | Seconds to show the sponsor screen |
| `--transition` | `1` | Fade duration at the join (seconds) |
| `--width` | from input | Output video width in pixels |
| `--height` | from input | Output video height in pixels |
