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

### `video_merger_multi.py` — Multi-Speaker Conference Video Merger

Like `video_merger.py` but accepts multiple speaker videos. The **slides** video is the master timeline; each speaker video is synchronized individually. Portions of the slides with no speaker coverage are shown as black.

**What it does:**

- Places slides (75%) and speaker (25%) side by side in a 1920x1080 output
- **Slides is the master** — output duration equals slides duration
- Accepts one or more speaker videos, each independently synchronized to slides
- Black panel where no speaker video covers a slides segment
- Auto-detects sync offset for each speaker via audio cross-correlation (or use `--offsets`)
- Handles both sync directions: speaker starts before or after slides recording begins
- Overlapping speaker segments are resolved by clipping the earlier one
- All layout, crop, background, and intro options from `video_merger.py` are supported

**Usage:**

```bash
# Single speaker, auto-detect sync
python3 video_merger_multi.py slides.mp4 spk1.mp4 -o output.mp4

# Multiple speakers, auto-detect sync for each
python3 video_merger_multi.py slides.mp4 spk1.mp4 spk2.mp4 -o output.mp4

# Manual offsets (seconds into slides where each speaker starts)
python3 video_merger_multi.py slides.mp4 spk1.mp4 spk2.mp4 -o output.mp4 \
    --offsets 0 3600

# Full example
python3 video_merger_multi.py slides.mp4 spk1.mp4 spk2.mp4 -o output.mp4 \
    --background background.jpg \
    --slides-autocrop \
    --intro-community logo_ruby.png \
    --intro-sponsor   logo_sponsor.png \
    --intro-music     music.mp3
```

**Offset convention (`--offsets`):**

| Value | Meaning |
|---|---|
| `300` | Speaker starts 5 minutes into the slides timeline |
| `-60` | Speaker has a 60s preamble before slides recording began |
| _(omitted)_ | Auto-detect via cross-correlation |

**Options:**

| Option | Default | Description |
|---|---|---|
| `-o / --output` | required | Output file path |
| `--offsets` | auto | Signed seconds per speaker (see above); provide fewer than N speakers to auto-detect the rest |
| `--slides-pct` | `0.72` | Fraction of slides width that contains slides |
| `--width` | `1920` | Output width in pixels |
| `--height` | `1080` | Output height in pixels |
| `--background` | — | Background image behind both panels |
| `--slides-autocrop` | off | Auto-detect and remove black borders from slides |
| `--slides-crop-top/bottom/left/right` | `0` | Manual crop pixels |
| `--intro-community` | — | Community logo PNG |
| `--intro-sponsor` | — | Sponsor logo PNG |
| `--intro-music` | — | Intro music MP3 |
| `--intro-community-duration` | `2` | Community screen duration (s) |
| `--intro-sponsor-duration` | `2` | Sponsor screen duration (s) |

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
