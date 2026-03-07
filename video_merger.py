#!/usr/bin/env python3
"""
video_merger_full.py — Merge slides + speaker into a synchronized side-by-side conference video
                        with optional black-border crop, background image, and intro sequence.

Usage:
    python3 video_merger_full.py slides.mp4 speaker.mp4 final.mp4
    python3 video_merger_full.py slides.mp4 speaker.mp4 final.mp4 --offset 312
    python3 video_merger_full.py slides.mp4 speaker.mp4 final.mp4 \\
        --slides-autocrop \\
        --background background.jpg \\
        --intro-community logo_ruby.png \\
        --intro-sponsor   logo_sponsor.png \\
        --intro-music     music.mp3

Requirements:
    brew install ffmpeg
    pip3 install numpy scipy
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, filtfilt


# ── Constants ──────────────────────────────────────────────────────────────────

SLIDES_COLUMN_FRACTION = 0.75   # Fraction of output width reserved for slides
BACKGROUND_MARGIN_PX   = 30     # Pixels of background visible as border around each panel
LOGO_WIDTH_FRACTION    = 0.40   # Logo width relative to output width in intro clips
LOGO_FADE_DURATION_S   = 0.4    # Fade-in / fade-out duration for logo clips
INTRO_FPS              = 30     # Frame rate for intro clips
MUSIC_FADE_OUT_S       = 0.5    # Music fade-out duration before end of intro


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CropRegion:
    """A rectangular crop region in pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    def constrain_width(self, max_w: int) -> CropRegion:
        """Return a copy with width capped so the right edge does not exceed max_w."""
        return CropRegion(self.x, self.y, min(self.w, max_w - self.x), self.h)


# ── FFmpeg helpers ─────────────────────────────────────────────────────────────

def ffmpeg(*args: str | Path, show_progress: bool = False) -> None:
    """Run an FFmpeg command, raising RuntimeError on non-zero exit.

    When show_progress=True, FFmpeg's stderr is passed through to the terminal
    so the user sees encoding progress. Otherwise stderr is captured for error
    reporting only.
    """
    cmd = ["ffmpeg", "-y", *[str(a) for a in args]]
    result = subprocess.run(
        cmd,
        stderr=None if show_progress else subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"FFmpeg failed:\n{stderr[-3000:]}")


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def probe_dimensions(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def probe_fps(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    n, d = result.stdout.strip().split("/")
    return float(n) / float(d)


# ── Sync ───────────────────────────────────────────────────────────────────────

def find_sync_offset(speaker_path: Path, slides_path: Path, max_search_s: int = 700) -> float:
    """Return seconds to skip from the slides start to align audio with the speaker.

    Extracts 60 s of speech audio from the speaker and up to max_search_s + 60 s
    from the slides, applies a 300–3400 Hz bandpass filter to isolate voice, then
    uses FFT-based cross-correlation to find the best alignment point.
    """
    print("  Detecting sync via cross-correlation...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp     = Path(tmp_dir)
        spk_wav = tmp / "spk.wav"
        sld_wav = tmp / "sld.wav"

        ffmpeg("-i", speaker_path, "-ac", "1", "-ar", "16000", "-vn", "-t", "60", spk_wav)
        ffmpeg("-i", slides_path, "-ac", "1", "-ar", "16000", "-vn",
               "-t", str(max_search_s + 60), sld_wav)

        rate, spk_data = wavfile.read(spk_wav)
        _,    sld_data = wavfile.read(sld_wav)

        def bandpass_voice(signal: np.ndarray) -> np.ndarray:
            mono = signal.mean(axis=1) if signal.ndim > 1 else signal
            mono = mono.astype(np.float32)
            b, a = butter(4, [300 / (rate / 2), 3400 / (rate / 2)], btype="band")
            return filtfilt(b, a, mono)

        sld_filtered = bandpass_voice(sld_data)[:int(max_search_s * rate) + len(spk_data)]
        corr   = correlate(sld_filtered, bandpass_voice(spk_data), mode="valid")
        offset = int(np.argmax(corr)) / rate

    print(f"  Slides offset: {offset:.2f}s")
    return offset


# ── Slides crop detection ──────────────────────────────────────────────────────

def detect_crop(slides_path: Path, offset: float = 0.0, sample_s: int = 60) -> CropRegion | None:
    """Auto-detect black borders in the slides video using FFmpeg cropdetect.

    Samples `sample_s` seconds starting at `offset` and returns the median
    crop region across all frames. Using the median makes the result robust
    against outlier frames (e.g. a fully white sponsor intro would report a
    full-frame region and skew a max/min approach).

    Returns None if no crop values could be detected.
    """
    print("  Auto-detecting slides crop...")
    result = subprocess.run(
        ["ffmpeg", "-y",
         "-ss", str(offset), "-i", str(slides_path),
         "-t", str(sample_s),
         "-vf", "cropdetect=limit=32:round=2:skip=2",
         "-f", "null", "-"],
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
    )
    stderr = result.stderr.decode(errors="replace")

    crops: list[CropRegion] = []
    for m in re.finditer(r"crop=(\d+):(\d+):(\d+):(\d+)", stderr):
        w, h, x, y = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if w > 0 and h > 0:
            crops.append(CropRegion(x, y, w, h))

    if not crops:
        print("  Warning: cropdetect found nothing, skipping autocrop.")
        return None

    def median(values: list[int]) -> int:
        s = sorted(values)
        return s[len(s) // 2]

    region = CropRegion(
        x=median([c.x for c in crops]),
        y=median([c.y for c in crops]),
        w=median([c.w for c in crops]),
        h=median([c.h for c in crops]),
    )
    print(f"  Detected crop: {region} (from {len(crops)} frames)")
    return region


# ── Slides padding ─────────────────────────────────────────────────────────────

def pad_slides(slides_path: Path, output_path: Path, target_duration: float) -> Path:
    """Extend slides with a frozen last frame to reach target_duration.

    Returns slides_path unchanged if no padding is needed.
    """
    gap = target_duration - probe_duration(slides_path)
    if gap <= 0:
        return slides_path

    print(f"  Padding slides with {gap:.1f}s of frozen last frame...")
    tmp        = output_path.parent
    last_frame = tmp / "last_frame.png"
    frozen     = tmp / "frozen.mp4"
    concat_txt = tmp / "slides_concat.txt"

    ffmpeg("-sseof", "-1", "-i", slides_path, "-vframes", "1", "-q:v", "2", last_frame)
    ffmpeg(
        "-loop", "1", "-framerate", probe_fps(slides_path),
        "-i", last_frame,
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(gap + 2),
        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        frozen,
    )
    concat_txt.write_text(
        f"file '{slides_path.resolve()}'\n"
        f"file '{frozen.resolve()}'\n"
    )
    ffmpeg("-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", output_path)
    return output_path


# ── Intro ──────────────────────────────────────────────────────────────────────

def _make_logo_clip(
    logo_path: Path,
    duration_s: float,
    out_path: Path,
    output_w: int,
    output_h: int,
) -> None:
    """Render a single logo clip: white background, centered logo, white fade in/out."""
    logo_w   = int(output_w * LOGO_WIDTH_FRACTION)
    n_frames = int(duration_s * INTRO_FPS)
    fade_in  = f"fade=t=in:st=0:d={LOGO_FADE_DURATION_S}:color=white"
    fade_out = f"fade=t=out:st={duration_s - LOGO_FADE_DURATION_S}:d={LOGO_FADE_DURATION_S}:color=white"

    ffmpeg(
        "-f", "lavfi", "-i", f"color=white:s={output_w}x{output_h}:r={INTRO_FPS}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-filter_complex",
            f"movie='{logo_path}',scale={logo_w}:-2,loop=loop=-1:size=1:start=0[logo];"
            f"[0:v][logo]overlay=(W-w)/2:(H-h)/2,{fade_in},{fade_out}[outv]",
        "-map", "[outv]", "-map", "1:a",
        "-frames:v", str(n_frames), "-t", str(duration_s),
        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-r", str(INTRO_FPS),
        "-c:a", "aac", "-b:a", "64k",
        "-movflags", "+faststart",
        out_path,
    )


def build_intro(
    tmp: Path,
    output_w: int,
    output_h: int,
    community_logo: Path | None,
    sponsor_logo: Path | None,
    music_path: Path | None,
    community_dur: float,
    sponsor_dur: float,
) -> tuple[Path, float] | None:
    """Build the intro clip from logo images and optional music.

    Returns (path, total_duration_s) or None if no logos are provided.
    """
    if not community_logo and not sponsor_logo:
        return None

    clips: list[tuple[Path, float]] = []
    if community_logo:
        out = tmp / "intro_community.mp4"
        _make_logo_clip(community_logo, community_dur, out, output_w, output_h)
        clips.append((out, community_dur))
    if sponsor_logo:
        out = tmp / "intro_sponsor.mp4"
        _make_logo_clip(sponsor_logo, sponsor_dur, out, output_w, output_h)
        clips.append((out, sponsor_dur))

    total_dur = sum(d for _, d in clips)

    # Concatenate logo clips (re-encode to avoid black frames at joins)
    if len(clips) == 1:
        intro_video = clips[0][0]
    else:
        intro_video = tmp / "intro_video.mp4"
        n        = len(clips)
        inputs   = [arg for path, _ in clips for arg in ("-i", str(path))]
        filter_v = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
        filter_a = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[outa]"
        ffmpeg(
            *inputs,
            "-filter_complex", f"{filter_v};{filter_a}",
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k",
            "-movflags", "+faststart",
            intro_video,
        )

    if not music_path:
        return intro_video, total_dur

    # Replace silent audio with music, fading out before the end
    intro_with_music = tmp / "intro_with_music.mp4"
    ffmpeg(
        "-i", str(intro_video), "-i", str(music_path),
        "-filter_complex",
            f"[1:a]atrim=0:{total_dur},"
            f"afade=t=out:st={total_dur - MUSIC_FADE_OUT_S}:d={MUSIC_FADE_OUT_S}[music]",
        "-map", "0:v", "-map", "[music]",
        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        intro_with_music,
    )
    return intro_with_music, total_dur


# ── Merge ──────────────────────────────────────────────────────────────────────

def _build_background_filtergraph(
    crop: CropRegion,
    slides_out_w: int,
    spkr_out_w: int,
    output_w: int,
    output_h: int,
) -> str:
    """Filtergraph: background image fills frame, slides and speaker overlaid with margins."""
    m            = BACKGROUND_MARGIN_PX
    slides_max_w = slides_out_w - 2 * m
    slides_max_h = output_h     - 2 * m
    spkr_max_w   = spkr_out_w   - 2 * m
    spkr_max_h   = output_h     - 2 * m
    return (
        f"[2:v]scale={output_w}:{output_h}:force_original_aspect_ratio=increase,"
        f"crop={output_w}:{output_h},setsar=1,format=yuv420p[bg];"
        f"[0:v]crop={crop.w}:{crop.h}:{crop.x}:{crop.y},"
        f"scale={slides_max_w}:{slides_max_h}:force_original_aspect_ratio=decrease,"
        f"setsar=1,format=yuv420p[sv];"
        f"[1:v]scale={spkr_max_w}:{spkr_max_h}:force_original_aspect_ratio=decrease,"
        f"setsar=1,format=yuv420p[spv];"
        f"[bg][sv]overlay=({slides_out_w}-w)/2:(H-h)/2[bg_sv];"
        f"[bg_sv][spv]overlay={slides_out_w}+({spkr_out_w}-w)/2:(H-h)/2[outv]"
    )


def _build_plain_filtergraph(
    crop: CropRegion,
    slides_out_w: int,
    spkr_out_w: int,
    output_w: int,
    output_h: int,
) -> str:
    """Filtergraph: slides and speaker on black background, stacked horizontally."""
    return (
        f"[0:v]crop={crop.w}:{crop.h}:{crop.x}:{crop.y},"
        f"scale={slides_out_w}:{output_h}:force_original_aspect_ratio=decrease,"
        f"pad={slides_out_w}:{output_h}:(ow-iw)/2:(oh-ih)/2:black[sv];"
        f"[1:v]scale={spkr_out_w}:{output_h}:force_original_aspect_ratio=decrease,"
        f"pad={spkr_out_w}:{output_h}:(ow-iw)/2:(oh-ih)/2:black[spv];"
        f"[sv][spv]hstack=inputs=2[outv]"
    )


def merge(
    slides_path: Path,
    speaker_path: Path,
    output_path: Path,
    slides_col_w: int,
    output_w: int,
    output_h: int,
    background_path: Path | None = None,
    crop: CropRegion | None = None,
) -> None:
    """Combine slides (75%) and speaker (25%) side by side into output_path.

    If background_path is given, both panels are overlaid on the image with a
    margin so the background is visible as a frame around the content.
    If crop is given, it is applied to the slides source before scaling.
    """
    slides_out_w = (int(output_w * SLIDES_COLUMN_FRACTION) // 2) * 2
    spkr_out_w   = output_w - slides_out_w
    _, slides_h  = probe_dimensions(slides_path)

    slides_crop = (
        crop.constrain_width(slides_col_w) if crop
        else CropRegion(0, 0, slides_col_w, slides_h)
    )

    print(f"  Encoding {output_w}x{output_h} (slides {slides_out_w}px | speaker {spkr_out_w}px)")
    if crop:
        print(f"  Slides crop: {slides_crop}")

    if background_path:
        fg     = _build_background_filtergraph(slides_crop, slides_out_w, spkr_out_w, output_w, output_h)
        inputs = ["-i", slides_path, "-i", speaker_path, "-loop", "1", "-i", background_path]
    else:
        fg     = _build_plain_filtergraph(slides_crop, slides_out_w, spkr_out_w, output_w, output_h)
        inputs = ["-i", slides_path, "-i", speaker_path]

    ffmpeg(
        *inputs,
        "-filter_complex", fg,
        "-map", "[outv]", "-map", "1:a",
        "-shortest",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        output_path,
        show_progress=True,
    )


def prepend_intro(
    intro_path: Path,
    main_path: Path,
    output_path: Path,
    intro_dur: float,
    transition_s: float = 1.0,
) -> None:
    """Concatenate intro + main video with a white fade transition at the join."""
    print("  Prepending intro...")
    fade_out_st = max(0.0, intro_dur - transition_s)
    ffmpeg(
        "-i", str(intro_path), "-i", str(main_path),
        "-filter_complex",
            f"[0:v]setsar=1,fade=t=out:st={fade_out_st}:d={transition_s}:color=white[v0];"
            f"[1:v]setsar=1,fade=t=in:st=0:d={transition_s}:color=white[v1];"
            f"[0:a]afade=t=out:st={fade_out_st}:d={transition_s}[a0];"
            f"[1:a]afade=t=in:st=0:d={transition_s}[a1];"
            f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]",
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        output_path,
        show_progress=True,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video_merger_full.py",
        description="Merge slides + speaker into a synchronized side-by-side conference video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("slides",  help="Path to slides recording (MP4)")
    p.add_argument("speaker", help="Path to speaker recording (MP4)")
    p.add_argument("output",  help="Output file path (MP4)")

    sync = p.add_argument_group("sync")
    sync.add_argument("--offset", type=float, default=None, metavar="S",
                      help="Seconds to skip from slides start; omit to auto-detect")

    layout = p.add_argument_group("layout")
    layout.add_argument("--slides-pct", type=float, default=0.72, metavar="F",
                        help="Fraction of slides width that contains slides (excludes thumbnails)")
    layout.add_argument("--width",  type=int, default=1920, metavar="PX")
    layout.add_argument("--height", type=int, default=1080, metavar="PX")
    layout.add_argument("--background", default=None, metavar="IMAGE",
                        help="Background image shown behind slides and speaker")

    crop = p.add_argument_group("slides crop (mutually exclusive: autocrop or manual)")
    crop.add_argument("--slides-autocrop", action="store_true",
                      help="Auto-detect and remove black borders from slides")
    crop.add_argument("--slides-crop-top",    type=int, default=0, metavar="PX")
    crop.add_argument("--slides-crop-bottom", type=int, default=0, metavar="PX")
    crop.add_argument("--slides-crop-left",   type=int, default=0, metavar="PX")
    crop.add_argument("--slides-crop-right",  type=int, default=0, metavar="PX")

    intro = p.add_argument_group("intro")
    intro.add_argument("--intro-community", default=None, metavar="PNG",
                       help="Community logo PNG shown first")
    intro.add_argument("--intro-sponsor", default=None, metavar="PNG",
                       help="Sponsor logo PNG shown second")
    intro.add_argument("--intro-music", default=None, metavar="MP3",
                       help="Background music for the intro")
    intro.add_argument("--intro-community-duration", type=float, default=2.0, metavar="S")
    intro.add_argument("--intro-sponsor-duration",   type=float, default=2.0, metavar="S")

    return p


def _validate_paths(args: argparse.Namespace) -> None:
    for path, label in [(args.slides, "slides"), (args.speaker, "speaker")]:
        if not Path(path).exists():
            sys.exit(f"error: {label} file not found: {path}")
    for path, label in [
        (args.background,      "--background"),
        (args.intro_community, "--intro-community"),
        (args.intro_sponsor,   "--intro-sponsor"),
        (args.intro_music,     "--intro-music"),
    ]:
        if path and not Path(path).exists():
            sys.exit(f"error: {label} file not found: {path}")


def _resolve_crop(args: argparse.Namespace, slides_path: Path, offset: float) -> CropRegion | None:
    """Return a CropRegion from manual args or autocrop, or None if no crop is requested."""
    manual = (args.slides_crop_top, args.slides_crop_bottom,
              args.slides_crop_left, args.slides_crop_right)
    if any(manual):
        top, bottom, left, right = manual
        sw, sh = probe_dimensions(slides_path)
        return CropRegion(x=left, y=top, w=sw - left - right, h=sh - top - bottom)
    if args.slides_autocrop:
        return detect_crop(slides_path, offset=offset)
    return None


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    _validate_paths(args)

    slides_path  = Path(args.slides)
    speaker_path = Path(args.speaker)
    output_path  = Path(args.output)

    speaker_dur  = probe_duration(speaker_path)
    slides_w, _  = probe_dimensions(slides_path)
    slides_col_w = (int(slides_w * args.slides_pct) // 2) * 2
    has_intro    = bool(args.intro_community or args.intro_sponsor)

    print("\n[VideoMerger]")
    print(f"  Speaker : {speaker_path.name}  ({int(speaker_dur // 60)}m{int(speaker_dur % 60)}s)")
    print(f"  Slides  : {slides_path.name}")
    print(f"  Output  : {output_path.name}")
    if has_intro:
        parts: list[str] = []
        total_intro = 0.0
        if args.intro_community:
            parts.append(f"community {args.intro_community_duration}s")
            total_intro += args.intro_community_duration
        if args.intro_sponsor:
            parts.append(f"sponsor {args.intro_sponsor_duration}s")
            total_intro += args.intro_sponsor_duration
        if args.intro_music:
            parts.append("with music")
        print(f"  Intro   : {total_intro:.1f}s ({', '.join(parts)})")
    print()

    offset = args.offset if args.offset is not None else find_sync_offset(speaker_path, slides_path)
    crop   = _resolve_crop(args, slides_path, offset)

    with tempfile.TemporaryDirectory(prefix="merger_") as tmp_dir:
        tmp = Path(tmp_dir)

        # Step 1: trim slides to remove sponsor intro
        if offset >= 0.1:
            print(f"  Trimming {offset:.2f}s from slides start...")
            slides_trimmed = tmp / "trimmed.mp4"
            ffmpeg("-ss", offset, "-i", slides_path, "-c", "copy", slides_trimmed)
        else:
            slides_trimmed = slides_path

        # Step 2: pad slides so they outlast the speaker
        slides_ready = pad_slides(slides_trimmed, tmp / "padded.mp4", speaker_dur)

        # Step 3: merge slides + speaker side by side
        print()
        merged = tmp / "merged.mp4" if has_intro else output_path
        merge(
            slides_ready, speaker_path, merged,
            slides_col_w, args.width, args.height,
            background_path=Path(args.background) if args.background else None,
            crop=crop,
        )

        # Step 4: build intro and prepend it
        if has_intro:
            print()
            intro_result = build_intro(
                tmp, args.width, args.height,
                community_logo=Path(args.intro_community) if args.intro_community else None,
                sponsor_logo=Path(args.intro_sponsor)     if args.intro_sponsor   else None,
                music_path=Path(args.intro_music)         if args.intro_music     else None,
                community_dur=args.intro_community_duration,
                sponsor_dur=args.intro_sponsor_duration,
            )
            if intro_result:
                intro_path, intro_dur = intro_result
                prepend_intro(intro_path, merged, output_path, intro_dur)

    dur = probe_duration(output_path)
    mb  = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Done: {output_path.name} ({mb:.1f} MB, {int(dur // 60)}m{int(dur % 60)}s)\n")


if __name__ == "__main__":
    main()
