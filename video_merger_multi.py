#!/usr/bin/env python3
"""
video_merger_multi.py — Merge slides + multiple speaker videos into a synchronized
                        side-by-side conference video.

The SLIDES video is the master timeline. Each speaker video is synchronized to the
slides using audio cross-correlation. Portions of the slides with no speaker coverage
are shown as black in the speaker panel.

Usage:
    python3 video_merger_multi.py slides.mp4 spk1.mp4 -o output.mp4
    python3 video_merger_multi.py slides.mp4 spk1.mp4 spk2.mp4 -o output.mp4
    python3 video_merger_multi.py slides.mp4 spk1.mp4 spk2.mp4 -o output.mp4 \\
        --offsets 0 3600
    python3 video_merger_multi.py slides.mp4 spk1.mp4 spk2.mp4 -o output.mp4 \\
        --background background.jpg --slides-autocrop \\
        --intro-community logo.png --intro-music music.mp3

Offset convention (--offsets):
    Positive → speaker starts this many seconds into the slides timeline.
    Negative → speaker has a preamble of |offset| seconds before slides begins.
    Omit    → auto-detect via audio cross-correlation.

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

SLIDES_COLUMN_FRACTION = 0.75
BACKGROUND_MARGIN_PX   = 30
LOGO_WIDTH_FRACTION    = 0.40
LOGO_FADE_DURATION_S   = 0.4
INTRO_FPS              = 30
MUSIC_FADE_OUT_S       = 0.5
COMPOSITE_FPS          = 30   # frame rate used for black filler clips


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CropRegion:
    x: int
    y: int
    w: int
    h: int

    def constrain_width(self, max_w: int) -> CropRegion:
        return CropRegion(self.x, self.y, min(self.w, max_w - self.x), self.h)


@dataclass
class SpeakerSegment:
    """One speaker video and where it falls in the slides timeline."""
    path: Path
    slides_start: float   # seconds into slides where this speaker's content begins
    speaker_trim: float   # seconds to skip from the speaker video's start (preamble)
    duration: float       # effective duration = raw_duration - speaker_trim

    @property
    def slides_end(self) -> float:
        return self.slides_start + self.duration


# ── FFmpeg helpers ─────────────────────────────────────────────────────────────

def ffmpeg(*args: str | Path, show_progress: bool = False) -> None:
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

def find_speaker_offset(
    speaker_path: Path,
    slides_path: Path,
    slides_duration: float,
) -> tuple[float, float]:
    """Find where a speaker video aligns with the slides (master) timeline.

    Returns (slides_start, speaker_trim):
        slides_start — seconds into slides where this speaker's content begins.
        speaker_trim — seconds to skip from speaker start (speaker has a preamble).

    Uses bidirectional full cross-correlation:
        lag > 0  → speaker content starts at slides[lag:] → slides_start=lag, trim=0
        lag < 0  → speaker content begins before slides   → slides_start=0, trim=|lag|
    """
    print("    Running audio cross-correlation...")
    max_load_s = min(int(slides_duration) + 60, 14400)   # cap at 4 h to limit memory

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp     = Path(tmp_dir)
        spk_wav = tmp / "spk.wav"
        sld_wav = tmp / "sld.wav"

        ffmpeg("-i", str(speaker_path), "-ac", "1", "-ar", "16000", "-vn",
               "-t", str(max_load_s), spk_wav)
        ffmpeg("-i", str(slides_path), "-ac", "1", "-ar", "16000", "-vn",
               "-t", str(max_load_s), sld_wav)

        rate, spk_data = wavfile.read(spk_wav)
        _,    sld_data = wavfile.read(sld_wav)

        def bandpass_voice(sig: np.ndarray) -> np.ndarray:
            mono = sig.mean(axis=1) if sig.ndim > 1 else sig
            mono = mono.astype(np.float32)
            b, a = butter(4, [300 / (rate / 2), 3400 / (rate / 2)], btype="band")
            return filtfilt(b, a, mono)

        spk_f = bandpass_voice(spk_data)
        sld_f = bandpass_voice(sld_data)

        corr   = correlate(sld_f, spk_f, mode="full")
        center = len(spk_f) - 1
        sr     = int(max_load_s * rate)
        lo     = max(0, center - sr)
        hi     = min(len(corr), center + sr + 1)
        peak   = lo + int(np.argmax(corr[lo:hi]))
        lag    = (peak - center) / rate   # positive: sld leads; negative: spk leads

    slides_start = max(0.0, lag)
    speaker_trim = max(0.0, -lag)
    print(f"    → slides_start={slides_start:.2f}s  speaker_trim={speaker_trim:.2f}s")
    return slides_start, speaker_trim


# ── Slides crop detection ──────────────────────────────────────────────────────

def detect_crop(slides_path: Path, sample_s: int = 60) -> CropRegion | None:
    print("  Auto-detecting slides crop...")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(slides_path),
         "-t", str(sample_s),
         "-vf", "cropdetect=limit=32:round=2:skip=2",
         "-f", "null", "-"],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
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


# ── Speaker composite ──────────────────────────────────────────────────────────

def _make_black_clip(output: Path, duration: float, w: int, h: int) -> None:
    """Create a silent black video clip at COMPOSITE_FPS."""
    ffmpeg(
        "-f", "lavfi", "-i", f"color=black:s={w}x{h}:r={COMPOSITE_FPS}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", str(duration),
        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k",
        output,
    )


def _extract_speaker_clip(
    speaker_path: Path,
    output: Path,
    start: float,
    duration: float,
    w: int,
    h: int,
) -> None:
    """Extract [start, start+duration] from speaker, scaled/padded to w×h at COMPOSITE_FPS."""
    ffmpeg(
        "-ss", str(start), "-i", str(speaker_path),
        "-t", str(duration),
        "-vf", (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"fps={COMPOSITE_FPS},setsar=1"
        ),
        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", "-ar", "44100", "-ac", "2",
        output,
    )


def build_speaker_composite(
    segments: list[SpeakerSegment],
    slides_duration: float,
    output_path: Path,
    tmp: Path,
    ref_w: int,
    ref_h: int,
) -> None:
    """Assemble speaker segments + black gaps into one video of exactly slides_duration."""
    segs = sorted(segments, key=lambda s: s.slides_start)

    clips: list[Path] = []
    current = 0.0

    for i, seg in enumerate(segs):
        seg_slides_start = seg.slides_start
        seg_slides_end   = seg.slides_end

        # Clip end at next segment start to avoid overlaps
        if i + 1 < len(segs) and seg_slides_end > segs[i + 1].slides_start:
            print(
                f"    Warning: {seg.path.name} overlaps with {segs[i+1].path.name}; "
                f"clipping at {segs[i+1].slides_start:.1f}s"
            )
            seg_slides_end = segs[i + 1].slides_start

        seg_slides_end = min(seg_slides_end, slides_duration)

        if seg_slides_start >= slides_duration:
            break

        # Leading black gap
        gap = seg_slides_start - current
        if gap > 0.05:
            black = tmp / f"black_{i:03d}.mp4"
            print(f"    Gap  [{current:.1f}s – {seg_slides_start:.1f}s]  black ({gap:.1f}s)")
            _make_black_clip(black, gap, ref_w, ref_h)
            clips.append(black)

        # Speaker clip (handle case where current > seg_slides_start due to overlap resolution)
        actual_start   = max(seg_slides_start, current)
        speaker_offset = seg.speaker_trim + (actual_start - seg_slides_start)
        clip_dur       = seg_slides_end - actual_start

        if clip_dur > 0.05:
            clip = tmp / f"speaker_{i:03d}.mp4"
            print(
                f"    Spkr [{actual_start:.1f}s – {seg_slides_end:.1f}s]"
                f"  {seg.path.name} from {speaker_offset:.1f}s ({clip_dur:.1f}s)"
            )
            _extract_speaker_clip(seg.path, clip, speaker_offset, clip_dur, ref_w, ref_h)
            clips.append(clip)

        current = seg_slides_end

    # Trailing black
    tail = slides_duration - current
    if tail > 0.05:
        black = tmp / "black_tail.mp4"
        print(f"    Gap  [{current:.1f}s – {slides_duration:.1f}s]  black ({tail:.1f}s)")
        _make_black_clip(black, tail, ref_w, ref_h)
        clips.append(black)

    if not clips:
        print("  No speaker segments — building full black track.")
        _make_black_clip(output_path, slides_duration, ref_w, ref_h)
        return

    concat_txt = tmp / "composite_concat.txt"
    concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in clips) + "\n")
    print(f"  Concatenating {len(clips)} clip(s)...")
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", output_path)


# ── Merge ──────────────────────────────────────────────────────────────────────

def _build_background_filtergraph(
    crop: CropRegion,
    slides_out_w: int,
    spkr_out_w: int,
    output_w: int,
    output_h: int,
) -> str:
    m = BACKGROUND_MARGIN_PX
    return (
        f"[2:v]scale={output_w}:{output_h}:force_original_aspect_ratio=increase,"
        f"crop={output_w}:{output_h},setsar=1,format=yuv420p[bg];"
        f"[0:v]crop={crop.w}:{crop.h}:{crop.x}:{crop.y},"
        f"scale={slides_out_w - 2*m}:{output_h - 2*m}:force_original_aspect_ratio=decrease,"
        f"setsar=1,format=yuv420p[sv];"
        f"[1:v]scale={spkr_out_w - 2*m}:{output_h - 2*m}:force_original_aspect_ratio=decrease,"
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
    speaker_composite_path: Path,
    output_path: Path,
    slides_col_w: int,
    output_w: int,
    output_h: int,
    background_path: Path | None = None,
    crop: CropRegion | None = None,
) -> None:
    """Encode final video: slides (75%) + speaker composite (25%). Audio from slides."""
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
        inputs = ["-i", str(slides_path), "-i", str(speaker_composite_path),
                  "-loop", "1", "-i", str(background_path)]
    else:
        fg     = _build_plain_filtergraph(slides_crop, slides_out_w, spkr_out_w, output_w, output_h)
        inputs = ["-i", str(slides_path), "-i", str(speaker_composite_path)]

    ffmpeg(
        *inputs,
        "-filter_complex", fg,
        "-map", "[outv]", "-map", "0:a",   # audio from slides (master timeline)
        "-shortest",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        output_path,
        show_progress=True,
    )


# ── Intro ──────────────────────────────────────────────────────────────────────

def _make_logo_clip(
    logo_path: Path,
    duration_s: float,
    out_path: Path,
    output_w: int,
    output_h: int,
) -> None:
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
        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
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
            "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
            intro_video,
        )

    if not music_path:
        return intro_video, total_dur

    intro_with_music = tmp / "intro_with_music.mp4"
    ffmpeg(
        "-i", str(intro_video), "-i", str(music_path),
        "-filter_complex",
            f"[1:a]atrim=0:{total_dur},"
            f"afade=t=out:st={total_dur - MUSIC_FADE_OUT_S}:d={MUSIC_FADE_OUT_S}[music]",
        "-map", "0:v", "-map", "[music]",
        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-shortest",
        intro_with_music,
    )
    return intro_with_music, total_dur


def prepend_intro(
    intro_path: Path,
    main_path: Path,
    output_path: Path,
    intro_dur: float,
    transition_s: float = 1.0,
) -> None:
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
        prog="video_merger_multi.py",
        description=(
            "Merge slides + multiple speaker videos into a synchronized side-by-side video. "
            "Slides is the master timeline; gaps with no speaker coverage are shown as black."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("slides",   help="Slides recording (MP4) — master timeline")
    p.add_argument("speakers", nargs="+", help="One or more speaker recordings (MP4)")
    p.add_argument("-o", "--output", required=True, help="Output file path (MP4)")

    sync = p.add_argument_group("sync")
    sync.add_argument(
        "--offsets", nargs="*", type=float, default=None, metavar="S",
        help=(
            "Signed offset in seconds for each speaker (space-separated). "
            "Positive: speaker starts this many seconds into slides. "
            "Negative: speaker has a preamble of |offset| seconds before slides begins. "
            "Provide fewer values than speakers to auto-detect the remaining ones."
        ),
    )

    layout = p.add_argument_group("layout")
    layout.add_argument("--slides-pct", type=float, default=0.72, metavar="F",
                        help="Fraction of slides width that contains slides (excludes thumbnails)")
    layout.add_argument("--width",  type=int, default=1920, metavar="PX")
    layout.add_argument("--height", type=int, default=1080, metavar="PX")
    layout.add_argument("--background", default=None, metavar="IMAGE",
                        help="Background image shown behind slides and speaker panels")

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
    intro.add_argument("--intro-sponsor",   default=None, metavar="PNG",
                       help="Sponsor logo PNG shown second")
    intro.add_argument("--intro-music",     default=None, metavar="MP3",
                       help="Background music for the intro")
    intro.add_argument("--intro-community-duration", type=float, default=2.0, metavar="S")
    intro.add_argument("--intro-sponsor-duration",   type=float, default=2.0, metavar="S")

    return p


def _validate_paths(args: argparse.Namespace) -> None:
    if not Path(args.slides).exists():
        sys.exit(f"error: slides file not found: {args.slides}")
    for spk in args.speakers:
        if not Path(spk).exists():
            sys.exit(f"error: speaker file not found: {spk}")
    for path, label in [
        (args.background,      "--background"),
        (args.intro_community, "--intro-community"),
        (args.intro_sponsor,   "--intro-sponsor"),
        (args.intro_music,     "--intro-music"),
    ]:
        if path and not Path(path).exists():
            sys.exit(f"error: {label} file not found: {path}")
    if args.offsets and len(args.offsets) > len(args.speakers):
        sys.exit(
            f"error: --offsets has {len(args.offsets)} values "
            f"but only {len(args.speakers)} speaker(s) provided"
        )


def _resolve_crop(args: argparse.Namespace, slides_path: Path) -> CropRegion | None:
    manual = (args.slides_crop_top, args.slides_crop_bottom,
              args.slides_crop_left, args.slides_crop_right)
    if any(manual):
        top, bottom, left, right = manual
        sw, sh = probe_dimensions(slides_path)
        return CropRegion(x=left, y=top, w=sw - left - right, h=sh - top - bottom)
    if args.slides_autocrop:
        return detect_crop(slides_path)
    return None


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    _validate_paths(args)

    slides_path   = Path(args.slides)
    speaker_paths = [Path(s) for s in args.speakers]
    output_path   = Path(args.output)

    slides_dur   = probe_duration(slides_path)
    slides_w, _  = probe_dimensions(slides_path)
    slides_col_w = (int(slides_w * args.slides_pct) // 2) * 2
    ref_w, ref_h = probe_dimensions(speaker_paths[0])
    has_intro    = bool(args.intro_community or args.intro_sponsor)

    print("\n[VideoMergerMulti]")
    print(f"  Slides   : {slides_path.name}  ({int(slides_dur // 60)}m{int(slides_dur % 60)}s)  ← master")
    print(f"  Speakers : {len(speaker_paths)}")
    for i, sp in enumerate(speaker_paths):
        dur = probe_duration(sp)
        print(f"    [{i+1}] {sp.name}  ({int(dur // 60)}m{int(dur % 60)}s)")
    print(f"  Output   : {output_path.name}")
    if has_intro:
        print(f"  Intro    : community={args.intro_community}  sponsor={args.intro_sponsor}")
    print()

    # ── Resolve offsets ─────────────────────────────────────────────────────────
    segments: list[SpeakerSegment] = []
    for i, spk_path in enumerate(speaker_paths):
        manual = args.offsets[i] if (args.offsets and i < len(args.offsets)) else None

        if manual is not None:
            slides_start = max(0.0, manual)
            speaker_trim = max(0.0, -manual)
            print(f"  Speaker {i+1}/{len(speaker_paths)} ({spk_path.name}): "
                  f"offset={manual:+.2f}s (manual)")
        else:
            print(f"  Syncing speaker {i+1}/{len(speaker_paths)}: {spk_path.name}")
            slides_start, speaker_trim = find_speaker_offset(spk_path, slides_path, slides_dur)

        raw_dur = probe_duration(spk_path)
        segments.append(SpeakerSegment(
            path=spk_path,
            slides_start=slides_start,
            speaker_trim=speaker_trim,
            duration=raw_dur - speaker_trim,
        ))

    print()
    for seg in sorted(segments, key=lambda s: s.slides_start):
        print(
            f"  {seg.path.name}: slides [{seg.slides_start:.1f}s – {seg.slides_end:.1f}s]"
            + (f"  (trim speaker {seg.speaker_trim:.1f}s)" if seg.speaker_trim > 0.1 else "")
        )

    crop = _resolve_crop(args, slides_path)

    with tempfile.TemporaryDirectory(prefix="merger_multi_") as tmp_dir:
        tmp = Path(tmp_dir)

        # Step 1: build speaker composite
        print("\n[Step 1] Building speaker composite...")
        speaker_composite = tmp / "speaker_composite.mp4"
        build_speaker_composite(segments, slides_dur, speaker_composite, tmp, ref_w, ref_h)

        # Step 2: merge slides + composite
        print("\n[Step 2] Merging slides + speaker composite...")
        merged = tmp / "merged.mp4" if has_intro else output_path
        merge(
            slides_path, speaker_composite, merged,
            slides_col_w, args.width, args.height,
            background_path=Path(args.background) if args.background else None,
            crop=crop,
        )

        # Step 3: build and prepend intro
        if has_intro:
            print("\n[Step 3] Building intro...")
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
