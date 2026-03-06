#!/usr/bin/env python3
"""
video_merger.py - Merge slides + speaker into a synchronized side-by-side video.

Requirements:
  1. Merge slides (75%) and speaker (25%) side by side.
  2. Crop slides video to remove participant column on the right (~72% of width).
  3. Speaker is the master: output starts and ends with the speaker.
  4. Sync slides to speaker via audio cross-correlation, trimming the sponsor intro.
  5. Speaker video plays in full, untouched.
  6. If slides end before speaker, freeze the last frame until speaker finishes.
  7. Output resolution: 1920x1080 (configurable).
  8. Auto-detect slides offset. Manual override via --offset.

Usage:
    python3 video_merger.py slides.mp4 speaker.mp4 output.mp4
    python3 video_merger.py slides.mp4 speaker.mp4 output.mp4 --offset 312
    python3 video_merger.py slides.mp4 speaker.mp4 output.mp4 --slides-pct 0.74

Requirements:
    brew install ffmpeg
    pip3 install numpy scipy
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, filtfilt


def run(cmd, show_progress=False):
    cmd = ["ffmpeg", "-y"] + [str(a) for a in cmd]
    result = subprocess.run(cmd) if show_progress else subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{getattr(result, 'stderr', '')[-600:]}")


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return float(r.stdout.strip())


def dimensions(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def fps(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    n, d = r.stdout.strip().split("/")
    return float(n) / float(d)


def find_offset(speaker_path, slides_path, max_search_s=700):
    """Cross-correlate audio to find how many seconds to skip from slides start."""
    print("  Detecting sync via cross-correlation...")
    with tempfile.TemporaryDirectory() as tmp:
        spk_wav = Path(tmp) / "spk.wav"
        sld_wav = Path(tmp) / "sld.wav"

        run(["-i", speaker_path, "-ac", "1", "-ar", "16000", "-vn", "-t", "60", spk_wav])
        run(["-i", slides_path,  "-ac", "1", "-ar", "16000", "-vn", "-t", str(max_search_s + 60), sld_wav])

        rate, spk = wavfile.read(spk_wav)
        _,    sld = wavfile.read(sld_wav)

        def process(x):
            x = x.mean(axis=1) if x.ndim > 1 else x
            x = x.astype(np.float32)
            nyq = rate / 2
            b, a = butter(4, [300 / nyq, 3400 / nyq], btype="band")
            return filtfilt(b, a, x)

        spk_f = process(spk)
        sld_f = process(sld)

        corr   = correlate(sld_f[:int(max_search_s * rate) + len(spk_f)], spk_f, mode="valid")
        offset = int(np.argmax(corr)) / rate
        print(f"  Slides offset: {offset:.2f}s")
        return offset


def pad_slides(slides_path, output_path, target_duration):
    """Extend slides with a frozen last frame so they are longer than the speaker."""
    gap = target_duration - duration(slides_path)
    if gap <= 0:
        return slides_path

    print(f"  Padding slides with {gap:.1f}s of frozen last frame...")
    tmp = Path(output_path).parent

    last_frame = tmp / "last_frame.png"
    frozen     = tmp / "frozen.mp4"
    concat_txt = tmp / "concat.txt"

    run(["-sseof", "-1", "-i", slides_path, "-vframes", "1", "-q:v", "2", last_frame])
    run(["-loop", "1", "-framerate", fps(slides_path), "-i", last_frame,
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-t", str(gap + 2), "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast",
         "-c:a", "aac", "-b:a", "64k", "-pix_fmt", "yuv420p", frozen])

    concat_txt.write_text(
        "file '" + str(slides_path.resolve()) + "'\n"
        "file '" + str(frozen.resolve()) + "'\n"
    )
    run(["-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", output_path])
    return output_path


def merge(slides_path, speaker_path, output_path, slides_col_w, output_w, output_h):
    """Combine slides (75%) and speaker (25%) side by side. Speaker drives duration via -shortest."""
    slides_out_w = (int(output_w * 0.75) // 2) * 2
    spkr_out_w   = output_w - slides_out_w
    _, slides_h  = dimensions(slides_path)

    print(f"  Encoding {output_w}x{output_h} (slides {slides_out_w}px | speaker {spkr_out_w}px)...")

    fg = (
        f"[0:v]crop={slides_col_w}:{slides_h}:0:0,"
        f"scale={slides_out_w}:{output_h}:force_original_aspect_ratio=decrease,"
        f"pad={slides_out_w}:{output_h}:(ow-iw)/2:(oh-ih)/2:black[sv];"
        f"[1:v]scale={spkr_out_w}:{output_h}:force_original_aspect_ratio=decrease,"
        f"pad={spkr_out_w}:{output_h}:(ow-iw)/2:(oh-ih)/2:black[spv];"
        f"[sv][spv]hstack=inputs=2[outv]"
    )
    run(["-i", slides_path, "-i", speaker_path,
         "-filter_complex", fg, "-map", "[outv]", "-map", "1:a",
         "-shortest",
         "-c:v", "libx264", "-crf", "23", "-preset", "fast",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
         output_path], show_progress=True)


def main():
    parser = argparse.ArgumentParser(description="Merge slides + speaker side by side.")
    parser.add_argument("slides")
    parser.add_argument("speaker")
    parser.add_argument("output")
    parser.add_argument("--slides-pct", type=float, default=0.72)
    parser.add_argument("--offset",     type=float, default=None)
    parser.add_argument("--width",      type=int,   default=1920)
    parser.add_argument("--height",     type=int,   default=1080)
    args = parser.parse_args()

    slides_path  = Path(args.slides)
    speaker_path = Path(args.speaker)
    output_path  = Path(args.output)

    if not slides_path.exists():  sys.exit(f"Not found: {slides_path}")
    if not speaker_path.exists(): sys.exit(f"Not found: {speaker_path}")

    speaker_dur  = duration(speaker_path)
    w, _         = dimensions(slides_path)
    slides_col_w = (int(w * args.slides_pct) // 2) * 2

    print(f"\n[VideoMerger]")
    print(f"  Speaker: {speaker_path.name} ({int(speaker_dur//60)}m{int(speaker_dur%60)}s) — master")
    print(f"  Slides:  {slides_path.name}")
    print(f"  Output:  {output_path.name}\n")

    offset = args.offset if args.offset is not None else find_offset(speaker_path, slides_path)

    with tempfile.TemporaryDirectory(prefix="merger_") as tmp:
        tmp = Path(tmp)

        # Step 1: trim sponsor intro from slides
        slides_trimmed = tmp / "trimmed.mp4"
        if offset >= 0.1:
            print(f"  Trimming {offset:.2f}s from slides start...")
            run(["-ss", offset, "-i", slides_path, "-c", "copy", slides_trimmed])
        else:
            slides_trimmed = slides_path

        # Step 2: pad slides so they outlast the speaker (enables safe use of -shortest)
        slides_padded = tmp / "padded.mp4"
        slides_padded = pad_slides(slides_trimmed, slides_padded, speaker_dur)

        # Step 3: merge
        print()
        merge(slides_padded, speaker_path, output_path, slides_col_w, args.width, args.height)

    dur = duration(output_path)
    mb  = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Done: {output_path.name} ({mb:.1f} MB, {int(dur//60)}m{int(dur%60)}s)\n")


if __name__ == "__main__":
    main()
