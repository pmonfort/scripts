#!/usr/bin/env python3
"""
add_intro.py - Prepend an intro sequence to any video.

Usage:
    python3 add_intro.py input.mp4 output.mp4 \\
        --intro-community logo_ruby.png \\
        --intro-sponsor   logo_sponsor.png \\
        --intro-music     music.mp3

Options:
    --intro-community           Path to community logo PNG
    --intro-sponsor             Path to sponsor logo PNG
    --intro-music               Path to intro music MP3
    --intro-community-duration  Seconds for community screen (default: 2)
    --intro-sponsor-duration    Seconds for sponsor screen (default: 2)
    --transition                Fade duration at the join in seconds (default: 1)
    --width                     Output width (default: from input video)
    --height                    Output height (default: from input video)

Requirements:
    brew install ffmpeg
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, show_progress=False):
    cmd = ["ffmpeg", "-y"] + [str(a) for a in cmd]
    result = subprocess.run(cmd) if show_progress else subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{getattr(result, 'stderr', '')[-800:]}")


def duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def dimensions(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def make_logo_clip(logo_path, clip_dur, out_path, output_w, output_h):
    fps    = 30
    fade_s = 0.4
    logo_w = int(output_w * 0.4)
    run([
        "-f", "lavfi", "-i", f"color=white:s={output_w}x{output_h}:r={fps}",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-filter_complex",
            f"movie='{logo_path}',scale={logo_w}:-2,loop=loop=-1:size=1:start=0[logo];"
            f"[0:v][logo]overlay=(W-w)/2:(H-h)/2,"
            f"fade=t=in:st=0:d={fade_s}:color=white,"
            f"fade=t=out:st={clip_dur - fade_s}:d={fade_s}:color=white[outv]",
        "-map", "[outv]", "-map", "1:a",
        "-frames:v", str(int(clip_dur * fps)), "-t", str(clip_dur),
        "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
        out_path,
    ])


def build_intro(tmp, output_w, output_h, community_logo, sponsor_logo, music_path,
                community_dur, sponsor_dur):
    clips     = []
    total_dur = 0.0

    if community_logo:
        p = tmp / "intro_community.mp4"
        make_logo_clip(community_logo, community_dur, p, output_w, output_h)
        clips.append(p)
        total_dur += community_dur

    if sponsor_logo:
        p = tmp / "intro_sponsor.mp4"
        make_logo_clip(sponsor_logo, sponsor_dur, p, output_w, output_h)
        clips.append(p)
        total_dur += sponsor_dur

    if len(clips) == 1:
        intro_video = clips[0]
    else:
        n        = len(clips)
        inputs   = [arg for c in clips for arg in ("-i", str(c))]
        filter_v = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
        filter_a = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[outa]"
        intro_video = tmp / "intro_video.mp4"
        run([
            *inputs,
            "-filter_complex", f"{filter_v};{filter_a}",
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart",
            intro_video,
        ])

    if not music_path:
        return intro_video, total_dur

    intro_with_music = tmp / "intro_with_music.mp4"
    run([
        "-i", str(intro_video), "-i", str(music_path),
        "-filter_complex",
            f"[1:a]atrim=0:{total_dur},afade=t=out:st={total_dur - 0.5}:d=0.5[music]",
        "-map", "0:v", "-map", "[music]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest",
        intro_with_music,
    ])
    return intro_with_music, total_dur


def prepend_intro(intro_path, intro_dur, main_path, output_path, transition_s):
    fade_out_st = intro_dur - transition_s
    run([
        "-i", str(intro_path), "-i", str(main_path),
        "-filter_complex",
            f"[0:v]fade=t=out:st={fade_out_st}:d={transition_s}:color=white[v0];"
            f"[1:v]fade=t=in:st=0:d={transition_s}:color=white[v1];"
            f"[0:a]afade=t=out:st={fade_out_st}:d={transition_s}[a0];"
            f"[1:a]afade=t=in:st=0:d={transition_s}[a1];"
            f"[v0][a0][v1][a1]concat=n=2:v=1:a=1[outv][outa]",
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        output_path,
    ], show_progress=True)


def main():
    parser = argparse.ArgumentParser(description="Prepend an intro sequence to a video.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--intro-community",          default=None)
    parser.add_argument("--intro-sponsor",            default=None)
    parser.add_argument("--intro-music",              default=None)
    parser.add_argument("--intro-community-duration", type=float, default=2.0)
    parser.add_argument("--intro-sponsor-duration",   type=float, default=2.0)
    parser.add_argument("--transition",               type=float, default=1.0)
    parser.add_argument("--width",                    type=int,   default=None)
    parser.add_argument("--height",                   type=int,   default=None)
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"Not found: {input_path}")
    if not args.intro_community and not args.intro_sponsor:
        sys.exit("Provide at least one of --intro-community or --intro-sponsor")
    for label, val in [("--intro-community", args.intro_community),
                       ("--intro-sponsor",   args.intro_sponsor),
                       ("--intro-music",     args.intro_music)]:
        if val and not Path(val).exists():
            sys.exit(f"Not found ({label}): {val}")

    w, h     = dimensions(input_path)
    output_w = args.width  or w
    output_h = args.height or h

    total_intro = (args.intro_community_duration if args.intro_community else 0) + \
                  (args.intro_sponsor_duration   if args.intro_sponsor   else 0)
    parts = []
    if args.intro_community: parts.append(f"community {args.intro_community_duration}s")
    if args.intro_sponsor:   parts.append(f"sponsor {args.intro_sponsor_duration}s")
    if args.intro_music:     parts.append("with music")

    print(f"\n[AddIntro]")
    print(f"  Input:      {input_path.name}")
    print(f"  Output:     {output_path.name}")
    print(f"  Intro:      {total_intro:.1f}s ({', '.join(parts)})")
    print(f"  Transition: {args.transition}s\n")

    with tempfile.TemporaryDirectory(prefix="intro_") as tmp:
        tmp = Path(tmp)
        intro, intro_dur = build_intro(
            tmp, output_w, output_h,
            community_logo = Path(args.intro_community) if args.intro_community else None,
            sponsor_logo   = Path(args.intro_sponsor)   if args.intro_sponsor   else None,
            music_path     = Path(args.intro_music)     if args.intro_music     else None,
            community_dur  = args.intro_community_duration,
            sponsor_dur    = args.intro_sponsor_duration,
        )
        prepend_intro(intro, intro_dur, input_path, output_path, args.transition)

    dur = duration(output_path)
    mb  = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Done: {output_path.name} ({mb:.1f} MB, {int(dur//60)}m{int(dur%60)}s)\n")


if __name__ == "__main__":
    main()
