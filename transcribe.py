#!/usr/bin/env python3
"""
transcribe.py — Extract a speaker-diarized transcription from a video file.

Usage:
    python3 transcribe.py video.mp4
    python3 transcribe.py video.mp4 -o transcription.txt
    python3 transcribe.py video.mp4 --language es
    python3 transcribe.py video.mp4 --model large-v3 --num-speakers 3
    HF_TOKEN=hf_xxx python3 transcribe.py video.mp4

Environment:
    HF_TOKEN    HuggingFace access token (required for speaker diarization).
                Can also be passed via --hf-token.

Requirements:
    brew install ffmpeg
    pip3 install faster-whisper pyannote.audio torch torchaudio

    Accept the pyannote model terms before first use:
      https://huggingface.co/pyannote/segmentation-3.0
      https://huggingface.co/pyannote/speaker-diarization-3.1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def _progress_dots(stop_event: threading.Event) -> None:
    while not stop_event.wait(60):
        print(".", end="", flush=True)


# ── Audio extraction ───────────────────────────────────────────────────────────

def extract_audio(video_path: Path, audio_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
         str(audio_path)],
        capture_output=True, text=True, check=True,
    )


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe(audio_path: Path, model_size: str, language: str | None,
               device: str, compute_type: str) -> list[dict]:
    from faster_whisper import WhisperModel

    print(f"Loading Whisper model '{model_size}' on {device} ({compute_type})…")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("Transcribing…", end="", flush=True)
    stop = threading.Event()
    dots = threading.Thread(target=_progress_dots, args=(stop,), daemon=True)
    dots.start()

    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
    )

    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({"start": w.start, "end": w.end, "text": w.word})

    stop.set()
    dots.join()
    print(f"\nDetected language: {info.language} (probability {info.language_probability:.2f})")
    print(f"Transcribed {len(words)} words.")
    return words


# ── Speaker diarization ───────────────────────────────────────────────────────

def diarize(audio_path: Path, hf_token: str,
            num_speakers: int | None, device: str) -> list[dict]:
    import torch
    import huggingface_hub
    from huggingface_hub import login

    login(token=hf_token)

    _original_download = huggingface_hub.hf_hub_download
    def _patched_download(*args, **kwargs):
        token = kwargs.pop("use_auth_token", None)
        if token and "token" not in kwargs:
            kwargs["token"] = token
        return _original_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = _patched_download

    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    from pyannote.audio import Pipeline

    print("Loading diarization pipeline…")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    torch_device = torch.device(device)
    pipeline.to(torch_device)

    kwargs = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers

    print("Running speaker diarization…", end="", flush=True)
    stop = threading.Event()
    dots = threading.Thread(target=_progress_dots, args=(stop,), daemon=True)
    dots.start()

    result = pipeline(str(audio_path), **kwargs)

    stop.set()
    dots.join()

    segments = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    speakers = {s["speaker"] for s in segments}
    print(f"\nFound {len(speakers)} speaker(s).")
    return segments


# ── Merge transcription + diarization ─────────────────────────────────────────

def assign_speakers(words: list[dict], diar_segments: list[dict]) -> list[dict]:
    result = []
    for word in words:
        mid = (word["start"] + word["end"]) / 2
        speaker = "Unknown"
        for seg in diar_segments:
            if seg["start"] <= mid <= seg["end"]:
                speaker = seg["speaker"]
                break
        result.append({**word, "speaker": speaker})
    return result


def format_transcript(speaker_words: list[dict]) -> str:
    if not speaker_words:
        return ""

    speaker_map: dict[str, str] = {}
    counter = 0

    lines: list[str] = []
    current_speaker = None
    current_text: list[str] = []

    for word in speaker_words:
        spk = word["speaker"]
        if spk not in speaker_map:
            counter += 1
            speaker_map[spk] = f"Speaker {counter}"

        if spk != current_speaker:
            if current_text:
                lines.append(f"{speaker_map[current_speaker]}: {''.join(current_text).strip()}")
            current_speaker = spk
            current_text = [word["text"]]
        else:
            current_text.append(word["text"])

    if current_text:
        lines.append(f"{speaker_map[current_speaker]}: {''.join(current_text).strip()}")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract a speaker-diarized transcription from a video.",
    )
    p.add_argument("video", type=Path, help="Input video file")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output .txt path (default: <video_stem>_transcription.txt)")
    p.add_argument("--language", default=None,
                   help="Language code, e.g. 'es', 'en' (default: auto-detect)")
    p.add_argument("--model", default="large-v3",
                   help="Whisper model size (default: large-v3)")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Number of speakers if known (improves diarization)")
    p.add_argument("--device", default="cpu",
                   help="Torch device: cpu, cuda, mps (default: cpu)")
    p.add_argument("--compute-type", default="int8",
                   help="Whisper compute type: int8, float16, float32 (default: int8)")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (default: $HF_TOKEN env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        sys.exit(f"Error: file not found: {args.video}")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        sys.exit(
            "Error: HuggingFace token required for speaker diarization.\n"
            "Set HF_TOKEN env var or pass --hf-token.\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
            "Then accept terms at:\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            "  https://huggingface.co/pyannote/speaker-diarization-3.1"
        )

    output = args.output or args.video.with_name(f"{args.video.stem}_transcription.txt")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"

        print(f"Extracting audio from {args.video}…")
        extract_audio(args.video, wav_path)

        words = transcribe(wav_path, args.model, args.language,
                           args.device, args.compute_type)

        diar_segments = diarize(wav_path, hf_token, args.num_speakers, args.device)

    speaker_words = assign_speakers(words, diar_segments)
    transcript = format_transcript(speaker_words)

    output.write_text(transcript, encoding="utf-8")
    print(f"\nTranscription saved to {output}")
    print(f"Preview:\n{'─' * 60}")
    for line in transcript.splitlines()[:20]:
        print(line)
    if transcript.count("\n") > 20:
        print(f"… ({transcript.count(chr(10)) + 1 - 20} more lines)")


if __name__ == "__main__":
    main()
