import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import webrtcvad

def extract_audio(video_path: Path, out_wav_path: Path, sample_rate=16000):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", str(sample_rate), "-ac", "1",
        "-vn", "-f", "wav", str(out_wav_path)
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def read_wav(path: Path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

def frame_generator(frame_ms, audio, sample_rate):
    frame_size = int(sample_rate * (frame_ms / 1000.0))
    total_frames = len(audio)
    offset = 0
    while offset + frame_size <= total_frames:
        frame = audio[offset:offset + frame_size]
        pcm16 = (frame * 32767).astype(np.int16).tobytes()
        start = offset / sample_rate
        end = (offset + frame_size) / sample_rate
        yield start, end, pcm16
        offset += frame_size

def vad_collector(audio, sample_rate, aggressiveness=2, frame_ms=30, padding_ms=300):
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(frame_ms, audio, sample_rate))
    is_speech = [vad.is_speech(f[2], sample_rate) for f in frames]
    segs = []
    start_idx = None
    for i, speech in enumerate(is_speech):
        if speech and start_idx is None:
            start_idx = i
        elif not speech and start_idx is not None:
            seg_start = frames[max(0, start_idx - int(padding_ms / frame_ms))][0]
            seg_end = frames[min(len(frames) - 1, i - 1 + int(padding_ms / frame_ms))][1]
            segs.append((seg_start, seg_end))
            start_idx = None
    if start_idx is not None:
        seg_start = frames[max(0, start_idx - int(padding_ms / frame_ms))][0]
        seg_end = frames[-1][1]
        segs.append((seg_start, seg_end))
    merged = []
    for s, e in segs:
        if not merged or s > merged[-1][1] + 0.01:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(float(s), float(e)) for s, e in merged]

def save_segment_wav(full_wav_path: Path, start: float, end: float, out_path: Path):
    audio, sr = sf.read(full_wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    s = int(start * sr)
    e = int(end * sr)
    seg = audio[s:e]
    sf.write(out_path, seg, sr, subtype='PCM_16')
