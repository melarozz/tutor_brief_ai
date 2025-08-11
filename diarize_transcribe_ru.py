import os
import sys
import json
import wave
import math
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import soundfile as sf
from vosk import Model, KaldiRecognizer
import webrtcvad
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
from pydub import AudioSegment
import requests

# --- helpers ----------------------------------------------------

def extract_audio(video_path, out_wav_path, sample_rate=16000):
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ar", str(sample_rate), "-ac", "1",
        "-vn", "-f", "wav", str(out_wav_path)
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def read_wav(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr

def frame_generator(frame_ms, audio, sample_rate):
    bytes_per_sample = 2
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

def save_segment_wav(full_wav_path, start, end, out_path):
    audio, sr = sf.read(full_wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    s = int(start * sr)
    e = int(end * sr)
    seg = audio[s:e]
    sf.write(out_path, seg, sr, subtype='PCM_16')

# --- data --------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = None
    embedding: np.ndarray = None

# --- core --------------------------------------------------------

def transcribe_segment_with_vosk(model, wav_path):
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += (res.get("text", "") + " ")
    res = json.loads(rec.FinalResult())
    text += res.get("text", "")
    return text.strip()

def cluster_speakers(segments, min_speakers=1, max_speakers=8):
    embeddings = np.vstack([s.embedding for s in segments])
    best_k = min(max_speakers, len(segments))
    if best_k <= 1:
        labels = np.zeros(len(segments), dtype=int)
    else:
        k = min(best_k, max(1, int(math.sqrt(len(segments)))))
        k = max(min_speakers, k)
        cl = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        labels = cl.fit_predict(embeddings)
    return labels

def merge_adjacent_segments(segments):
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if seg.speaker == prev.speaker and abs(seg.start - prev.end) < 0.5:
            prev.end = seg.end
            prev.text = (prev.text + " " + seg.text).strip()
        else:
            out.append(seg)
    return out

def write_srt(segments, out_srt):
    def fmt_time(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    with open(out_srt, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{fmt_time(s.start)} --> {fmt_time(s.end)}\n")
            f.write(f"{s.speaker}: {s.text}\n\n")

# --- mistral -----------------------------------------------------

def send_to_mistral(transcript_json):
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MISTRAL_API_KEY environment variable")

    prompt = (
        "Ты — умный и опытный помощник.\n"
        "Твоя задача: составить краткий, но содержательный конспект урока.\n"
        "Главный акцент должен быть на то, что сделал(и) ученик(и) и на ученика в целом\n"
        "Действия учителя нужно максимально сократить в конспекте или вообще убрать, если это не важно в контексте фокуса на ученика\n"
        "Формат:\n"
        "Тема 1 — Важный момент 1; Важный момент 2; ...\n"
        "Тема 2 — Важный момент 1; Важный момент 2; ...\n"
        "Требования:\n"
        "- Игнорируй шум, шутки и не относящиеся к теме реплики.\n"
        "- Определи, где говорит учитель, а где ученики, даже если это не всегда явно.\n"
        "- Группируй идеи по темам, а внутри тем — по смыслу.\n"
        "- Не пиши лишнего, только суть.\n\n"
        f"Вот транскрипт в JSON:\n{json.dumps(transcript_json, ensure_ascii=False)}"
    )

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-large-latest",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# --- main --------------------------------------------------------

def main(video_path, vosk_model_dir, out_json):
    video_path = Path(video_path)
    out_json = Path(out_json)
    tmpdir = Path(tempfile.mkdtemp(prefix="diarize_"))
    audio_wav = tmpdir / "audio.wav"
    print("Extracting audio to", audio_wav)
    extract_audio(video_path, audio_wav)

    print("Loading audio for VAD...")
    audio, sr = read_wav(audio_wav)
    if sr != 16000:
        raise RuntimeError("Audio sample rate must be 16000")

    print("Running VAD...")
    speech_segs = vad_collector(audio, sr, aggressiveness=2, frame_ms=30, padding_ms=300)
    print(f"Detected {len(speech_segs)} speech segments.")

    print("Loading Vosk model...")
    model = Model(str(vosk_model_dir))
    encoder = VoiceEncoder()

    segments = []
    for idx, (s, e) in enumerate(speech_segs):
        seg_wav = tmpdir / f"seg_{idx:04d}.wav"
        save_segment_wav(audio_wav, s, e, seg_wav)
        text = transcribe_segment_with_vosk(model, str(seg_wav))
        try:
            wav_f = preprocess_wav(str(seg_wav))
            emb = encoder.embed_utterance(wav_f)
        except Exception as ex:
            print("Embedding error:", ex)
            emb = np.zeros((encoder.embedding_size,))
        seg = Segment(start=s, end=e, text=text, embedding=emb)
        segments.append(seg)

    if not segments:
        print("No speech segments found.")
        return

    print("Clustering speakers...")
    labels = cluster_speakers(segments)
    for seg, lab in zip(segments, labels):
        seg.speaker = f"Speaker_{int(lab)}"

    print("Merging segments...")
    merged = merge_adjacent_segments(segments)

    output_data = [
        {
            "speaker": s.speaker,
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "text": s.text
        }
        for s in merged
    ]

    out_json.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
    out_srt = out_json.with_suffix(".srt")
    write_srt([Segment(**seg) for seg in output_data], out_srt)
    print("Transcript saved to", out_json)

    print("Sending to Mistral for summary...")
    summary = send_to_mistral(output_data)
    summary_path = out_json.with_suffix(".summary.txt")
    summary_path.write_text(summary, encoding="utf-8")
    print("Summary saved to", summary_path)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python diarize_transcribe_ru.py <video_path> <vosk_model_dir> <output_json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
