from pathlib import Path
import tempfile
import json
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

from .audio import extract_audio, read_wav, vad_collector, save_segment_wav
from .segment import Segment
from .transcriber import Transcriber
from .clustering import cluster_speakers, merge_adjacent_segments
from .summarizer import send_to_mistral

class Diarizer:
    def __init__(self, vosk_model_dir: str):
        self.vosk_model_dir = Path(vosk_model_dir)
        self.encoder = VoiceEncoder()
        self.transcriber = Transcriber(str(self.vosk_model_dir))

    def write_srt(self, segments, out_srt):
        def fmt_time(t):
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        with open(out_srt, "w", encoding="utf-8") as f:
            for i, s in enumerate(segments, start=1):
                f.write(f"{i}\n")
                f.write(f"{fmt_time(s.start)} --> {fmt_time(s.end)}\n")
                f.write(f"{s.speaker}: {s.text}\n\n")

    def diarize(self, video_path: str, output_json: str):
        video_path = Path(video_path)
        output_json = Path(output_json)
        tmpdir = Path(tempfile.mkdtemp(prefix="diarize_"))
        audio_wav = tmpdir / "audio.wav"
        print("Extracting audio...")
        extract_audio(video_path, audio_wav)

        print("Reading audio for VAD...")
        audio, sr = read_wav(audio_wav)
        if sr != 16000:
            raise RuntimeError("Audio sample rate must be 16000")

        print("Running VAD...")
        speech_segs = vad_collector(audio, sr, aggressiveness=2, frame_ms=30, padding_ms=300)
        print(f"Detected {len(speech_segs)} speech segments.")

        segments = []
        for idx, (s, e) in enumerate(speech_segs):
            seg_wav = tmpdir / f"seg_{idx:04d}.wav"
            save_segment_wav(audio_wav, s, e, seg_wav)
            text = self.transcriber.transcribe_segment(str(seg_wav))
            try:
                wav_f = preprocess_wav(str(seg_wav))
                emb = self.encoder.embed_utterance(wav_f)
            except Exception as ex:
                print("Embedding error:", ex)
                emb = np.zeros((self.encoder.embedding_size,))
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

        output_json.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        out_srt = output_json.with_suffix(".srt")
        self.write_srt(merged, out_srt)
        print(f"Transcript saved to {output_json}")

        print("Sending to Mistral for summary...")
        summary = send_to_mistral(output_data)
        summary_path = output_json.with_suffix(".summary.txt")
        summary_path.write_text(summary, encoding="utf-8")
        print(f"Summary saved to {summary_path}")
