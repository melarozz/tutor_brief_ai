import wave
import json
from vosk import Model, KaldiRecognizer

class Transcriber:
    def __init__(self, model_dir: str):
        self.model = Model(model_dir)

    def transcribe_segment(self, wav_path: str) -> str:
        wf = wave.open(wav_path, "rb")
        rec = KaldiRecognizer(self.model, wf.getframerate())
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
