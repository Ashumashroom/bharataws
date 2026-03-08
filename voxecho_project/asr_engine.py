import whisper
import warnings

warnings.filterwarnings("ignore")

class ASREngine:
    def __init__(self, model_name="base"):
        print(f"Loading ASR Model ({model_name})...")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        """Converts audio file to text."""
        print("\n📝 Transcribing via Whisper...")
        result = self.model.transcribe(audio_path)
        text = result["text"].strip()
        print(f"🗣️ Recognized Text: '{text}'")
        return text