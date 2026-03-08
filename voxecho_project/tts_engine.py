from TTS.api import TTS
import warnings

warnings.filterwarnings("ignore")

class TTSEngine:
    def __init__(self, model_name="tts_models/es/css10/vits"):
        print(f"Loading TTS Model ({model_name})...")
        self.model = TTS(model_name=model_name, progress_bar=False)

    def synthesize(self, text, output_path="local_dubbed_output.wav"):
        """Generates audio file from text."""
        print("🔊 Synthesizing audio via Coqui VITS...")
        self.model.tts_to_file(text=text, file_path=output_path)
        return output_path