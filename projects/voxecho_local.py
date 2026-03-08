import time
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from transformers import pipeline
from TTS.api import TTS
import warnings

# Suppress verbose ML library warnings for clean output
warnings.filterwarnings("ignore")

def record_audio(filename="local_input.wav", duration=5, fs=16000):
    """Records audio from the local microphone."""
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"✅ Audio saved to {filename}")
    return filename

def run_local_pipeline():
    print("Loading local Machine Learning models... (This takes a moment)")
    
    # 1. Load ASR Model
    asr_model = whisper.load_model("base")
    
    # 2. Load Translation Model (English to Spanish)
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    
    # 3. Load TTS Model (Spanish VITS)
    # This is an actual VITS model, identical in architecture to what you will put on SageMaker
    tts_model = TTS(model_name="tts_models/es/css10/vits", progress_bar=False)

    print("\n=== Models Loaded Successfully ===")
    
    # Get Input
    input_file = record_audio(duration=4)
    
    start_time = time.time()
    
    # Step 1: Transcribe
    print("\n📝 Transcribing via Whisper...")
    transcription_result = asr_model.transcribe(input_file)
    text = transcription_result["text"].strip()
    print(f"🗣️ Recognized Text: '{text}'")
    
    # Step 2: Translate
    print("🧠 Translating via Hugging Face...")
    translation_result = translator(text)
    translated_text = translation_result[0]['translation_text']
    print(f"🌐 Translated Text: '{translated_text}'")
    
    # Step 3: Synthesize
    print("🔊 Synthesizing audio via Coqui VITS...")
    output_file = "local_dubbed_output.wav"
    tts_model.tts_to_file(text=translated_text, file_path=output_file)
    
    end_time = time.time()
    
    print("-" * 30)
    print(f"✅ Pipeline finished! Audio saved to '{output_file}'")
    print(f"⏱️ Local End-to-End Latency: {end_time - start_time:.2f} seconds")
    print("-" * 30)

if __name__ == "__main__":
    run_local_pipeline()