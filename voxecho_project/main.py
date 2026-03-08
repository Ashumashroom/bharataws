import time
from audio_io import record_audio
from asr_engine import ASREngine
from translator import TranslatorEngine
from tts_engine import TTSEngine

def main():
    print("=== Initializing VoxEcho Pipeline ===")
    
    # 1. Initialize Classes (This loads all models into memory)
    # This might take a moment, but it only happens once!
    asr = ASREngine()
    translator = TranslatorEngine()
    tts = TTSEngine()
    
    print("\n=== Models Loaded Successfully ===")
    
    # 2. Get Input
    input_file = record_audio(duration=4)
    
    # 3. Start processing pipeline and timing it
    start_time = time.time()
    
    text = asr.transcribe(input_file)
    translated_text = translator.translate(text)
    output_file = tts.synthesize(translated_text)
    
    end_time = time.time()
    
    # 4. Output Results
    print("-" * 30)
    print(f"✅ Pipeline finished! Audio saved to '{output_file}'")
    print(f"⏱️ Local End-to-End Latency: {end_time - start_time:.2f} seconds")
    print("-" * 30)

if __name__ == "__main__":
    main()