import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="local_input.wav", duration=4, fs=16000):
    """Records audio from the local microphone."""
    print(f"\n🎤 Recording for {duration} seconds... Speak now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"✅ Audio saved to {filename}")
    return filename