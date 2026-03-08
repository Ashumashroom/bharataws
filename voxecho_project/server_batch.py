import os
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from asr_engine import ASREngine
from translator import TranslatorEngine
from tts_engine import TTSEngine

app = FastAPI()

print("=== Loading VoxEcho Models (Batch Mode) ===")
asr = ASREngine()
translator = TranslatorEngine()
tts = TTSEngine()
print("=== Models Ready! ===")

@app.post("/upload/audio")
async def process_audio(file: UploadFile = File(...)):
    """Method 1: Process an Audio File"""
    print(f"\n📥 Received Audio: {file.filename}")
    
    input_path = f"temp_input_{file.filename}"
    output_path = f"dubbed_{file.filename}.wav"
    
    # Save the uploaded file to disk
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
        
    # 1. Transcribe
    text = asr.transcribe(input_path)
    
    # 2. Translate
    translated_text = translator.translate(text)
    
    # 3. Synthesize
    tts.synthesize(translated_text, output_path)
    
    # Cleanup input
    if os.path.exists(input_path): os.remove(input_path)
    
    # Send the file back to the browser
    return FileResponse(output_path, media_type="audio/wav", filename=f"dubbed_{file.filename}")

@app.post("/upload/video")
async def process_video(file: UploadFile = File(...)):
    """Method 2: Process a Video File (Voice Replacement)"""
    print(f"\n📥 Received Video: {file.filename}")
    
    input_video = f"temp_input_{file.filename}"
    ai_audio = f"temp_ai_audio.wav"
    final_video = f"dubbed_video.mp4"
    
    with open(input_video, "wb") as buffer:
        buffer.write(await file.read())
        
    # 1. Whisper can transcribe directly from an mp4 file!
    print("🎬 Extracting and Transcribing video audio...")
    text = asr.transcribe(input_video)
    
    # 2. Translate
    translated_text = translator.translate(text)
    
    # 3. Synthesize the new audio track
    tts.synthesize(translated_text, ai_audio)
    
    # 4. Merge the new audio onto the original video using FFmpeg
    # This strips the original audio (-map 0:v:0) and adds the AI audio (-map 1:a:0)
    print("🎬 Merging AI Audio with Original Video...")
    merge_cmd = [
        "ffmpeg", "-y", 
        "-i", input_video,      # Input 0: Original Video
        "-i", ai_audio,         # Input 1: AI Dubbed Audio
        "-c:v", "copy",         # Copy video frames exactly as they are
        "-map", "0:v:0",        # Take video from Input 0
        "-map", "1:a:0",        # Take audio from Input 1
        final_video
    ]
    subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup temp files
    if os.path.exists(input_video): os.remove(input_video)
    if os.path.exists(ai_audio): os.remove(ai_audio)
    
    return FileResponse(final_video, media_type="video/mp4", filename="dubbed_video.mp4")