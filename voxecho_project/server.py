from fastapi import FastAPI, WebSocket
from asr_engine import ASREngine
from translator import TranslatorEngine
from tts_engine import TTSEngine
import os

app = FastAPI()

# Load models once when server starts
asr = ASREngine()
translator = TranslatorEngine()
tts = TTSEngine()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to VoxEcho Stream!")
    
    chunk_counter = 0
    try:
        while True:
            # 1. Receive audio chunk bytes from the frontend
            audio_bytes = await websocket.receive_bytes()
            chunk_filename = f"server_input_{chunk_counter}.wav"
            
            with open(chunk_filename, "wb") as f:
                f.write(audio_bytes)
                
            # 2. Run your existing pipeline
            text = asr.transcribe(chunk_filename)
            if text and len(text) > 2:
                translated_text = translator.translate(text)
                output_filename = tts.synthesize(translated_text, f"server_output_{chunk_counter}.wav")
                
                # 3. Send the dubbed audio bytes back to the frontend
                with open(output_filename, "rb") as f:
                    dubbed_bytes = f.read()
                await websocket.send_bytes(dubbed_bytes)
                
            chunk_counter += 1
            
    except Exception as e:
        print(f"Connection closed: {e}")