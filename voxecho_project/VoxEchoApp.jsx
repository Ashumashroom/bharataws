import React, { useState, useRef } from 'react';

export default function VoxEchoApp() {
  const [file, setFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [resultUrl, setResultUrl] = useState(null);
  const [mode, setMode] = useState('audio'); // 'audio' or 'video'
  const [statusMessage, setStatusMessage] = useState('');
  
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResultUrl(null); // Clear previous results
    }
  };

  const processFile = async () => {
    if (!file) {
      alert('Please select a file first!');
      return;
    }

    setIsProcessing(true);
    setStatusMessage('⚙️ AI is processing... (Transcribing ➔ Translating ➔ Synthesizing)');
    
    const formData = new FormData();
    formData.append('file', file);

    // Determine the correct backend endpoint based on the selected mode
    const endpoint = mode === 'audio' ? 'http://localhost:8000/upload/audio' : 'http://localhost:8000/upload/video';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Failed to process media on the server.');

      // Convert the response to a blob so we can play it in the browser
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      
      setResultUrl(url);
      setStatusMessage('✅ Processing Complete!');
    } catch (error) {
      console.error('Error:', error);
      setStatusMessage('❌ Error processing file. Make sure your FastAPI server is running!');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '40px', fontFamily: 'sans-serif' }}>
      <div style={{ background: 'white', padding: '30px', borderRadius: '12px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}>
        
        <h1 style={{ textAlign: 'center', color: '#2563eb' }}>🎙️ VoxEcho AI</h1>
        <p style={{ textAlign: 'center', color: '#666', marginBottom: '30px' }}>
          Upload English media and get it dubbed in Spanish.
        </p>

        {/* Mode Selector */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '20px' }}>
          <button 
            onClick={() => { setMode('audio'); setFile(null); setResultUrl(null); }}
            style={{ padding: '10px 20px', borderRadius: '8px', border: mode === 'audio' ? '2px solid #2563eb' : '1px solid #ccc', background: mode === 'audio' ? '#eff6ff' : 'white', cursor: 'pointer', fontWeight: 'bold' }}
          >
            🎵 Audio Dubbing
          </button>
          <button 
            onClick={() => { setMode('video'); setFile(null); setResultUrl(null); }}
            style={{ padding: '10px 20px', borderRadius: '8px', border: mode === 'video' ? '2px solid #2563eb' : '1px solid #ccc', background: mode === 'video' ? '#eff6ff' : 'white', cursor: 'pointer', fontWeight: 'bold' }}
          >
            🎬 Video Dubbing
          </button>
        </div>

        {/* Upload Area */}
        <div style={{ border: '2px dashed #cbd5e1', padding: '40px', textAlign: 'center', borderRadius: '8px', backgroundColor: '#f8fafc' }}>
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange} 
            accept={mode === 'audio' ? "audio/*" : "video/mp4,video/*"}
            style={{ marginBottom: '20px' }}
          />
          <br />
          <button 
            onClick={processFile} 
            disabled={isProcessing || !file}
            style={{ backgroundColor: isProcessing ? '#94a3b8' : '#2563eb', color: 'white', padding: '12px 24px', borderRadius: '6px', border: 'none', fontWeight: 'bold', cursor: isProcessing ? 'not-allowed' : 'pointer', fontSize: '16px' }}
          >
            {isProcessing ? 'Processing...' : `Dub ${mode === 'audio' ? 'Audio' : 'Video'}`}
          </button>
        </div>

        {/* Status Message */}
        {statusMessage && (
          <div style={{ marginTop: '20px', textAlign: 'center', fontWeight: 'bold', color: isProcessing ? '#d97706' : '#16a34a' }}>
            {statusMessage}
          </div>
        )}

        {/* Result Player */}
        {resultUrl && (
          <div style={{ marginTop: '30px', padding: '20px', backgroundColor: '#f0fdf4', borderRadius: '8px', border: '1px solid #bbf7d0', textAlign: 'center' }}>
            <h3 style={{ color: '#166534', marginTop: 0 }}>✨ AI Dubbing Result</h3>
            
            {mode === 'audio' ? (
              <audio controls src={resultUrl} style={{ width: '100%', marginTop: '10px' }} />
            ) : (
              <video controls src={resultUrl} style={{ width: '100%', maxHeight: '400px', marginTop: '10px', borderRadius: '8px' }} />
            )}
            
            <br />
            <a href={resultUrl} download={`dubbed_${file.name}`}>
              <button style={{ marginTop: '15px', backgroundColor: '#16a34a', color: 'white', padding: '10px 20px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}>
                ⬇️ Download Result
              </button>
            </a>
          </div>
        )}

      </div>
    </div>
  );
}