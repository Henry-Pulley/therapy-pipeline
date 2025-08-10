#!/usr/bin/env python
"""Simple fallback transcription using vanilla Whisper"""
import whisper
import json
from pathlib import Path

def transcribe_simple(audio_path, model_name="base"):
    """Use vanilla Whisper for transcription"""
    print(f"Loading Whisper {model_name} model...")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(str(audio_path))
    
    # Convert to WhisperX-like format
    output = {
        "segments": result["segments"],
        "text": result["text"],
        "language": result.get("language", "en")
    }
    
    return output

if __name__ == "__main__":
    audio_file = Path("templates/out/sample/sample_16k.wav")
    if audio_file.exists():
        result = transcribe_simple(audio_file)
        
        # Save output
        output_file = audio_file.parent / f"{audio_file.stem}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"Transcription saved to {output_file}")
        print(f"First 500 chars: {result['text'][:500]}")
    else:
        print(f"Audio file not found: {audio_file}")