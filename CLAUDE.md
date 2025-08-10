# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a therapy session audio processing pipeline that transcribes therapy sessions and generates SOAP (Subjective, Objective, Assessment, Plan) notes. The pipeline uses WhisperX for ASR, optional Pyannote for speaker diarization, and either Ollama LLM or rule-based methods for note generation.

## Core Architecture

### Main Components

1. **runner.py** - Main pipeline orchestrator that processes audio files through multiple stages:
   - Audio preprocessing (FFmpeg normalization, resampling to 16kHz)
   - Optional speaker diarization (Pyannote)
   - Speech transcription with alignment (WhisperX)
   - Context extraction and highlight selection
   - SOAP note generation (LLM or rule-based)

2. **config.yaml** - Central configuration with environment variable placeholders:
   - ASR settings (Whisper model, language, VAD)
   - Diarization settings (max speakers, HF token)
   - Audio preprocessing (sample rate, mono, loudness normalization)
   - Note generation mode (Ollama LLM vs rule-based)
   - I/O directories

3. **templates/soap.j2** - Jinja2 template for SOAP note structure

### Processing Flow

1. Audio files placed in `templates/in/` directory
2. Pipeline preprocesses audio to 16kHz mono WAV
3. Optional diarization generates speaker labels (RTTM format)
4. WhisperX transcribes with word-level alignment
5. System extracts key quotes and themes from transcript
6. Generates SOAP note using either:
   - Ollama LLM with prompt engineering
   - Rule-based template filling
7. Outputs saved to `templates/out/{filename}/` with:
   - Preprocessed audio
   - Aligned transcript JSON
   - Generated SOAP note markdown

## Development Commands

### Running the Pipeline
```bash
python runner.py
```

### Required Environment Variables
```bash
export HF_TOKEN="your_huggingface_token"  # Required for Pyannote diarization
```

### Dependencies Installation
The project requires:
- WhisperX (with alignment models)
- Pyannote.audio (for diarization)
- FFmpeg (for audio processing)
- Ollama (if using LLM mode)
- Python packages: pyyaml, rich, soundfile, jinja2, requests

### Testing Single Files
Place audio files (wav, mp3, m4a, aac, flac) in `templates/in/` and run `python runner.py`

## Configuration Notes

- Environment variables in config.yaml are placeholders (e.g., VAD, DIARIZATION_ENABLED)
- Ollama must be running locally on port 11434 if using LLM mode
- WhisperX uses large-v3 model by default with float32 compute
- Diarization requires HuggingFace token for pyannote/speaker-diarization-3.1 model

## Key Functions

- `preprocess_audio()` - FFmpeg-based audio normalization and format conversion
- `run_diarization()` - Executes Pyannote pipeline via inline Python
- `run_whisperx()` - CLI wrapper for WhisperX with VAD and alignment
- `extract_highlights()` - Heuristic selection of important utterances
- `make_soap_note_ollama()` - LLM-based note generation with Ollama API
- `make_soap_note_rules()` - Template-based fallback note generation