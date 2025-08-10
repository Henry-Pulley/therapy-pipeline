#!/usr/bin/env python3
"""Merge diarization RTTM with WhisperX transcript"""

import json
from pathlib import Path
import sys

def parse_rttm(rttm_path):
    """Parse RTTM file to get speaker segments"""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append({
                    'start': start,
                    'end': start + duration,
                    'speaker': speaker
                })
    return sorted(segments, key=lambda x: x['start'])

def find_speaker(time, speaker_segments):
    """Find speaker for a given timestamp"""
    for seg in speaker_segments:
        if seg['start'] <= time <= seg['end']:
            return seg['speaker']
    # Find nearest speaker if exact match not found
    min_dist = float('inf')
    nearest_speaker = 'SPEAKER_00'
    for seg in speaker_segments:
        dist = min(abs(time - seg['start']), abs(time - seg['end']))
        if dist < min_dist:
            min_dist = dist
            nearest_speaker = seg['speaker']
    return nearest_speaker

def merge_diarization(transcript_path, rttm_path, output_path):
    """Merge diarization info into transcript"""
    # Load transcript
    with open(transcript_path) as f:
        data = json.load(f)
    
    # Parse RTTM
    speaker_segments = parse_rttm(rttm_path)
    
    # Add speaker to each segment
    for segment in data.get('segments', []):
        # Use middle of segment for speaker assignment
        mid_time = (segment.get('start', 0) + segment.get('end', 0)) / 2
        speaker = find_speaker(mid_time, speaker_segments)
        segment['speaker'] = speaker
    
    # Save merged result
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Merged diarization → {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python merge_diarization.py <session_id>")
        sys.exit(1)
    
    session = sys.argv[1]
    base_dir = Path("templates/out") / session
    
    transcript = base_dir / "asr.aligned.json"
    rttm = base_dir / "diarization.rttm"
    output = base_dir / "asr.aligned.json"
    
    if not transcript.exists():
        print(f"Error: Transcript not found: {transcript}")
        sys.exit(1)
    if not rttm.exists():
        print(f"Error: RTTM not found: {rttm}")
        sys.exit(1)
    
    merge_diarization(transcript, rttm, output)