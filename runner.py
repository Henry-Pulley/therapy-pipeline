import os, sys, json, subprocess, yaml, datetime
from pathlib import Path
from rich import print
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

# ---------- helpers ----------
def sh(cmd):
    print(f"[bold blue]$ {' '.join(cmd)}[/]")
    subprocess.run(cmd, check=True)

def preprocess_audio(src, dst, sr=16000, loudnorm=True, mono=True):
    af = []
    if loudnorm: af.append('loudnorm=I=-24:TP=-2:LRA=11')
    ff = ["ffmpeg","-y","-i",str(src)]
    if mono: ff += ["-ac","1"]
    ff += ["-ar",str(sr)]
    if af: ff += ["-af","+,".join(af)]
    ff += [str(dst)]
    sh(ff)

def run_diarization(wav_path, max_speakers=2):
    # Uses pyannote pipeline via CLI to keep it simple
    rttm = Path(wav_path).with_suffix(".rttm")
    env = os.environ.copy()
    assert env.get("HF_TOKEN"), f"Set HF_TOKEN for pyannote. Current value: {env.get('HF_TOKEN')}"
    # Minimal Python one-liner to diarize (avoids packaging a second file)
    code = f"""
from pyannote.audio import Pipeline
import os, sys
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"])
dz = pipeline("{wav_path}", num_speakers={max_speakers})
with open("{rttm}", "w") as f: f.write(dz.to_rttm())
"""
    sh(["python","-c",code])
    return rttm

def run_whisperx(wav_path, lang="en", vad=True, rttm=None):
    """Try WhisperX, fall back to vanilla Whisper if it fails"""
    outdir = Path(wav_path).parent
    base = Path(wav_path).stem
    json_path = outdir / f"{base}.json"
    
    # Try WhisperX first (known to have segfault issues with current deps)
    try:
        args = ["whisperx", str(wav_path), "--model", "base", "--language", lang,
                "--output_dir", str(outdir), "--compute_type","float32", "--no_align"]
        if rttm: args += ["--diarize","--rttm", str(rttm)]
        sh(args)
        if json_path.exists():
            return json_path
    except Exception as e:
        print(f"[yellow]WhisperX failed: {e}. Falling back to vanilla Whisper...[/]")
    
    # Fallback to vanilla Whisper
    try:
        import whisper
        print(f"[blue]Using vanilla Whisper as fallback...[/]")
        model = whisper.load_model("base")
        result = model.transcribe(str(wav_path), language=lang)
        
        # Save in WhisperX-compatible format
        output = {
            "segments": result["segments"],
            "text": result["text"],
            "language": result.get("language", lang)
        }
        json_path.write_text(json.dumps(output, indent=2))
        return json_path
    except Exception as e:
        print(f"[red]Both WhisperX and Whisper failed: {e}[/]")
        raise

def load_transcript(json_path):
    with open(json_path) as f: return json.load(f)

def extract_highlights(data, per_speaker=3):
    # simple heuristic: longest, most contentful utterances per speaker
    utts = []
    for seg in data.get("segments", []):
        spk = seg.get("speaker","UNK")
        text = seg.get("text","").strip()
        if not text: continue
        dur = float(seg["end"] - seg["start"])
        score = len(text.split()) * (1 + 0.1*dur)
        utts.append((spk, seg["start"], text, score))
    utts.sort(key=lambda x: x[3], reverse=True)
    picks = {}
    for spk, start, text, _ in utts:
        picks.setdefault(spk, [])
        if len(picks[spk]) < per_speaker:
            picks[spk].append({"speaker": spk, "time": f"{start:0.1f}s", "quote": text})
    # flatten preserve order by time
    flat = sorted([q for v in picks.values() for q in v], key=lambda q: float(q["time"][:-1]))
    return flat

def make_soap_note_ollama(context, template_path, model="llama3.1:8b", temperature=0.7):
    import jinja2, requests, json as pyjson
    tpl = jinja2.Template(Path(template_path).read_text())
    # Prompt engineering kept minimal & deterministic
    sys_prompt = (
        "You are a licensed therapist writing a concise, professional SOAP note from a session transcript. "
        "Keep it factual, no diagnosis unless stated. Avoid hallucinations. Use client’s own words when quoting."
    )
    user_prompt = pyjson.dumps(context)
    payload = {"model": model, "prompt": f"{sys_prompt}\n\nDATA:\n{user_prompt}", "options":{"temperature":temperature}}
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600, stream=False)
    r.raise_for_status()
    # Ollama streams; response may be chunked lines with {"response": "...", "done": false}
    text = ""
    for line in r.iter_lines(decode_unicode=True):
        if not line: continue
        obj = pyjson.loads(line)
        if "response" in obj: text += obj["response"]
    # We used template for structure; here we pass through LLM result into slots (or skip template if note is full)
    return text.strip()

def make_soap_note_rules(context, template_path):
    from jinja2 import Template
    tpl = Template(Path(template_path).read_text())
    # dumb but safe: S = client concerns; O = quotes; A/P = summaries from end of session turns
    subjective = context["summary"][:800]
    assessment = context["themes"][:600]
    plan = context.get("next_steps","- Continue current treatment.\n- Homework: as discussed.")
    return tpl.render(subjective=subjective, observations=context["highlights"], assessment=assessment, plan=plan)

# ---------- pipeline ----------
def process_file(wav_in, cfg):
    out_root = Path(cfg["io"]["output_dir"]); out_root.mkdir(parents=True, exist_ok=True)
    base = Path(wav_in).stem
    work = out_root / base; work.mkdir(exist_ok=True)
    pre = work / f"{base}_16k.wav"
    preprocess_audio(wav_in, pre, sr=cfg["preprocess"]["sample_rate"],
                     loudnorm=cfg["preprocess"]["loudnorm"], mono=cfg["preprocess"]["mono"])

    rttm = None
    if cfg["diarization"]["enabled"]:
        # HF_TOKEN is already loaded from .env by load_dotenv()
        rttm = run_diarization(pre, cfg["diarization"]["max_speakers"])

    tx_json = run_whisperx(pre, lang=cfg["asr"]["language"], vad=cfg["asr"]["vad"], rttm=rttm)
    data = load_transcript(tx_json)

    # Build context for notes
    highlights = extract_highlights(data)
    # lightweight summaries
    full_text = " ".join(seg.get("text","") for seg in data.get("segments",[]))
    themes = "Key themes: " + ", ".join({w for w in full_text.lower().split() if w in {"sleep","work","family","anxiety","depression","panic","trauma","relationship"}})
    context = {
        "highlights": highlights,
        "summary": full_text[:1500],
        "themes": themes if len(themes) > 12 else "Key themes: general psychotherapy concerns.",
        "next_steps": "- Review coping skills next session.\n- Assign brief homework if applicable."
    }

    if cfg["note"]["mode"] == "ollama":
        temp = cfg["note"].get("ollama_temperature", 0.7)
        note = make_soap_note_ollama(context, cfg["note"]["template"], cfg["note"]["ollama_model"], temperature=temp)
    else:
        note = make_soap_note_rules(context, cfg["note"]["template"])

    # Save artifacts
    (work / f"{base}.aligned.json").write_text(json.dumps(data, indent=2))
    (work / f"{base}.soap.md").write_text(note)
    print(f"[green]Done → {work}[/]")

if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    
    # Replace env var placeholders with actual values from environment
    for section in cfg:
        for key, val in cfg[section].items():
            if isinstance(val, str) and val in os.environ:
                env_val = os.environ[val]
                # Convert booleans and numbers
                if env_val.lower() in ('true', 'false'):
                    cfg[section][key] = env_val.lower() == 'true'
                else:
                    try:
                        cfg[section][key] = int(env_val) if '.' not in env_val else float(env_val)
                    except ValueError:
                        cfg[section][key] = env_val
    
    in_dir = Path(cfg["io"]["input_dir"])
    for p in sorted(in_dir.iterdir()):
        if p.suffix.lower() in {".wav",".mp3",".m4a",".aac",".flac"}:
            process_file(p, cfg)
